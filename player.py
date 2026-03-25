"""
player.py — Streaming Playback for HFPAC
=========================================
Decodes and plays a .hfpac file in real-time using sounddevice.

Design: streaming, not load-everything-first
---------------------------------------------
Rather than decoding the entire file into RAM before playing, a background
thread decodes frames ahead of time and pushes them onto a small queue.
The sounddevice callback pulls from that queue and sends audio to the
speakers. Memory usage stays constant regardless of file length.

Architecture
------------
                         background thread
                         ┌──────────────────────────┐
  .hfpac frames ────────►│ Huffman → LPC → float32  │
                         │ pushes blocks onto queue  │
                         └──────────┬───────────────┘
                                    │  queue (max 48 blocks)
                         ┌──────────▼───────────────┐
                         │  sounddevice callback     │──► speakers
                         │  pulls one block per tick │
                         └──────────────────────────┘

Controls (printed at start)
---------------------------
  P or SPACE — pause / resume
  + / -      — volume up / down (10% steps)
  Q          — quit

Dependencies
------------
  pip install sounddevice numpy
  Windows / macOS: sounddevice ships with PortAudio
  Linux:           sudo apt install libportaudio2
"""

import queue
import sys
import threading
import time
import logging

log = logging.getLogger(__name__)

import numpy as np
import sounddevice as sd

from hfpac_format import (read_hfpac, STEREO_MID_SIDE, ENTROPY_RICE, LPC_INTEGER,
                          FRAME_SYNC, FRAME_CONT, FRAME_SILENCE, Metadata,
                          display_version)
from huffman import build_code_table, decode as huffman_decode, deserialise_tree
from lpc import decode_frame, decode_frame_int, make_prior_history


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUEUE_MAXSIZE = 16  # Reduced to decrease seek/track-change latency
QUEUE_PREFILL = 4
VOLUME_STEP   = 0.10


def _fmt_version(version: int) -> str:
    return display_version(version)


# ---------------------------------------------------------------------------
# Internal: decode one EncodedFrame → float32 numpy array
# ---------------------------------------------------------------------------

def _decode_block(frame, frame_size: int, scale: float,
                  tree=None, entropy_mode: int = 0,
                  lpc_mode: int = 0,
                  prior_history=None) -> np.ndarray:
    """
    Decode one EncodedFrame → float32 array normalised to [-1, 1].

    For v6 (FRAME_SILENCE): returns a block of constant silence_value/scale.
    For FRAME_SYNC/CONT:    decodes residuals then reconstructs via LPC.
    prior_history is passed to decode_frame_int for CONT frames.
    """
    # Silence frame — output constant samples
    if frame.frame_type == FRAME_SILENCE:
        val = float(frame.silence_value) / scale
        return np.full(frame_size, val, dtype=np.float32)

    if entropy_mode == ENTROPY_RICE:
        from rice import decode as rice_decode
        residuals = rice_decode(
            frame.rice_payload, frame.rice_k, frame.num_bits, frame_size
        )
    else:
        if tree is None:
            if isinstance(frame.huffman_tree, bytes):
                tree, _ = deserialise_tree(frame.huffman_tree)
            else:
                tree = frame.huffman_tree
        residuals = huffman_decode(
            frame.huffman_payload, tree, frame.num_bits, frame_size
        )

    res_i32 = np.array(residuals, dtype=np.int32)
    if lpc_mode == LPC_INTEGER and frame.lpc_coeffs_int is not None:
        samples = decode_frame_int(res_i32, frame.lpc_coeffs_int,
                                   frame.lpc_precision,
                                   prior_history=prior_history)
    else:
        samples = decode_frame(res_i32, frame.lpc_coeffs)

    return (samples / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class HFPACPlayer:
    """
    Streaming .hfpac player with pause, volume, and quit controls.

    Usage:
        player = HFPACPlayer("my_song.hfpac")
        player.play()   # blocks until track ends or user quits
    """

    def __init__(self, path: str, volume: float = 1.0, gui_mode: bool = False, preloaded_data=None, progress_callback=None):
        self.path   = path
        self.volume = max(0.0, min(1.0, volume))
        self.gui_mode = gui_mode

        if preloaded_data:
            self._header, self._frames = preloaded_data
        else:
            if not self.gui_mode:
                log.info(f"Loading {path} ...")
            self._header, self._frames = read_hfpac(path, progress_callback=progress_callback)
        h = self._header

        self._frames_per_ch = h.num_frames // h.channels
        self._scale         = float(2 ** (h.bit_depth - 1))
        self._frame_size    = h.frame_size
        self._channels      = h.channels
        self._sample_rate   = h.sample_rate
        self._duration      = self._frames_per_ch * h.frame_size / h.sample_rate

        self._mid_side     = (
            getattr(h, 'stereo_mode', 0) == STEREO_MID_SIDE and h.channels == 2
        )
        self._entropy_mode = getattr(h, 'entropy_mode', 0)
        self._lpc_mode     = getattr(h, 'lpc_mode', 0)
        self._file_version = getattr(h, 'version', 7)

        # Per-channel LPC history for v6 CONT frame carry-over.
        # None = cold start (reset on SYNC or at playback start).
        self._ch_history   = [None] * h.channels

        # Playback state
        self._current_frame = 0           # index into one channel's frames
        self._paused        = False
        self._stopped       = False
        self._underrun      = False

        # Thread-safe audio queue — items are float32 ndarrays or None (sentinel)
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

        # Lock protecting _current_frame and _paused
        self._lock = threading.Lock()

        # EQ state
        try:
            from eq import get_eq_coeffs
            self.eq_enabled = True
            self.eq_gains = [0.0] * 10
            self._eq_coeffs = get_eq_coeffs(self._sample_rate, self.eq_gains)
            self._eq_state = np.zeros((self._channels, 10, 2), dtype=np.float64)
            self._eq_lock = threading.Lock()
        except ImportError:
            self.eq_enabled = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Start playback. Blocks until the track ends or the user quits."""
        if not self.gui_mode:
            self._print_info()
            self._print_controls()

        # Start background decode thread
        reader = threading.Thread(target=self._reader_loop, daemon=True)
        reader.start()

        # Pre-fill the queue before opening the audio device
        # to avoid an underrun right at the start
        target = min(QUEUE_PREFILL, self._frames_per_ch)
        while self._q.qsize() < target and reader.is_alive():
            time.sleep(0.01)

        try:
            with sd.OutputStream(
                samplerate = self._sample_rate,
                channels   = self._channels,
                dtype      = "float32",
                blocksize  = self._frame_size,
                latency    = "low",
                callback   = self._audio_callback,
            ):
                if self.gui_mode:
                    while not self._stopped:
                        time.sleep(0.05)
                else:
                    self._control_loop()

        except KeyboardInterrupt:
            pass
        finally:
            self._stopped = True
            if not self.gui_mode:
                log.info("[HFPAC] Playback stopped.")

    def toggle_pause(self) -> None:
        with self._lock:
            self._paused = not self._paused

    def set_eq_gains(self, gains) -> None:
        if self.eq_enabled:
            with self._eq_lock:
                self.eq_gains = gains
                from eq import get_eq_coeffs
                self._eq_coeffs = get_eq_coeffs(self._sample_rate, self.eq_gains)

    def stop(self) -> None:
        self._stopped = True

    def seek(self, seconds: float) -> None:
        """
        Seek to a position in the track.

        Finds the seek-table entry nearest to (but not past) the requested
        time, resets _current_frame to that point, and flushes the audio
        queue so no stale audio is played.

        For files without a seek table (v2–v4) the seek is approximate:
        we jump to the nearest frame boundary and decode forward from there,
        which is instant but may cause a brief glitch at the jump point.
        """
        seconds    = max(0.0, min(seconds, self._duration))
        target_idx = int(seconds * self._sample_rate / self._frame_size)
        target_idx = min(target_idx, self._frames_per_ch - 1)

        # Use seek table if available to jump to the nearest safe point
        seek_table = getattr(self._header, 'seek_table', None)
        if seek_table:
            # Seek table entries are global (cover all channels interleaved).
            # Convert to per-channel index for comparison.
            frames_per_ch = self._frames_per_ch
            best_frame    = 0
            for fi, _ in seek_table:
                per_ch = fi % frames_per_ch   # entries are at multiples of seek_interval
                if per_ch <= target_idx:
                    best_frame = max(best_frame, per_ch)
            target_idx = best_frame

        # Pre-pass: Rebuild exact history per channel to prevent sync drift
        # in existing v6 files where silence frames misaligned the channels.
        new_history = [None] * self._channels
        is_v6       = self._file_version >= 8
        
        if is_v6 and getattr(self, '_frames', None) is not None:
            for ch in range(self._channels):
                sync_idx = target_idx
                # Trace back to find the actual last FRAME_SYNC for THIS channel
                while sync_idx > 0:
                    f = self._frames[ch * self._frames_per_ch + sync_idx]
                    if getattr(f, 'frame_type', FRAME_SYNC) == FRAME_SYNC:
                        break
                    sync_idx -= 1
                
                # Decode forward from sync_idx to target_idx - 1 to build the exact history array
                hist = None
                for idx in range(sync_idx, target_idx):
                    frame = self._frames[ch * self._frames_per_ch + idx]
                    ftype = getattr(frame, 'frame_type', FRAME_SYNC)
                    
                    if ftype == FRAME_SYNC:
                        hist = None
                    elif ftype == FRAME_CONT and hist is None:
                        continue # Malformed/unrecoverable stream history jump point
                        
                    if ftype == FRAME_SILENCE:
                        if self._lpc_mode == LPC_INTEGER:
                            order = (len(frame.lpc_coeffs_int) if frame.lpc_coeffs_int is not None 
                                     else len(frame.lpc_coeffs))
                            hist = np.full(max(order, 1), getattr(frame, 'silence_value', 0), dtype=np.int64)
                        continue
                        
                    # Standard decoded frame
                    block = _decode_block(
                        frame, self._frame_size, self._scale,
                        tree=None, entropy_mode=self._entropy_mode,
                        lpc_mode=self._lpc_mode, prior_history=hist
                    )
                    
                    if self._lpc_mode == LPC_INTEGER and getattr(frame, 'lpc_coeffs_int', None) is not None:
                        samples_int = (block * self._scale).astype(np.float64)
                        hist = make_prior_history(samples_int, len(frame.lpc_coeffs_int))
                        
                new_history[ch] = hist

        # Set the target frame under the lock first, so the reader sees the
        # new position before it can increment _current_frame past it.
        with self._lock:
            self._current_frame = target_idx
            self._ch_history    = new_history
            if getattr(self, 'eq_enabled', False):
                with self._eq_lock:
                    self._eq_state.fill(0.0)

        # Flush stale audio twice: once before setting the frame pointer
        # (catches blocks already in the queue) and once after (catches any
        # block the reader finished and queued in the narrow window between
        # the lock release and the history reset).
        def _flush():
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break

        _flush()
        import time as _time; _time.sleep(0.005)   # yield — let reader finish current put
        _flush()

    # ------------------------------------------------------------------
    # Background reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """
        Runs on a daemon thread. Decodes frames ahead of playback and
        pushes interleaved float32 blocks onto the queue.

        v6 (file_version ≥ 8):
          - FRAME_SYNC    — reset per-channel history, decode normally
          - FRAME_CONT    — decode with prior history, update history
          - FRAME_SILENCE — output constant block, update history
        v2–v5.1:
          - No frame types; history is always None (cold start every frame)
        """
        cached_tree_bytes = None
        cached_tree       = None
        is_v6             = self._file_version >= 8

        while not self._stopped:
            with self._lock:
                idx    = self._current_frame
                paused = self._paused

            if paused:
                time.sleep(0.02)
                continue

            if idx >= self._frames_per_ch:
                try:
                    self._q.put(None, timeout=1.0)
                except queue.Full:
                    pass
                break

            channels_pcm = []
            for ch in range(self._channels):
                global_idx = ch * self._frames_per_ch + idx
                frame      = self._frames[global_idx]

                if is_v6:
                    # ── v6: handle frame types ────────────────────────
                    if frame.frame_type == FRAME_SILENCE:
                        block = _decode_block(
                            frame, self._frame_size, self._scale,
                            entropy_mode=self._entropy_mode,
                            lpc_mode=self._lpc_mode,
                        )
                        # History after silence: all samples = silence_value
                        if self._lpc_mode == LPC_INTEGER:
                            order = (len(frame.lpc_coeffs_int)
                                     if frame.lpc_coeffs_int is not None
                                     else len(frame.lpc_coeffs))
                            self._ch_history[ch] = np.full(
                                max(order, 1), frame.silence_value, dtype=np.int64)
                        channels_pcm.append(block)
                        continue

                    if frame.frame_type == FRAME_SYNC:
                        self._ch_history[ch] = None  # cold start

                    # Guard: never decode a CONT frame with uninitialised
                    # history — that produces integer overflow → loud static.
                    # This can happen when the race condition between seek()
                    # and the reader loop causes the reader to land on a CONT
                    # frame before it has seen a SYNC frame.  Skip forward
                    # until the next natural SYNC frame, which will reset
                    # history cleanly.  Output silence for the skipped frame
                    # so the audio stream stays gapless.
                    if (frame.frame_type == FRAME_CONT
                            and self._ch_history[ch] is None):
                        channels_pcm.append(
                            np.zeros(self._frame_size, dtype=np.float32))
                        continue

                    hist = self._ch_history[ch]
                else:
                    hist = None  # pre-v6: no history

                # ── Huffman tree caching (v2–v5.1 only) ──────────────
                if not is_v6:
                    if frame.huffman_tree is not cached_tree_bytes:
                        cached_tree_bytes = frame.huffman_tree
                        if isinstance(cached_tree_bytes, bytes):
                            cached_tree, _ = deserialise_tree(cached_tree_bytes)
                        else:
                            cached_tree = cached_tree_bytes
                    tree = cached_tree
                else:
                    tree = None  # v6 always stores per-frame tree

                block = _decode_block(
                    frame, self._frame_size, self._scale,
                    tree, self._entropy_mode, self._lpc_mode,
                    prior_history=hist,
                )

                # Update per-channel history for v6 CONT/SYNC frames
                if is_v6 and self._lpc_mode == LPC_INTEGER \
                        and frame.lpc_coeffs_int is not None:
                    # Convert float32 block back to integer scale for history
                    samples_int = (block * self._scale).astype(np.float64)
                    self._ch_history[ch] = make_prior_history(
                        samples_int, len(frame.lpc_coeffs_int))

                channels_pcm.append(block)

            # Reverse mid-side transform
            if self._mid_side:
                mid, side = channels_pcm[0], channels_pcm[1]
                if self._lpc_mode == LPC_INTEGER:
                    s = self._scale
                    mid_i  = np.round(mid  * s).astype(np.int64)
                    side_i = np.round(side * s).astype(np.int64)
                    channels_pcm[0] = ((mid_i + ((side_i + 1) >> 1)) / s).astype(np.float32)
                    channels_pcm[1] = ((mid_i - (side_i >> 1))       / s).astype(np.float32)
                else:
                    channels_pcm[0] = mid + side
                    channels_pcm[1] = mid - side

            if self._channels == 1:
                interleaved = channels_pcm[0].reshape(-1, 1)
            else:
                interleaved = np.stack(channels_pcm, axis=1)

            # Trim trailing padding if it exists and we're on the last frame
            pad = getattr(self._header, "trailing_padding", 0)
            if pad > 0 and idx == self._frames_per_ch - 1:
                interleaved = interleaved[:-pad]

            if getattr(self, 'eq_enabled', False):
                from eq import process_stereo_eq
                with self._eq_lock:
                    if any(g != 0.0 for g in self.eq_gains):
                        interleaved = process_stereo_eq(interleaved, self._eq_coeffs, self._eq_state)

            # Retry placing into queue until it succeeds, stopped, or seeked
            while not self._stopped:
                with self._lock:
                    if self._current_frame != idx:
                        break  # Seek happened, discard this block
                try:
                    self._q.put(interleaved, timeout=0.1)
                    with self._lock:
                        if self._current_frame == idx:
                            self._current_frame = idx + 1
                    break
                except queue.Full:
                    pass

    # ------------------------------------------------------------------
    # sounddevice callback (runs on the audio thread — must not block)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Called by sounddevice whenever the audio device needs more data.
        Pulls one block from the queue and applies volume.
        Writes silence on underrun rather than crashing.
        """
        if self._paused:
            outdata[:] = 0.0
            return

        try:
            block = self._q.get_nowait()
        except queue.Empty:
            outdata[:] = 0.0
            self._underrun = True
            return

        if block is None:
            # Sentinel — track finished
            outdata[:] = 0.0
            self._stopped = True
            raise sd.CallbackStop()

        block_len = len(block)
        if block_len < frames:
            outdata[:block_len] = np.clip(block * self.volume, -1.0, 1.0)
            outdata[block_len:] = 0.0
            self._stopped = True
            raise sd.CallbackStop()
        else:
            outdata[:] = np.clip(block * self.volume, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Control loop (main thread — reads keyboard while audio plays)
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """Dispatch to the right keyboard handler for the current OS."""
        if sys.platform == "win32":
            self._control_loop_windows()
        else:
            self._control_loop_unix()

    def _control_loop_windows(self) -> None:
        """Non-blocking keyboard input on Windows via msvcrt."""
        import msvcrt
        self._print_status()
        while not self._stopped:
            if msvcrt.kbhit():
                raw = msvcrt.getch()
                # Arrow keys emit b'\xe0' + code — consume and ignore
                if raw == b"\xe0":
                    msvcrt.getch()
                else:
                    self._handle_key(raw.decode("utf-8", errors="ignore").lower())
            self._print_status()
            time.sleep(0.25)

    def _control_loop_unix(self) -> None:
        """Non-blocking keyboard input on Unix/macOS via termios."""
        import termios, tty, select
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            self._print_status()
            while not self._stopped:
                if select.select([sys.stdin], [], [], 0.25)[0]:
                    key = sys.stdin.read(1).lower()
                    self._handle_key(key)
                self._print_status()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _handle_key(self, key: str) -> None:
        """Process a single keypress character."""
        if key in ("q", "\x03"):        # Q or Ctrl-C → quit
            self._stopped = True

        elif key in ("p", " "):         # P or Space → pause / resume
            with self._lock:
                self._paused = not self._paused

        elif key == "+":                # Volume up
            self.volume = min(1.0, round(self.volume + VOLUME_STEP, 2))

        elif key == "-":                # Volume down
            self.volume = max(0.0, round(self.volume - VOLUME_STEP, 2))

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _elapsed(self) -> float:
        """Estimate current playback position in seconds."""
        with self._lock:
            idx = self._current_frame
        # Subtract queue depth — the reader is ahead of the speakers
        playback_frame = max(0, idx - self._q.qsize())
        return playback_frame * self._frame_size / self._sample_rate

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m}:{s:02d}"

    def _progress_bar(self, elapsed: float, width: int = 35) -> str:
        ratio  = min(1.0, elapsed / max(1.0, self._duration))
        filled = int(ratio * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"

    def _print_status(self) -> None:
        elapsed  = self._elapsed()
        bar      = self._progress_bar(elapsed)
        state    = "⏸ PAUSED " if self._paused else "▶ PLAYING"
        vol_pct  = int(self.volume * 100)
        underrun = "  ⚠ buffer underrun" if self._underrun else ""
        log.debug(
            f"{state}  {bar}  "
            f"{self._fmt_time(elapsed)} / {self._fmt_time(self._duration)}  "
            f"Vol: {vol_pct:3d}%{underrun}"
        )
        self._underrun = False

    def _print_info(self) -> None:
        h           = self._header
        version_str = _fmt_version(h.version) if hasattr(h, 'version') else "v?"
        stereo_str  = "Mid-Side" if self._mid_side else "Independent"
        entropy_str = "Rice" if self._entropy_mode == ENTROPY_RICE else "Huffman"
        lpc_str     = "Integer" if self._lpc_mode == LPC_INTEGER else "Float32"
        meta        = getattr(h, 'metadata', None) or Metadata()
        log.info(f"{'─' * 52}")
        log.info(f"  {self.path}")
        log.info(f"{'─' * 52}")
        # Metadata — only show populated fields
        if meta.title:
            log.info(f"  Title        {meta.title}")
        if meta.artist:
            log.info(f"  Artist       {meta.artist}")
        if meta.album:
            log.info(f"  Album        {meta.album}")
        if meta.track_number:
            year_str = f"  ({meta.year})" if meta.year else ""
            log.info(f"  Track        {meta.track_number}{year_str}")
        elif meta.year:
            log.info(f"  Year         {meta.year}")
        if not meta.is_empty():
            log.info(f"{'─' * 52}")
        log.info(f"  Format       {'HFPAC ' + version_str:>14}")
        log.info(f"  LPC          {lpc_str:>14}")
        log.info(f"  Entropy      {entropy_str:>14}")
        log.info(f"  Sample rate  {h.sample_rate:>10,} Hz")
        log.info(f"  Channels     {h.channels:>10}")
        if h.channels == 2:
            log.info(f"  Stereo       {stereo_str:>14}")
        log.info(f"  Bit depth    {h.bit_depth:>9}-bit")
        log.info(f"  Duration     {self._fmt_time(self._duration):>10}"
                 f"  ({self._duration:.1f}s)")
        log.info(f"  Frames       {self._frames_per_ch:>10,} per channel")
        if hasattr(h, 'seek_table') and h.seek_table:
            log.info(f"  Seek points  {len(h.seek_table):>10,}")
        log.info(f"{'─' * 52}")

    def _print_controls(self) -> None:
        log.info("  Controls:")
        log.info("    P  or  SPACE  —  pause / resume")
        log.info("    +  /  -       —  volume up / down")
        log.info("    Q             —  quit")


# ---------------------------------------------------------------------------
# Convenience function called by main.py
# ---------------------------------------------------------------------------

def play_hfpac(path: str, volume: float = 1.0) -> None:
    """Play a .hfpac file. Blocks until finished or user quits."""
    player = HFPACPlayer(path, volume=volume)
    player.play()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    sys.path.insert(0, ".")
    import soundfile as sf
    from codec import encode_wav

    log.info("=== HFPAC player.py smoke test ===")
    log.info("  Generating a 3-second test tone ...")

    sr  = 44100
    t   = np.linspace(0, 3.0, sr * 3, endpoint=False)
    ch1 = np.sin(2 * np.pi * 440 * t) * 0.5   # 440 Hz left
    ch2 = np.sin(2 * np.pi * 660 * t) * 0.5   # 660 Hz right
    sf.write("player_test.wav", np.stack([ch1, ch2], axis=1), sr, subtype="PCM_16")
    encode_wav("player_test.wav", "player_test.hfpac")

    log.info("  Playing now — you should hear two tones (440 Hz L, 660 Hz R).")
    play_hfpac("player_test.hfpac")

    os.remove("player_test.wav")
    os.remove("player_test.hfpac")