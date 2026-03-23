"""
codec.py — Full Encode/Decode Pipeline for HFPAC
=================================================
This module is the orchestrator. It connects all the pieces:

  Encode:  .wav file → read PCM → LPC → Rice/Huffman → write .hfpac
  Decode:  .hfpac file → read frames → Rice/Huffman → LPC → write .wav

v6 additions:
  - Adaptive LPC order per frame (tries all even orders 2–20)
  - Subframe silence encoding (max |sample| ≤ 1 → FRAME_SILENCE)
  - History carry-over across CONT frames (23% lower boundary residuals)
  - SYNC/CONT/SILENCE frame types with seek table pointing to SYNC only

Dependencies:
  pip install numpy soundfile
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from lpc import (
    compute_lpc_coefficients,
    encode_frame, decode_frame,
    encode_frame_int, decode_frame_int,
    quantize_lpc_coefficients,
    make_prior_history,
    select_lpc_order,
    split_into_frames,
    DEFAULT_LPC_ORDER, DEFAULT_LPC_PRECISION,
    FRAME_SIZE, LPC_FLOAT, LPC_INTEGER,
)
from huffman import build_tree, build_code_table, encode as huffman_encode, decode as huffman_decode
from rice import choose_k, encode as rice_encode, decode as rice_decode
from hfpac_format import (
    HFPACHeader, EncodedFrame, Metadata,
    FRAMES_PER_BLOCK, SYNC_INTERVAL, SILENCE_THRESHOLD,
    STEREO_INDEPENDENT, STEREO_MID_SIDE,
    ENTROPY_HUFFMAN, ENTROPY_RICE,
    FRAME_SYNC, FRAME_CONT, FRAME_SILENCE,
    LPC_FLOAT as FMT_LPC_FLOAT, LPC_INTEGER as FMT_LPC_INTEGER,
    write_hfpac, read_hfpac,
)


# ---------------------------------------------------------------------------
# Phase 1 worker — runs in parallel (no shared state)
# ---------------------------------------------------------------------------

def _encode_frame_worker(args):
    """
    Select the best LPC order for one frame and compute its coefficients.
    Returns (frame_idx, best_order, coeffs, frame_samples).
    Called in parallel — no history needed at this stage.
    """
    frame_idx, frame_samples, adaptive = args
    if adaptive:
        best_order = select_lpc_order(frame_samples)
    else:
        best_order = DEFAULT_LPC_ORDER
    coeffs = compute_lpc_coefficients(frame_samples, best_order)
    return frame_idx, best_order, coeffs, frame_samples


# ---------------------------------------------------------------------------
# Mid-side stereo transforms
# ---------------------------------------------------------------------------

def _to_mid_side(left: np.ndarray, right: np.ndarray):
    return (left + right) / 2.0, (left - right) / 2.0

def _from_mid_side(mid: np.ndarray, side: np.ndarray):
    return mid + side, mid - side

def _to_mid_side_int(left: np.ndarray, right: np.ndarray):
    left_i  = np.round(left).astype(np.int64)
    right_i = np.round(right).astype(np.int64)
    return ((left_i + right_i) >> 1).astype(np.float64), \
           (left_i - right_i).astype(np.float64)

def _from_mid_side_int(mid: np.ndarray, side: np.ndarray):
    mid_i  = np.round(mid).astype(np.int64)
    side_i = np.round(side).astype(np.int64)
    return (mid_i + ((side_i + 1) >> 1)).astype(np.float64), \
           (mid_i - (side_i >> 1)).astype(np.float64)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_wav(path: str) -> Tuple[np.ndarray, int, int]:
    data, sr = sf.read(path, dtype="float64", always_2d=False)
    info     = sf.info(path)
    bd_str   = info.subtype_info
    if "24" in bd_str:
        bd, scale = 24, 2 ** 23
    else:
        bd, scale = 16, 2 ** 15
    return (data * scale).astype(np.float64), sr, bd


def _write_wav(path: str, samples: np.ndarray, sample_rate: int, bit_depth: int) -> None:
    scale   = 2 ** (bit_depth - 1)
    subtype = "PCM_16" if bit_depth == 16 else "PCM_24"
    sf.write(path, np.clip(samples / scale, -1.0, 1.0), sample_rate, subtype=subtype)


def _encode_channel(
    samples:          np.ndarray,
    frame_size:       int,
    channel_idx:      int,
    total_channels:   int,
    frames_per_block: int  = FRAMES_PER_BLOCK,
    entropy_mode:     int  = ENTROPY_RICE,
    lpc_mode:         int  = FMT_LPC_INTEGER,
    sync_interval:    int  = SYNC_INTERVAL,
    adaptive_order:   bool = True,
    file_version:     int  = None,
    progress_callback: callable = None,
    current_base:     int  = 0,
    total_total:      int  = 0,
) -> List[EncodedFrame]:
    """
    Encode one audio channel to a list of EncodedFrames.

    file_version controls which v6-specific features are active:
        ≥ 8 : silence detection, history carry-over, SYNC/CONT frame types
        < 8 : flat encoding — no frame types, no history (legacy v5.1/v7 compat)
    """
    from hfpac_format import FORMAT_VERSION as _FV
    fver = file_version if file_version is not None else _FV
    use_v6_features = (fver >= 8)
    # Adaptive order only makes sense for v8+ where n_coeffs is stored per frame
    effective_adaptive = adaptive_order and use_v6_features
    frames_list = list(split_into_frames(samples, frame_size))
    total_frames = len(frames_list)
    frame_data: dict = {}   # idx → (best_order, coeffs, frame_samples)

    # ── Phase 1: parallel LPC analysis ────────────────────────────────
    work_items = [(idx, fs, effective_adaptive) for idx, fs in frames_list]
    import os
    max_workers = max(1, os.cpu_count() - 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunksize = max(1, len(work_items) // (max_workers * 4))
        for result in executor.map(_encode_frame_worker, work_items, chunksize=chunksize):
            idx, best_order, coeffs, fs = result
            frame_data[idx] = (best_order, coeffs, fs)
            completed = len(frame_data)
            if progress_callback:
                progress_callback(current_base + completed, total_total)
            print(
                f"\r  Encoding ch {channel_idx + 1}/{total_channels} "
                f"frame {completed}/{total_frames} ...",
                end="", flush=True,
            )
    print()

    # ── Phase 2: sequential residual + entropy coding ─────────────────
    encoded_frames: List[EncodedFrame] = []
    prior_history = None    # None → zeros (SYNC frame behaviour)
    lpc_frame_count = 0     # counts only non-silence LPC frames
    phase_1_completed = total_frames

    for idx in range(total_frames):
        best_order, coeffs, fs = frame_data[idx]

        # Ensure sync points are aligned to absolute frame indices so that
        # seek table entries match exactly across all channels.
        is_sync_point = use_v6_features and (idx == 0 or (sync_interval > 0 and idx % sync_interval == 0))

        # Silence detection (v6 only)
        # Avoid creating silence frames at sync points to guarantee aligned seek entries.
        if use_v6_features and not is_sync_point and np.max(np.abs(fs)) <= SILENCE_THRESHOLD:
            silence_val = int(round(fs[0])) if len(fs) > 0 else 0
            encoded_frames.append(EncodedFrame(
                lpc_coeffs    = np.zeros(best_order),
                frame_type    = FRAME_SILENCE,
                silence_value = silence_val,
            ))
            if lpc_mode == FMT_LPC_INTEGER:
                prior_history = np.full(best_order, silence_val, dtype=np.int64)
            continue

        # Frame type assignment (v6 only; legacy uses FRAME_SYNC default)
        if use_v6_features:
            if is_sync_point:
                frame_type    = FRAME_SYNC
                prior_history = None
            else:
                frame_type = FRAME_CONT
        else:
            frame_type    = FRAME_SYNC
            prior_history = None   # no history for legacy formats

        # Compute residuals
        if lpc_mode == FMT_LPC_INTEGER:
            ci  = quantize_lpc_coefficients(coeffs, DEFAULT_LPC_PRECISION)
            res = encode_frame_int(fs, ci, DEFAULT_LPC_PRECISION,
                                   prior_history=prior_history).tolist()
            cf  = ci.astype(np.float64) / float(1 << DEFAULT_LPC_PRECISION)
            prec = DEFAULT_LPC_PRECISION
            # Update history from reconstructed samples (not raw fs)
            rec  = decode_frame_int(np.array(res, dtype=np.int32), ci,
                                    DEFAULT_LPC_PRECISION,
                                    prior_history=prior_history)
            if use_v6_features:
                prior_history = make_prior_history(rec, len(ci))
            else:
                prior_history = None
        else:
            cf   = np.array(coeffs, dtype=np.float32).astype(np.float64)
            res  = encode_frame(
                np.ascontiguousarray(fs, dtype=np.float64),
                np.ascontiguousarray(cf, dtype=np.float64),
            ).tolist()
            ci   = None
            prec = 0
            prior_history = None   # float path doesn't carry history

        # Entropy coding
        if entropy_mode == ENTROPY_RICE:
            k           = choose_k(res)
            payload, nb = rice_encode(res, k)
            encoded_frames.append(EncodedFrame(
                lpc_coeffs     = cf,
                lpc_coeffs_int = ci,
                lpc_precision  = prec,
                frame_type     = frame_type,
                rice_k         = k,
                rice_payload   = payload,
                num_bits       = nb,
            ))
        else:
            # Huffman — accumulate a block then encode
            # For v6 with adaptive order we build per-frame trees (no shared blocks)
            tree    = build_tree(res)
            codes   = build_code_table(tree)
            payload, nb = huffman_encode(res, codes)
            encoded_frames.append(EncodedFrame(
                lpc_coeffs      = cf,
                lpc_coeffs_int  = ci,
                lpc_precision   = prec,
                frame_type      = frame_type,
                huffman_tree    = tree,
                num_bits        = nb,
                huffman_payload = payload,
            ))

        lpc_frame_count += 1
        if progress_callback:
            progress_callback(current_base + phase_1_completed + idx + 1, total_total)

    return encoded_frames


def _decode_channel(
    frames:          List[EncodedFrame],
    frame_size:      int,
    num_samples:     int,
    channel_idx:     int,
    total_channels:  int,
    entropy_mode:    int  = ENTROPY_RICE,
    lpc_mode:        int  = FMT_LPC_INTEGER,
    file_version:    int  = 8,
) -> np.ndarray:
    """
    Decode one channel from a list of EncodedFrames.

    v8 (v6): handles FRAME_SYNC/CONT/SILENCE with history carry-over.
    v2–v7:   existing behaviour — no frame types, no history.
    """
    all_samples:      List[np.ndarray] = []
    total_frames      = len(frames)
    cached_tree_bytes = None
    cached_tree       = None
    prior_history     = None   # reset at SYNC or for pre-v8 files

    for frame_idx, fd in enumerate(frames):
        print(
            f"\r  Decoding ch {channel_idx + 1}/{total_channels} "
            f"frame {frame_idx + 1}/{total_frames} ...",
            end="", flush=True,
        )

        # ── Silence frame ─────────────────────────────────────────────
        if file_version >= 8 and fd.frame_type == FRAME_SILENCE:
            samples = np.full(frame_size, float(fd.silence_value))
            all_samples.append(samples)
            if lpc_mode == FMT_LPC_INTEGER:
                order = len(fd.lpc_coeffs) if len(fd.lpc_coeffs) > 0 \
                        else DEFAULT_LPC_ORDER
                prior_history = np.full(order, fd.silence_value,
                                        dtype=np.int64)
            continue

        # ── SYNC frame: reset history ──────────────────────────────────
        if file_version >= 8 and fd.frame_type == FRAME_SYNC:
            prior_history = None

        # ── Decode entropy payload ─────────────────────────────────────
        if entropy_mode == ENTROPY_RICE:
            residuals = rice_decode(
                fd.rice_payload, fd.rice_k, fd.num_bits, frame_size)
        else:
            if fd.huffman_tree is not cached_tree_bytes:
                cached_tree_bytes = fd.huffman_tree
                from huffman import deserialise_tree
                cached_tree, _ = deserialise_tree(cached_tree_bytes)
            residuals = huffman_decode(
                fd.huffman_payload, cached_tree, fd.num_bits, frame_size)

        # ── Reconstruct samples ────────────────────────────────────────
        res_i32 = np.array(residuals, dtype=np.int32)
        if lpc_mode == FMT_LPC_INTEGER and fd.lpc_coeffs_int is not None:
            samples = decode_frame_int(res_i32, fd.lpc_coeffs_int,
                                       fd.lpc_precision,
                                       prior_history=prior_history)
            # Only carry history forward for v8 (v6) files
            if file_version >= 8:
                prior_history = make_prior_history(samples, len(fd.lpc_coeffs_int))
            else:
                prior_history = None
        else:
            samples = decode_frame(res_i32, fd.lpc_coeffs)
            prior_history = None

        all_samples.append(samples)

    print()
    return np.concatenate(all_samples)[:num_samples]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_wav(
    input_wav:        str,
    output_hfpac:     str,
    lpc_order:        int      = DEFAULT_LPC_ORDER,   # used when adaptive_order=False
    frame_size:       int      = FRAME_SIZE,
    frames_per_block: int      = FRAMES_PER_BLOCK,
    stereo_mode:      int      = STEREO_MID_SIDE,
    entropy_mode:     int      = ENTROPY_RICE,
    lpc_mode:         int      = FMT_LPC_INTEGER,
    sync_interval:    int      = SYNC_INTERVAL,
    metadata:         Metadata = None,
    adaptive_order:   bool     = True,
    progress_callback: callable = None,
) -> dict:
    """
    Encode a WAV file to HFPAC v6 format.

    Args:
        sync_interval  — insert a FRAME_SYNC every N LPC frames (0 = no sync)
        adaptive_order — if True, select the best LPC order per frame (v6 default)
                         if False, use lpc_order for every frame (legacy behaviour)
    """
    print(f"[HFPAC] Encoding: {input_wav} → {output_hfpac}")
    t_start = time.perf_counter()

    # Capture FORMAT_VERSION now — it may be patched at runtime for legacy writes
    import hfpac_format as _hfmt
    hfmt_version = _hfmt.FORMAT_VERSION

    raw_samples, sr, bit_depth = _read_wav(input_wav)

    if raw_samples.ndim == 1:
        channels_data = [raw_samples]
        num_channels  = 1
    else:
        num_channels  = raw_samples.shape[1]
        channels_data = [raw_samples[:, c] for c in range(num_channels)]

    num_samples  = len(channels_data[0])
    entropy_name = "Rice" if entropy_mode == ENTROPY_RICE else "Huffman"
    lpc_name     = "Integer" if lpc_mode == FMT_LPC_INTEGER else "Float32"

    print(f"  Sample rate:  {sr} Hz")
    print(f"  Bit depth:    {bit_depth}-bit")
    print(f"  Channels:     {num_channels}")
    print(f"  Samples:      {num_samples:,}")
    print(f"  Duration:     {num_samples / sr:.2f}s")
    print(f"  LPC:          {lpc_name}  "
          f"({'adaptive 2–20' if adaptive_order else f'order {lpc_order}'})")
    print(f"  Entropy:      {entropy_name}")

    # Apply stereo transform
    actual_stereo_mode = STEREO_INDEPENDENT
    if num_channels == 2 and stereo_mode == STEREO_MID_SIDE:
        if lpc_mode == FMT_LPC_INTEGER:
            channels_data[0], channels_data[1] = _to_mid_side_int(
                channels_data[0], channels_data[1])
        else:
            channels_data[0], channels_data[1] = _to_mid_side(
                channels_data[0], channels_data[1])
        actual_stereo_mode = STEREO_MID_SIDE
        print(f"  Stereo mode:  Mid-Side")
    else:
        print(f"  Stereo mode:  Independent")

    total_frames_per_channel = (num_samples + frame_size - 1) // frame_size
    total_total = total_frames_per_channel * num_channels * 2

    all_frames: List[EncodedFrame] = []
    current_base = 0

    for ch_idx, ch_samples in enumerate(channels_data):
        ch_frames = _encode_channel(
            ch_samples, frame_size, ch_idx, num_channels,
            frames_per_block, entropy_mode, lpc_mode,
            sync_interval, adaptive_order,
            file_version=hfmt_version,
            progress_callback=progress_callback,
            current_base=current_base,
            total_total=total_total,
        )
        current_base += total_frames_per_channel * 2
        all_frames.extend(ch_frames)

    # lpc_order in the header is the max/default order for info purposes
    effective_order = lpc_order if not adaptive_order else DEFAULT_LPC_ORDER

    trailing_padding = (frame_size - (num_samples % frame_size)) % frame_size

    header = HFPACHeader(
        sample_rate      = sr,
        channels         = num_channels,
        bit_depth        = bit_depth,
        lpc_order        = effective_order,
        frame_size       = frame_size,
        num_samples      = num_samples,
        num_frames       = len(all_frames),
        frames_per_block = frames_per_block,
        stereo_mode      = actual_stereo_mode,
        entropy_mode     = entropy_mode,
        lpc_mode         = lpc_mode,
        sync_interval    = sync_interval,
        trailing_padding = trailing_padding,
        metadata         = metadata or Metadata(),
    )
    write_hfpac(output_hfpac, header, all_frames)

    elapsed     = time.perf_counter() - t_start
    input_size  = Path(input_wav).stat().st_size
    output_size = Path(output_hfpac).stat().st_size
    n_seek      = len(header.seek_table)

    print(f"\n  ✅ Done in {elapsed:.2f}s")
    print(f"  Input:  {input_size:,} bytes")
    print(f"  Output: {output_size:,} bytes")
    print(f"  Ratio:  {input_size / output_size:.2f}x")
    print(f"  Seek table: {n_seek} entries\n")

    return {
        "input_size":  input_size,
        "output_size": output_size,
        "ratio":       input_size / output_size,
        "encode_time": elapsed,
        "sample_rate": sr,
        "channels":    num_channels,
        "bit_depth":   bit_depth,
        "num_samples": num_samples,
        "duration":    num_samples / sr,
    }


def decode_hfpac(input_hfpac: str, output_wav: str) -> dict:
    """Decode an HFPAC file back to a WAV file — all versions v2–v8."""
    print(f"[HFPAC] Decoding: {input_hfpac} → {output_wav}")
    t_start = time.perf_counter()

    header, all_frames = read_hfpac(input_hfpac)
    entropy_mode = getattr(header, 'entropy_mode', ENTROPY_HUFFMAN)
    lpc_mode     = getattr(header, 'lpc_mode',     FMT_LPC_FLOAT)
    stereo_mode  = getattr(header, 'stereo_mode',  STEREO_INDEPENDENT)
    lpc_name     = "Integer" if lpc_mode == FMT_LPC_INTEGER else "Float32"
    entropy_name = "Rice"    if entropy_mode == ENTROPY_RICE else "Huffman"

    print(f"  Sample rate:  {header.sample_rate} Hz")
    print(f"  Channels:     {header.channels}")
    print(f"  Bit depth:    {header.bit_depth}-bit")
    print(f"  LPC mode:     {lpc_name}")
    print(f"  Entropy:      {entropy_name}")
    print(f"  Frames:       {header.num_frames}")
    print(f"  Duration:     {header.num_samples / header.sample_rate:.2f}s")

    frames_per_ch = header.num_frames // header.channels
    channel_pcm: List[np.ndarray] = []

    for ch_idx in range(header.channels):
        start     = ch_idx * frames_per_ch
        ch_frames = all_frames[start:start + frames_per_ch]
        pcm = _decode_channel(
            ch_frames, header.frame_size, header.num_samples,
            ch_idx, header.channels,
            entropy_mode, lpc_mode,
            file_version=header.version,
        )
        channel_pcm.append(pcm)

    # Reverse stereo transform
    if stereo_mode == STEREO_MID_SIDE and header.channels == 2:
        if lpc_mode == FMT_LPC_INTEGER:
            channel_pcm[0], channel_pcm[1] = _from_mid_side_int(
                channel_pcm[0], channel_pcm[1])
        else:
            channel_pcm[0], channel_pcm[1] = _from_mid_side(
                channel_pcm[0], channel_pcm[1])

    final = channel_pcm[0] if header.channels == 1 else np.stack(channel_pcm, axis=1)
    _write_wav(output_wav, final, header.sample_rate, header.bit_depth)

    elapsed = time.perf_counter() - t_start
    print(f"\n  ✅ Done in {elapsed:.2f}s\n")
    return {"decode_time": elapsed, "sample_rate": header.sample_rate,
            "channels": header.channels, "num_samples": header.num_samples}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, ".")

    print("=== HFPAC codec.py smoke test (v6) ===\n")

    sr = 44100; n = int(sr * 1.0)
    t  = np.linspace(0, 1.0, n, endpoint=False)
    # Mono with a genuine silent section — M/S doesn't apply to mono,
    # so silence is preserved through to _encode_channel
    sig = np.concatenate([np.sin(2*np.pi*440*t[:n//2])*0.8,
                          np.zeros(n - n//2)])

    tmp_wav     = "test_source.wav"
    tmp_hfpac   = "test_source.hfpac"
    tmp_decoded = "test_decoded.wav"

    sf.write(tmp_wav, sig, sr, subtype="PCM_16")
    print(f"Generated test WAV: {tmp_wav} ({os.path.getsize(tmp_wav):,} bytes)\n")

    enc_stats = encode_wav(tmp_wav, tmp_hfpac)
    dec_stats = decode_hfpac(tmp_hfpac, tmp_decoded)

    from hfpac_format import read_hfpac, FRAME_SYNC, FRAME_CONT, FRAME_SILENCE
    h, frames = read_hfpac(tmp_hfpac)
    sync_n    = sum(1 for f in frames if f.frame_type == FRAME_SYNC)
    silence_n = sum(1 for f in frames if f.frame_type == FRAME_SILENCE)
    print(f"Frame types:   SYNC={sync_n}  CONT={len(frames)-sync_n-silence_n}  SILENCE={silence_n}")
    assert silence_n > 0, "Expected at least one silence frame"
    assert sync_n    > 0, "Expected at least one sync frame"

    original,    _ = sf.read(tmp_wav,     dtype="float64")
    reconstructed, _ = sf.read(tmp_decoded, dtype="float64")
    max_err = np.max(np.abs(original - reconstructed))
    print(f"Roundtrip max error: {max_err:.6f}  (0.0 = bit-perfect)")
    assert max_err == 0.0, f"Not bit-perfect: {max_err}"

    for f in [tmp_wav, tmp_hfpac, tmp_decoded]:
        os.remove(f)

    print(f"\n✅ Full v6 WAV → HFPAC → WAV roundtrip complete!")