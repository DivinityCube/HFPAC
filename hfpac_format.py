"""
hfpac_format.py — Binary File Format for HFPAC
================================================
This module handles reading and writing the .hfpac binary file format.
It is the glue between the codec pipeline (lpc.py + huffman.py) and
the filesystem.

v6 (internal version 8) file layout:
┌─────────────────────────────────────────────────────────────┐
│  FIXED HEADER  (27 bytes, identical to v5.1)                │
│  METADATA BLOCK  (variable, CRC-32 protected)               │
│  SEEK TABLE  (4 + 12×N bytes, zeroed then overwritten)      │
│  FRAMES  (repeated num_frames times):                       │
│    v6 LPC frame (SYNC or CONT):                             │
│      [frame_type: uint8]  0=SYNC  1=CONT                    │
│      [n_coeffs: uint32]   (adaptive — varies per frame)     │
│      [lpc_precision: uint8]  [coefficients: int16 × N]      │
│      [rice_k: uint8]  [num_bits: uint32]                    │
│      [payload_size: uint32]  [payload: bytes]               │
│      [crc32: uint32]                                        │
│    v6 silence frame:                                        │
│      [frame_type: uint8]  2=SILENCE                         │
│      [silence_value: int32]  (constant sample value)        │
│      [crc32: uint32]                                        │
└─────────────────────────────────────────────────────────────┘

Seek table entries point ONLY to FRAME_SYNC frames.
FRAME_CONT frames carry history from the previous frame.
FRAME_SILENCE frames output silence_value for the whole frame.

All multi-byte integers are big-endian (network byte order).

Format version history
─────────────────────
  v2   (int 2) — float64 LPC, Huffman, one tree/frame
  v3   (int 3) — float32 LPC, Huffman, shared block trees
  v4   (int 4) — float32 LPC, Huffman, stereo_mode
  v4.5 (int 5) — float32 LPC, Rice or Huffman, entropy_mode
  v5   (int 6) — integer or float LPC, seek table
  v5.1 (int 7) — metadata block, per-frame CRC-32
  v6   (int 8) — adaptive LPC order, subframe silence, history carry-over
"""

import binascii
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, List, Optional

import numpy as np

from huffman import HuffmanNode, serialise_tree, deserialise_tree


# ---------------------------------------------------------------------------
# Magic bytes & format constants
# ---------------------------------------------------------------------------

MAGIC             = b"HFPAC"
FORMAT_VERSION    = 9        # v6.1 (displayed as "v6.1", stored as 9)
MIN_VERSION       = 2
HEADER_SIZE       = 27       # fixed header bytes (v5/v8) - larger for v9

FRAMES_PER_BLOCK  = 64       # Huffman block size (legacy, v3–v5.1)

STEREO_INDEPENDENT = 0
STEREO_MID_SIDE    = 1

ENTROPY_HUFFMAN = 0
ENTROPY_RICE    = 1

LPC_FLOAT   = 0
LPC_INTEGER = 1

# v6 frame type codes — prepended as uint8 to every frame
FRAME_SYNC    = 0   # LPC frame — history reset to zero; valid seek point
FRAME_CONT    = 1   # LPC frame — continues history from previous frame
FRAME_SILENCE = 2   # silence/constant frame — no LPC, just a value

FRAME_SIZE_DIVISOR = 256
SYNC_INTERVAL      = 64     # insert a FRAME_SYNC every N frames (configurable)
SILENCE_THRESHOLD  = 1      # max |sample| ≤ this → encode as FRAME_SILENCE

# Internal version integer → user-facing display string.
# The internal int is never shown directly to users.
_VERSION_DISPLAY = {
    2: "v2",
    3: "v3",
    4: "v4",
    5: "v4.5",
    6: "v5",
    7: "v5.1",
    8: "v6",
    9: "v6.1",
}


def display_version(internal_version: int) -> str:
    """
    Convert an internal format version integer to the user-facing string.

    The internal version (2–8) is never shown directly to users.
    Call this wherever a version string is displayed in any UI or message.
    """
    return _VERSION_DISPLAY.get(internal_version, f"v?({internal_version})")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Metadata:
    """
    Optional track metadata stored in v5.1+ files.

    Fields not set are stored as empty strings / 0 and displayed as unknown.
    """
    title:        str = ""
    artist:       str = ""
    album:        str = ""
    track_number: int = 0    # 0 = unknown
    year:         int = 0    # 0 = unknown

    def is_empty(self) -> bool:
        return not any([self.title, self.artist, self.album,
                        self.track_number, self.year])

@dataclass
class HFPACHeader:
    """All metadata for one .hfpac file."""
    sample_rate:      int
    channels:         int
    bit_depth:        int
    lpc_order:        int       # max/default LPC order (v6: each frame may differ)
    frame_size:       int
    num_samples:      int
    num_frames:       int
    frames_per_block: int  = FRAMES_PER_BLOCK
    stereo_mode:      int  = STEREO_INDEPENDENT
    entropy_mode:     int  = ENTROPY_RICE
    lpc_mode:         int  = LPC_INTEGER
    encoder_delay:    int  = 0
    trailing_padding: int  = 0
    sync_interval:    int  = SYNC_INTERVAL   # v6: replaces seek_interval
    version:          int  = FORMAT_VERSION
    seek_table:       list = None
    metadata:         Metadata = None

    def __post_init__(self):
        if self.seek_table is None:
            self.seek_table = []
        if self.metadata is None:
            self.metadata = Metadata()

    # Backwards-compat alias — older code used seek_interval
    @property
    def seek_interval(self) -> int:
        return self.sync_interval

    @seek_interval.setter
    def seek_interval(self, v: int):
        self.sync_interval = v


@dataclass
class EncodedFrame:
    """Everything needed to store / restore one frame of audio."""
    lpc_coeffs:      np.ndarray       # float64 (both modes)

    # v6 frame type
    frame_type:      int          = FRAME_SYNC   # FRAME_SYNC / FRAME_CONT / FRAME_SILENCE
    silence_value:   int          = 0            # only used when frame_type == FRAME_SILENCE

    # Huffman fields (entropy_mode == ENTROPY_HUFFMAN)
    huffman_tree:    any          = None
    num_bits:        int          = 0
    huffman_payload: bytes        = b""

    # Rice fields (entropy_mode == ENTROPY_RICE)
    rice_k:          int          = 0
    rice_payload:    bytes        = b""

    # Integer LPC fields (lpc_mode == LPC_INTEGER)
    lpc_precision:   int          = 0
    lpc_coeffs_int:  np.ndarray   = None


# ---------------------------------------------------------------------------
# CRC helpers
# ---------------------------------------------------------------------------

def _crc32(data: bytes) -> int:
    """Return CRC-32 of data as an unsigned 32-bit integer."""
    return binascii.crc32(data) & 0xFFFFFFFF


class _ReadTracker:
    """
    Wraps a binary file handle and records every byte read through it.
    Used to compute CRC over frame bytes without knowing the frame size
    in advance.
    """
    def __init__(self, f: BinaryIO):
        self._f   = f
        self._buf = bytearray()

    def read(self, n: int) -> bytes:
        data = self._f.read(n)
        self._buf.extend(data)
        return data

    def tell(self)           -> int:  return self._f.tell()
    def seek(self, *args)    -> int:  return self._f.seek(*args)

    @property
    def captured(self) -> bytes:
        return bytes(self._buf)

    def reset(self):
        self._buf = bytearray()


# ---------------------------------------------------------------------------
# Metadata block read / write
# ---------------------------------------------------------------------------
# Layout:
#   [block_size: uint32]         total bytes that follow this field
#   [title_len: uint16][title: utf-8 bytes]
#   [artist_len: uint16][artist: utf-8 bytes]
#   [album_len: uint16][album: utf-8 bytes]
#   [track_number: uint16]       0 = unknown
#   [year: uint16]               0 = unknown
#   [metadata_crc32: uint32]     CRC-32 of all preceding metadata bytes

def _write_metadata_block(f: BinaryIO, meta: Metadata) -> None:
    """Serialise and write the metadata block."""
    def _str(s): return s.encode("utf-8") if s else b""
    t  = _str(meta.title)
    ar = _str(meta.artist)
    al = _str(meta.album)

    body = struct.pack(">H", len(t))  + t  + \
           struct.pack(">H", len(ar)) + ar + \
           struct.pack(">H", len(al)) + al + \
           struct.pack(">HH", meta.track_number, meta.year)

    crc  = _crc32(body)
    block = body + struct.pack(">I", crc)
    f.write(struct.pack(">I", len(block)))
    f.write(block)


def _read_metadata_block(f: BinaryIO) -> Metadata:
    """Read and verify the metadata block."""
    (block_size,) = struct.unpack(">I", f.read(4))
    block = f.read(block_size)

    # Last 4 bytes are the CRC
    body        = block[:-4]
    stored_crc  = struct.unpack(">I", block[-4:])[0]
    computed    = _crc32(body)
    if stored_crc != computed:
        raise ValueError(
            f"Metadata CRC mismatch (stored {stored_crc:#010x}, "
            f"computed {computed:#010x}) — metadata may be corrupted."
        )

    pos = 0
    def _read_str():
        nonlocal pos
        (length,) = struct.unpack_from(">H", body, pos); pos += 2
        s = body[pos:pos + length].decode("utf-8", errors="replace"); pos += length
        return s

    title  = _read_str()
    artist = _read_str()
    album  = _read_str()
    track_number, year = struct.unpack_from(">HH", body, pos)

    return Metadata(title=title, artist=artist, album=album,
                    track_number=track_number, year=year)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

# v6/v7/v8 share the same 27-byte fixed header struct.
# v8 reuses the last byte as sync_interval (previously seek_interval - same
# position, same meaning, just renamed to reflect that sync frames = seek pts).
_V9_HEADER_FMT = ">5sBIBBBBIIBBBBBHH"
_V8_HEADER_FMT = ">5sBIBBBBIIBBBBB"
_V8_SEEK_ENTRY = ">IQ"   # frame_idx (uint32) + byte_offset (uint64) = 12 bytes


def _seek_table_size(num_entries: int) -> int:
    return 4 + num_entries * struct.calcsize(_V8_SEEK_ENTRY)


def _num_seek_entries(num_frames: int, sync_interval: int) -> int:
    """Number of seek entries — one per FRAME_SYNC.
    Worst case (all SYNC): ceil(num_frames / sync_interval)."""
    if sync_interval <= 0:
        return 0
    return (num_frames + sync_interval - 1) // sync_interval


def write_header(f: BinaryIO, header: HFPACHeader) -> None:
    """Write the 27-byte fixed header (layout identical across v6–v8)."""
    if header.version >= 9:
        data = struct.pack(
            _V9_HEADER_FMT,
            MAGIC,
            header.version,
            header.sample_rate,
            header.channels,
            header.bit_depth,
            header.lpc_order,
            header.frame_size // FRAME_SIZE_DIVISOR,
            header.num_samples,
            header.num_frames,
            header.frames_per_block,
            header.stereo_mode,
            header.entropy_mode,
            header.lpc_mode,
            header.sync_interval,
            header.encoder_delay,
            header.trailing_padding,
        )
    else:
        data = struct.pack(
            _V8_HEADER_FMT,
            MAGIC,
            header.version,
            header.sample_rate,
            header.channels,
            header.bit_depth,
            header.lpc_order,
            header.frame_size // FRAME_SIZE_DIVISOR,
            header.num_samples,
            header.num_frames,
            header.frames_per_block,
            header.stereo_mode,
            header.entropy_mode,
            header.lpc_mode,
            header.sync_interval,
        )
    f.write(data)


def _write_seek_table_placeholder(f: BinaryIO, num_entries: int) -> int:
    pos = f.tell()
    f.write(struct.pack(">I", num_entries))
    f.write(b"\x00" * (num_entries * struct.calcsize(_V8_SEEK_ENTRY)))
    return pos


def _overwrite_seek_table(f: BinaryIO, table_pos: int, seek_table: list) -> None:
    f.seek(table_pos)
    f.write(struct.pack(">I", len(seek_table)))
    for frame_idx, byte_offset in seek_table:
        f.write(struct.pack(_V8_SEEK_ENTRY, frame_idx, byte_offset))


def write_frame(f: BinaryIO, frame: EncodedFrame,
                entropy_mode: int, lpc_mode: int = LPC_FLOAT,
                write_crc: bool = False,
                file_version: int = None) -> None:
    """
    Write one encoded frame, optionally appending a CRC-32.

    file_version controls whether the v8 frame_type byte is written:
        ≥ 8 : writes frame_type byte + silence/LPC content + CRC
        < 8 : writes LPC/entropy content only (± CRC for v7)

    Defaults to FORMAT_VERSION if not supplied.
    """
    import io
    ver = file_version if file_version is not None else FORMAT_VERSION
    buf = io.BytesIO()
    _write_frame_body(buf, frame, entropy_mode, lpc_mode, ver)
    frame_bytes = buf.getvalue()
    f.write(frame_bytes)
    if write_crc:
        f.write(struct.pack(">I", _crc32(frame_bytes)))


def _write_frame_body(f: BinaryIO, frame: EncodedFrame,
                      entropy_mode: int, lpc_mode: int,
                      file_version: int = None) -> None:
    """Write frame content; prepends frame_type byte only for v8+."""
    ver = file_version if file_version is not None else FORMAT_VERSION

    if ver >= 8:
        f.write(struct.pack(">B", frame.frame_type))
        if frame.frame_type == FRAME_SILENCE:
            f.write(struct.pack(">i", frame.silence_value))
            return

    # LPC coefficients
    if lpc_mode == LPC_INTEGER and frame.lpc_coeffs_int is not None:
        coeffs_int = np.asarray(frame.lpc_coeffs_int, dtype=np.int16)
        f.write(struct.pack(">I", len(coeffs_int)))
        f.write(struct.pack(">B", frame.lpc_precision))
        f.write(coeffs_int.astype(">i2").tobytes())
    else:
        coeffs_f32 = np.array(frame.lpc_coeffs, dtype=np.float32).astype(">f4")
        f.write(struct.pack(">I", len(frame.lpc_coeffs)))
        f.write(coeffs_f32.tobytes())

    # Entropy payload
    if entropy_mode == ENTROPY_HUFFMAN:
        if frame.huffman_tree is not None:
            tree_bytes = (
                frame.huffman_tree if isinstance(frame.huffman_tree, bytes)
                else serialise_tree(frame.huffman_tree)
            )
            f.write(struct.pack(">I", len(tree_bytes)))
            f.write(tree_bytes)
        else:
            f.write(struct.pack(">I", 0))
        f.write(struct.pack(">I", frame.num_bits))
        f.write(struct.pack(">I", len(frame.huffman_payload)))
        f.write(frame.huffman_payload)
    else:
        f.write(struct.pack(">B", frame.rice_k))
        f.write(struct.pack(">I", frame.num_bits))
        f.write(struct.pack(">I", len(frame.rice_payload)))
        f.write(frame.rice_payload)


def write_hfpac(path: str, header: HFPACHeader,
                frames: List[EncodedFrame]) -> None:
    """
    Write a complete .hfpac file.

    v8 (v6) layout:
        [fixed header: 27 bytes]
        [metadata block: variable]
        [seek table: 4 + 12×N bytes]   ← zeroed, overwritten after encoding
        [frame bytes][crc32: 4 bytes]
        ...

    Seek table entries are only written for FRAME_SYNC frames.
    The placeholder is sized from the actual SYNC frame count so it is
    always large enough regardless of how the encoder places sync points.
    """
    # Always write in the currently-active format version.
    # header.version reflects the default at class-definition time, but
    # FORMAT_VERSION can be patched at runtime (e.g. for legacy-format tests).
    fver = FORMAT_VERSION
    header.version = fver   # keep header consistent with what's written

    if fver >= 8:
        n_entries = sum(1 for fr in frames if fr.frame_type == FRAME_SYNC)
    else:
        n_entries = _num_seek_entries(len(frames), header.sync_interval)

    with open(path, "wb") as f:
        write_header(f, header)
        # Metadata block: v7+
        if fver >= 7:
            _write_metadata_block(f, header.metadata or Metadata())
        table_pos  = _write_seek_table_placeholder(f, n_entries)
        seek_table: list = []

        for i, frame in enumerate(frames):
            if fver >= 8:
                # v8: seek table tracks FRAME_SYNC entries
                if frame.frame_type == FRAME_SYNC:
                    seek_table.append((i, f.tell()))
            else:
                # v6/v7: interval-based seek points
                if header.sync_interval > 0 and i % header.sync_interval == 0:
                    seek_table.append((i, f.tell()))
            write_frame(f, frame, header.entropy_mode, header.lpc_mode,
                        write_crc=(fver >= 7),
                        file_version=fver)

        _overwrite_seek_table(f, table_pos, seek_table)

    header.seek_table = seek_table


# ---------------------------------------------------------------------------
# Read — version-aware (v2 → v8)
# ---------------------------------------------------------------------------

def _peek_version(f: BinaryIO):
    raw = f.read(6)
    f.seek(0)
    if len(raw) < 6:
        return None, None
    return raw[:5], raw[5]


def _read_seek_table(f: BinaryIO) -> list:
    (n,) = struct.unpack(">I", f.read(4))
    table = []
    for _ in range(n):
        frame_idx, byte_offset = struct.unpack(
            _V8_SEEK_ENTRY, f.read(struct.calcsize(_V8_SEEK_ENTRY)))
        table.append((frame_idx, byte_offset))
    return table


def read_header(f: BinaryIO) -> HFPACHeader:
    """
    Read the global header — v2 through v8.

    v2 (22 bytes): float64 LPC, Huffman, one tree/frame
    v3 (23 bytes): float32 LPC, Huffman, shared block trees
    v4 (24 bytes): float32 LPC, Huffman, stereo_mode
    v5 (25 bytes): float32 LPC, Rice or Huffman, entropy_mode
    v6 (27 bytes): integer or float LPC, seek table
    v7 (27 bytes): same + metadata block + per-frame CRC
    v8 (27 bytes): same + frame_type byte, adaptive order, silence frames
    """
    magic, version = _peek_version(f)

    if magic != MAGIC:
        raise ValueError(
            f"Not a valid .hfpac file — bad magic bytes: {magic!r}\n"
            f"Expected: {MAGIC!r}"
        )
    if version < MIN_VERSION:
        raise ValueError(
            f"Unsupported .hfpac format: {display_version(version)}. "
            f"This build supports {display_version(MIN_VERSION)}–"
            f"{display_version(FORMAT_VERSION)}.\n"
            f"Please re-encode the file."
        )
    if version > FORMAT_VERSION:
        raise ValueError(
            f"File is from a newer version of HFPAC "
            f"({display_version(version)}), "
            f"this build only supports up to "
            f"{display_version(FORMAT_VERSION)}.\n"
            f"Please update HFPAC."
        )

    if version == 2:
        fmt = ">5sBIBBBBII"
        _, _, sr, ch, bd, lpc, fs, ns, nf = struct.unpack(
            fmt, f.read(struct.calcsize(fmt)))
        return HFPACHeader(sr, ch, bd, lpc, fs * FRAME_SIZE_DIVISOR, ns, nf,
                           frames_per_block=1, stereo_mode=STEREO_INDEPENDENT,
                           entropy_mode=ENTROPY_HUFFMAN, lpc_mode=LPC_FLOAT,
                           sync_interval=0, version=2, seek_table=[])

    if version == 3:
        fmt = ">5sBIBBBBIIB"
        _, _, sr, ch, bd, lpc, fs, ns, nf, fpb = struct.unpack(
            fmt, f.read(struct.calcsize(fmt)))
        return HFPACHeader(sr, ch, bd, lpc, fs * FRAME_SIZE_DIVISOR, ns, nf,
                           frames_per_block=fpb, stereo_mode=STEREO_INDEPENDENT,
                           entropy_mode=ENTROPY_HUFFMAN, lpc_mode=LPC_FLOAT,
                           sync_interval=0, version=3, seek_table=[])

    if version == 4:
        fmt = ">5sBIBBBBIIBB"
        _, _, sr, ch, bd, lpc, fs, ns, nf, fpb, sm = struct.unpack(
            fmt, f.read(struct.calcsize(fmt)))
        return HFPACHeader(sr, ch, bd, lpc, fs * FRAME_SIZE_DIVISOR, ns, nf,
                           frames_per_block=fpb, stereo_mode=sm,
                           entropy_mode=ENTROPY_HUFFMAN, lpc_mode=LPC_FLOAT,
                           sync_interval=0, version=4, seek_table=[])

    if version == 5:
        fmt = ">5sBIBBBBIIBBB"
        _, _, sr, ch, bd, lpc, fs, ns, nf, fpb, sm, em = struct.unpack(
            fmt, f.read(struct.calcsize(fmt)))
        return HFPACHeader(sr, ch, bd, lpc, fs * FRAME_SIZE_DIVISOR, ns, nf,
                           frames_per_block=fpb, stereo_mode=sm,
                           entropy_mode=em, lpc_mode=LPC_FLOAT,
                           sync_interval=0, version=5, seek_table=[])

    if version == 9:
        _, _, sr, ch, bd, lpc, fs, ns, nf, fpb, sm, em, lm, si, delay, pad = struct.unpack(
            _V9_HEADER_FMT, f.read(struct.calcsize(_V9_HEADER_FMT)))
    else:
        # v6, v7, v8 — identical 27-byte fixed header struct
        _, _, sr, ch, bd, lpc, fs, ns, nf, fpb, sm, em, lm, si = struct.unpack(
            _V8_HEADER_FMT, f.read(struct.calcsize(_V8_HEADER_FMT)))
        delay = 0
        pad = 0

    if version == 6:
        seek_table = _read_seek_table(f)
        return HFPACHeader(sr, ch, bd, lpc, fs * FRAME_SIZE_DIVISOR, ns, nf,
                           frames_per_block=fpb, stereo_mode=sm,
                           entropy_mode=em, lpc_mode=lm,
                           sync_interval=si, encoder_delay=delay, trailing_padding=pad,
                           version=6, seek_table=seek_table, metadata=Metadata())

    # v7 and v8+ both have metadata block before seek table
    metadata   = _read_metadata_block(f)
    seek_table = _read_seek_table(f)

    if version == 7:
        return HFPACHeader(sr, ch, bd, lpc, fs * FRAME_SIZE_DIVISOR, ns, nf,
                           frames_per_block=fpb, stereo_mode=sm,
                           entropy_mode=em, lpc_mode=lm,
                           sync_interval=si, encoder_delay=delay, trailing_padding=pad,
                           version=7, seek_table=seek_table, metadata=metadata)

    if version == 8:
        # v8 (v6)
        return HFPACHeader(sr, ch, bd, lpc, fs * FRAME_SIZE_DIVISOR, ns, nf,
                           frames_per_block=fpb, stereo_mode=sm,
                           entropy_mode=em, lpc_mode=lm,
                           sync_interval=si, encoder_delay=delay, trailing_padding=pad,
                           version=8, seek_table=seek_table, metadata=metadata)

    # v9 (v6.1)
    return HFPACHeader(sr, ch, bd, lpc, fs * FRAME_SIZE_DIVISOR, ns, nf,
                       frames_per_block=fpb, stereo_mode=sm,
                       entropy_mode=em, lpc_mode=lm,
                       sync_interval=si, encoder_delay=delay, trailing_padding=pad,
                       version=9, seek_table=seek_table, metadata=metadata)


# ---------------------------------------------------------------------------
# Per-version frame readers
# ---------------------------------------------------------------------------

def _read_frame_v2(f: BinaryIO) -> EncodedFrame:
    """v2: float64 LPC, Huffman, tree always present."""
    (n,)   = struct.unpack(">I", f.read(4))
    coeffs = np.array([struct.unpack(">d", f.read(8))[0] for _ in range(n)],
                      dtype=np.float64)
    (ts,)  = struct.unpack(">I", f.read(4))
    tree   = f.read(ts)
    (nb,)  = struct.unpack(">I", f.read(4))
    (ps,)  = struct.unpack(">I", f.read(4))
    return EncodedFrame(lpc_coeffs=coeffs, huffman_tree=tree,
                        num_bits=nb, huffman_payload=f.read(ps))
    (nb,)   = struct.unpack(">I", f.read(4))
    (ps,)   = struct.unpack(">I", f.read(4))
    return EncodedFrame(lpc_coeffs=coeffs, huffman_tree=tree,
                        num_bits=nb, huffman_payload=f.read(ps))


def _read_lpc_float(f: BinaryIO):
    """Read float32 LPC coefficients."""
    (n,)   = struct.unpack(">I", f.read(4))
    coeffs = np.frombuffer(f.read(n * 4), dtype=">f4").astype(np.float64)
    return coeffs, None, 0


def _read_lpc_int(f: BinaryIO):
    """Read int16 LPC coefficients with precision byte."""
    (n,)         = struct.unpack(">I", f.read(4))
    (precision,) = struct.unpack(">B", f.read(1))
    coeffs_int   = np.frombuffer(f.read(n * 2), dtype=">i2").astype(np.int16)
    coeffs_f64   = coeffs_int.astype(np.float64) / float(1 << precision)
    return coeffs_f64, coeffs_int, precision


def _read_entropy_huffman(f: BinaryIO) -> dict:
    (ts,)  = struct.unpack(">I", f.read(4))
    tree   = f.read(ts) if ts > 0 else None
    (nb,)  = struct.unpack(">I", f.read(4))
    (ps,)  = struct.unpack(">I", f.read(4))
    return dict(huffman_tree=tree, num_bits=nb, huffman_payload=f.read(ps))


def _read_entropy_rice(f: BinaryIO) -> dict:
    (k,)  = struct.unpack(">B", f.read(1))
    (nb,) = struct.unpack(">I", f.read(4))
    (ps,) = struct.unpack(">I", f.read(4))
    return dict(rice_k=k, num_bits=nb, rice_payload=f.read(ps))


def _read_lpc_and_entropy(f, entropy_mode: int, lpc_mode: int) -> EncodedFrame:
    """Read LPC coefficients + entropy payload (shared by v5–v8 LPC frames)."""
    if lpc_mode == LPC_INTEGER:
        coeffs, coeffs_int, precision = _read_lpc_int(f)
    else:
        coeffs, coeffs_int, precision = _read_lpc_float(f)
    ent = (_read_entropy_rice(f) if entropy_mode == ENTROPY_RICE
           else _read_entropy_huffman(f))
    return EncodedFrame(lpc_coeffs=coeffs, lpc_coeffs_int=coeffs_int,
                        lpc_precision=precision, **ent)


def _read_frame_v5(f: BinaryIO, entropy_mode: int) -> EncodedFrame:
    """v5 (v4.5): float32 LPC, no frame-type byte."""
    coeffs, _, _ = _read_lpc_float(f)
    ent = (_read_entropy_rice(f) if entropy_mode == ENTROPY_RICE
           else _read_entropy_huffman(f))
    return EncodedFrame(lpc_coeffs=coeffs, **ent)


def _read_frame_v6_or_v7(f: BinaryIO, entropy_mode: int,
                          lpc_mode: int, frame_number: int,
                          has_crc: bool) -> EncodedFrame:
    """
    v6 (int 6): no frame-type byte, no CRC.
    v7 (int 7): no frame-type byte, with CRC-32.
    """
    if has_crc:
        tracker = _ReadTracker(f)
        frame   = _read_lpc_and_entropy(tracker, entropy_mode, lpc_mode)
        stored_crc, = struct.unpack(">I", f.read(4))
        _verify_crc(tracker.captured, stored_crc, frame_number)
        return frame
    return _read_lpc_and_entropy(f, entropy_mode, lpc_mode)


def _read_frame_v8(f: BinaryIO, entropy_mode: int,
                   lpc_mode: int, frame_number: int) -> EncodedFrame:
    """
    v8 (v6): frame_type byte + LPC/silence content + CRC-32.
    """
    tracker = _ReadTracker(f)
    (frame_type,) = struct.unpack(">B", tracker.read(1))

    if frame_type == FRAME_SILENCE:
        (silence_value,) = struct.unpack(">i", tracker.read(4))
        stored_crc, = struct.unpack(">I", f.read(4))
        _verify_crc(tracker.captured, stored_crc, frame_number)
        return EncodedFrame(lpc_coeffs=np.zeros(0),
                            frame_type=FRAME_SILENCE,
                            silence_value=silence_value)

    frame = _read_lpc_and_entropy(tracker, entropy_mode, lpc_mode)
    frame.frame_type = frame_type
    stored_crc, = struct.unpack(">I", f.read(4))
    _verify_crc(tracker.captured, stored_crc, frame_number)
    return frame


def _verify_crc(frame_bytes: bytes, stored_crc: int, frame_number: int) -> None:
    computed = _crc32(frame_bytes)
    if stored_crc != computed:
        raise ValueError(
            f"CRC mismatch in frame {frame_number} — file is corrupted.\n"
            f"  Stored:   {stored_crc:#010x}\n"
            f"  Computed: {computed:#010x}"
        )


def read_hfpac(path: str, progress_callback=None):
    """
    Read a complete .hfpac file — v2 through v8.

    v7+: verifies per-frame CRC-32.
    v8:  additionally handles FRAME_SYNC/CONT/SILENCE frame types.
    v3–v7 Huffman: propagates shared block trees.

    Returns:
        (header, frames) — HFPACHeader and list of EncodedFrame
    """
    with open(path, "rb") as f:
        header = read_header(f)

        if header.version == 2:
            def _read_one(i): return _read_frame_v2(f)
        elif header.version <= 5:
            def _read_one(i): return _read_frame_v5(f, header.entropy_mode)
        elif header.version <= 7:
            has_crc = (header.version == 7)
            def _read_one(i): return _read_frame_v6_or_v7(
                f, header.entropy_mode, header.lpc_mode, i, has_crc)
        else:
            def _read_one(i): return _read_frame_v8(
                f, header.entropy_mode, header.lpc_mode, i)

        frames             = []
        current_tree_bytes = None

        for i in range(header.num_frames):
            frame = _read_one(i)

            # Propagate shared Huffman block trees (v3–v7 only;
            # v8 uses adaptive order so every LPC frame has its own coeffs)
            if 3 <= header.version <= 7 and header.entropy_mode == ENTROPY_HUFFMAN:
                if frame.huffman_tree is not None:
                    current_tree_bytes = frame.huffman_tree
                elif getattr(frame, 'frame_type', FRAME_SYNC) != FRAME_SILENCE:
                    frame = EncodedFrame(
                        lpc_coeffs      = frame.lpc_coeffs,
                        lpc_coeffs_int  = frame.lpc_coeffs_int,
                        lpc_precision   = frame.lpc_precision,
                        frame_type      = getattr(frame, 'frame_type', FRAME_SYNC),
                        huffman_tree    = current_tree_bytes,
                        num_bits        = frame.num_bits,
                        huffman_payload = frame.huffman_payload,
                    )

            frames.append(frame)

            if progress_callback and i % max(1, header.num_frames // 100) == 0:
                progress_callback(i, header.num_frames)
                import time
                time.sleep(0.001)  # Release the GIL briefly so the GUI thread can update

        if progress_callback:
            progress_callback(header.num_frames, header.num_frames)

    return header, frames


def verify_hfpac(path: str) -> dict:
    """
    Verify the integrity of a .hfpac file without decoding audio.

    For v7–v8 files: checks per-frame CRC-32 and metadata CRC-32.
    For v2–v6 files: confirms the file is readable and not truncated
    (no CRC data is available in those versions).

    Returns a dict with keys:
        version (int), ok (bool), frames_checked (int),
        first_bad_frame (int or None), error (str or None)
    """
    result = dict(version=None, ok=False, frames_checked=0,
                  first_bad_frame=None, error=None)
    try:
        header, frames = read_hfpac(path)
        result['version']        = header.version
        result['frames_checked'] = len(frames)
        result['ok']             = True
        if header.version < 7:
            result['error'] = (
                f"{display_version(header.version)} files have no CRC data — "
                f"readability confirmed but integrity cannot be verified."
            )
    except ValueError as e:
        msg = str(e)
        result['error'] = msg
        if 'frame' in msg.lower():
            import re
            m = re.search(r'frame (\d+)', msg, re.IGNORECASE)
            if m:
                result['first_bad_frame'] = int(m.group(1))
    return result


def seek_to_frame(path: str, frame_idx: int):
    """
    Open a file and seek directly to a specific frame using the seek table.

    For v8 (v6) files: all seek-table entries point to FRAME_SYNC frames,
    so the caller can start decoding with zeroed history at the returned offset.

    Returns:
        (file_handle, header, best_frame_idx) — f positioned at that frame.
        Caller is responsible for closing the file handle.

    Raises ValueError if the file has no seek table (v2–v5).
    """
    with open(path, "rb") as f:
        header = read_header(f)

    if header.version < 6 or not header.seek_table:
        raise ValueError(
            "This file has no seek table. Re-encode with HFPAC v6+ to enable seeking."
        )

    best_idx    = 0
    best_offset = None
    for fi, offset in header.seek_table:
        if fi <= frame_idx:
            best_idx    = fi
            best_offset = offset

    if best_offset is None:
        raise ValueError(f"No seek point found for frame {frame_idx}.")

    f = open(path, "rb")
    read_header(f)   # advance past header + metadata + seek table
    f.seek(best_offset)
    return f, header, best_idx


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, ".")
    from lpc import (
        compute_lpc_coefficients, encode_frame_int, decode_frame_int,
        quantize_lpc_coefficients, split_into_frames,
        FRAME_SIZE, DEFAULT_LPC_ORDER, DEFAULT_LPC_PRECISION,
    )
    from rice import choose_k, encode as rice_enc, decode as rice_dec

    print("=== HFPAC hfpac_format.py smoke test (v6) ===\n")

    sr       = 44100
    duration = FRAME_SIZE * 4   # 4 frames per channel → 2 SYNC + 2 CONT
    t        = np.linspace(0, duration / sr, duration, endpoint=False)
    ch1      = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.float64)
    ch2      = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.float64)

    # Build v6 frames: alternate SYNC / CONT, include one SILENCE frame
    encoded_frames: List[EncodedFrame] = []
    for ch_idx, ch in enumerate([ch1, ch2]):
        for frame_idx, (_, fs) in enumerate(split_into_frames(ch, FRAME_SIZE)):
            # Make one silence frame for testing
            if ch_idx == 0 and frame_idx == 2:
                encoded_frames.append(EncodedFrame(
                    lpc_coeffs    = np.zeros(DEFAULT_LPC_ORDER),
                    frame_type    = FRAME_SILENCE,
                    silence_value = 0,
                ))
                continue

            coeffs      = compute_lpc_coefficients(fs, DEFAULT_LPC_ORDER)
            coeffs_int  = quantize_lpc_coefficients(coeffs, DEFAULT_LPC_PRECISION)
            residuals   = encode_frame_int(fs, coeffs_int, DEFAULT_LPC_PRECISION).tolist()
            k           = choose_k(residuals)
            payload, nb = rice_enc(residuals, k)
            coeffs_f64  = coeffs_int.astype(np.float64) / float(1 << DEFAULT_LPC_PRECISION)
            ftype       = FRAME_SYNC if frame_idx % SYNC_INTERVAL == 0 else FRAME_CONT
            encoded_frames.append(EncodedFrame(
                lpc_coeffs     = coeffs_f64,
                lpc_coeffs_int = coeffs_int,
                lpc_precision  = DEFAULT_LPC_PRECISION,
                frame_type     = ftype,
                rice_k         = k,
                rice_payload   = payload,
                num_bits       = nb,
            ))

    out_path = "test_output.hfpac"
    meta = Metadata(title="Smoke Test", artist="HFPAC",
                    album="Unit Tests", track_number=1, year=2026)
    header = HFPACHeader(
        sample_rate   = sr,
        channels      = 2,
        bit_depth     = 16,
        lpc_order     = DEFAULT_LPC_ORDER,
        frame_size    = FRAME_SIZE,
        num_samples   = duration,
        num_frames    = len(encoded_frames),
        entropy_mode  = ENTROPY_RICE,
        lpc_mode      = LPC_INTEGER,
        sync_interval = SYNC_INTERVAL,
        metadata      = meta,
    )
    write_hfpac(out_path, header, encoded_frames)

    raw_size  = duration * 2 * 2
    file_size = os.path.getsize(out_path)
    print(f"Written:          {out_path}")
    print(f"Format version:   v{FORMAT_VERSION}  (displayed as v6)")
    print(f"Raw PCM size:     {raw_size:,} bytes")
    print(f".hfpac size:      {file_size:,} bytes")
    print(f"Seek entries:     {len(header.seek_table)}")

    # Read back and verify
    r_header, r_frames = read_hfpac(out_path)
    assert r_header.sample_rate  == sr
    assert r_header.channels     == 2
    assert r_header.num_frames   == len(encoded_frames)
    assert r_header.version      == FORMAT_VERSION
    assert r_header.lpc_mode     == LPC_INTEGER
    assert r_header.metadata.title  == "Smoke Test"
    assert r_header.metadata.year   == 2026

    # Count frame types
    sync_count     = sum(1 for f in r_frames if f.frame_type == FRAME_SYNC)
    cont_count     = sum(1 for f in r_frames if f.frame_type == FRAME_CONT)
    silence_count  = sum(1 for f in r_frames if f.frame_type == FRAME_SILENCE)
    print(f"\nFrame types:      SYNC={sync_count}  CONT={cont_count}  SILENCE={silence_count}")
    assert silence_count == 1,  "Expected 1 silence frame"
    assert sync_count    >= 1,  "Expected at least 1 sync frame"

    # Verify seek table only contains SYNC frames
    sync_frame_indices = {i for i, f in enumerate(r_frames) if f.frame_type == FRAME_SYNC}
    for fi, _ in r_header.seek_table:
        assert fi in sync_frame_indices, \
            f"Seek entry at frame {fi} is not a SYNC frame"
    print(f"Seek table:       all {len(r_header.seek_table)} entries point to SYNC frames ✅")

    # Verify CRC
    result = verify_hfpac(out_path)
    assert result['ok'], f"verify_hfpac failed: {result['error']}"
    print(f"CRC verification: PASS  ({result['frames_checked']} frames) ✅")

    # Verify metadata CRC
    assert r_header.metadata.title  == "Smoke Test"
    assert r_header.metadata.artist == "HFPAC"
    print(f"Metadata:         {r_header.metadata.title} — "
          f"{r_header.metadata.artist} ({r_header.metadata.year}) ✅")

    os.remove(out_path)
    print(f"\n✅  All smoke test assertions passed (v{FORMAT_VERSION} / v6).")