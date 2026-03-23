"""
metrics.py — Quality & Compression Metrics for HFPAC
=====================================================
Provides objective measurements to evaluate how well HFPAC performed:

  SNR   (Signal-to-Noise Ratio)
        How much louder the original signal is than the error.
        Higher = better. Lossless codecs typically exceed 80 dB.

  PSNR  (Peak Signal-to-Noise Ratio)
        Like SNR but normalised to the maximum possible signal level.
        Useful for comparing across recordings with different volumes.
        > 96 dB is considered transparent (indistinguishable from original).

  File size & compression ratio
        Raw WAV bytes vs .hfpac bytes.

  Bit rate
        How many kilobits per second the compressed file uses.
        CD quality WAV = 1411 kbps. We aim to be well below that.

  Encoding / decoding speed
        Reported as a multiple of real-time (e.g. 10× means we can
        encode 10 seconds of audio per second of wall-clock time).
"""

from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Signal quality
# ---------------------------------------------------------------------------

def compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio in decibels.

        SNR = 10 · log10( Σ x² / Σ (x - x̂)² )

    Args:
        original      — original PCM samples (any shape, float64)
        reconstructed — decoded PCM samples (same shape)

    Returns:
        SNR in dB (float). Returns inf if reconstruction is perfect.
    """
    signal_power = np.sum(original.astype(np.float64) ** 2)
    noise        = original.astype(np.float64) - reconstructed.astype(np.float64)
    noise_power  = np.sum(noise ** 2)

    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray, bit_depth: int = 16) -> float:
    """
    Peak Signal-to-Noise Ratio in decibels.

        PSNR = 10 · log10( PEAK² / MSE )

    where PEAK is the maximum representable value for the bit depth
    and MSE is the mean squared error.

    Args:
        original      — original PCM samples (float64, integer-scaled)
        reconstructed — decoded PCM samples
        bit_depth     — bit depth of the audio (16 or 24)

    Returns:
        PSNR in dB (float). Returns inf if reconstruction is perfect.
    """
    peak = float(2 ** (bit_depth - 1))
    mse  = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)

    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(peak ** 2 / mse)


def compute_max_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Maximum absolute sample error between original and reconstructed."""
    return float(np.max(np.abs(
        original.astype(np.float64) - reconstructed.astype(np.float64)
    )))


# ---------------------------------------------------------------------------
# Compression & bitrate
# ---------------------------------------------------------------------------

def compression_ratio(original_path: str, compressed_path: str) -> float:
    """Ratio of original file size to compressed file size."""
    orig = Path(original_path).stat().st_size
    comp = Path(compressed_path).stat().st_size
    return orig / comp


def space_saving(original_path: str, compressed_path: str) -> float:
    """Percentage of space saved by compression (0–100)."""
    orig = Path(original_path).stat().st_size
    comp = Path(compressed_path).stat().st_size
    return (1.0 - comp / orig) * 100.0


def bitrate_kbps(compressed_path: str, duration_seconds: float) -> float:
    """
    Compressed bitrate in kilobits per second.

    Args:
        compressed_path  — path to the .hfpac file
        duration_seconds — audio duration in seconds

    Returns:
        bitrate in kbps
    """
    size_bits = Path(compressed_path).stat().st_size * 8
    return size_bits / duration_seconds / 1000.0


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

def realtime_factor(audio_duration_s: float, processing_time_s: float) -> float:
    """
    How many times faster than real-time the codec ran.

        factor = audio_duration / processing_time

    e.g. factor = 5.0 means 5 seconds of audio encoded per second of wall time.
    """
    if processing_time_s == 0:
        return float("inf")
    return audio_duration_s / processing_time_s


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def compare_wav_files(
    original_path: str,
    reconstructed_path: str,
    compressed_path:    Optional[str] = None,
    encode_time:        Optional[float] = None,
    decode_time:        Optional[float] = None,
) -> dict:
    """
    Load both WAV files and produce a full quality + compression report.

    Args:
        original_path     — path to the original .wav
        reconstructed_path — path to the decoded .wav
        compressed_path   — path to the .hfpac file (optional, for size stats)
        encode_time       — seconds taken to encode (optional)
        decode_time       — seconds taken to decode (optional)

    Returns:
        dict of all metrics (also prints a formatted report to stdout)
    """
    # Load both files at float64, scaled to integer range
    orig_data,  orig_sr  = sf.read(original_path,      dtype="float64")
    reco_data,  reco_sr  = sf.read(reconstructed_path, dtype="float64")
    orig_info            = sf.info(original_path)

    bit_depth = 24 if "24" in orig_info.subtype_info else 16
    scale     = float(2 ** (bit_depth - 1))
    orig_data = (orig_data * scale).astype(np.float64)
    reco_data = (reco_data * scale).astype(np.float64)

    # Trim to same length (last frame may add a tiny amount of padding)
    min_len   = min(len(orig_data), len(reco_data))
    orig_data = orig_data[:min_len]
    reco_data = reco_data[:min_len]

    duration  = min_len / orig_sr
    channels  = orig_data.shape[1] if orig_data.ndim == 2 else 1

    # Quality metrics
    snr   = compute_snr(orig_data,  reco_data)
    psnr  = compute_psnr(orig_data, reco_data, bit_depth)
    max_e = compute_max_error(orig_data, reco_data)
    lsb   = 1.0  # 1 LSB in integer-scaled space

    # File size metrics
    orig_size = Path(original_path).stat().st_size
    results = {
        "snr_db":       snr,
        "psnr_db":      psnr,
        "max_error":    max_e,
        "lossless":     max_e <= lsb,
        "sample_rate":  orig_sr,
        "channels":     channels,
        "bit_depth":    bit_depth,
        "duration_s":   duration,
        "orig_size":    orig_size,
    }

    if compressed_path:
        comp_size = Path(compressed_path).stat().st_size
        ratio     = orig_size / comp_size
        saving    = (1.0 - comp_size / orig_size) * 100.0
        brate     = bitrate_kbps(compressed_path, duration)
        wav_brate = (orig_size * 8) / duration / 1000.0
        results.update({
            "comp_size":          comp_size,
            "compression_ratio":  ratio,
            "space_saving_pct":   saving,
            "hfpac_kbps":         brate,
            "wav_kbps":           wav_brate,
        })

    if encode_time is not None:
        results["encode_time_s"]  = encode_time
        results["encode_rtf"]     = realtime_factor(duration, encode_time)
    if decode_time is not None:
        results["decode_time_s"]  = decode_time
        results["decode_rtf"]     = realtime_factor(duration, decode_time)

    # ---- Print report ----
    _print_report(results)
    return results


def _print_report(r: dict) -> None:
    """Print a formatted metrics report to stdout."""
    sep = "─" * 52

    print(f"\n{'═' * 52}")
    print(f"  HFPAC Quality & Compression Report")
    print(f"{'═' * 52}")

    print(f"\n  Audio")
    print(f"  {sep}")
    print(f"  Sample rate   {r['sample_rate']:>10,} Hz")
    print(f"  Channels      {r['channels']:>10}")
    print(f"  Bit depth     {r['bit_depth']:>10}-bit")
    print(f"  Duration      {r['duration_s']:>10.2f} s")

    print(f"\n  Quality")
    print(f"  {sep}")
    snr_str  = f"{r['snr_db']:.1f} dB"  if r['snr_db']  != float('inf') else "∞ (perfect)"
    psnr_str = f"{r['psnr_db']:.1f} dB" if r['psnr_db'] != float('inf') else "∞ (perfect)"
    print(f"  SNR           {snr_str:>14}")
    print(f"  PSNR          {psnr_str:>14}")
    print(f"  Max error     {r['max_error']:>14.6f}")
    lossless_label = "✅ YES" if r['lossless'] else "❌ NO (lossy)"
    print(f"  Lossless      {lossless_label:>14}")

    if "comp_size" in r:
        print(f"\n  Compression")
        print(f"  {sep}")
        print(f"  WAV size      {r['orig_size']:>10,} bytes")
        print(f"  HFPAC size    {r['comp_size']:>10,} bytes")
        print(f"  Ratio         {r['compression_ratio']:>10.2f}×")
        print(f"  Space saved   {r['space_saving_pct']:>10.1f}%")
        print(f"  WAV bitrate   {r['wav_kbps']:>10.1f} kbps")
        print(f"  HFPAC bitrate {r['hfpac_kbps']:>10.1f} kbps")

    if "encode_time_s" in r:
        print(f"\n  Speed")
        print(f"  {sep}")
        print(f"  Encode time   {r['encode_time_s']:>10.2f} s")
        print(f"  Encode speed  {r['encode_rtf']:>9.1f}× realtime")
    if "decode_time_s" in r:
        print(f"  Decode time   {r['decode_time_s']:>10.2f} s")
        print(f"  Decode speed  {r['decode_rtf']:>9.1f}× realtime")

    print(f"\n{'═' * 52}\n")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, ".")
    from codec import encode_wav, decode_hfpac

    print("=== HFPAC metrics.py smoke test ===\n")

    # Generate a 2-second stereo test signal
    sr  = 44100
    n   = sr * 2
    t   = np.linspace(0, 2.0, n, endpoint=False)
    ch1 = np.sin(2 * np.pi * 440 * t) * 0.8
    ch2 = np.sin(2 * np.pi * 880 * t) * 0.6 + np.sin(2 * np.pi * 1320 * t) * 0.2
    import soundfile as sf
    sf.write("metrics_test.wav", np.stack([ch1, ch2], axis=1), sr, subtype="PCM_16")

    enc_stats = encode_wav("metrics_test.wav", "metrics_test.hfpac")
    dec_stats = decode_hfpac("metrics_test.hfpac", "metrics_test_decoded.wav")

    compare_wav_files(
        original_path      = "metrics_test.wav",
        reconstructed_path = "metrics_test_decoded.wav",
        compressed_path    = "metrics_test.hfpac",
        encode_time        = enc_stats["encode_time"],
        decode_time        = dec_stats["decode_time"],
    )

    for f in ["metrics_test.wav", "metrics_test.hfpac", "metrics_test_decoded.wav"]:
        os.remove(f)

    print("✅ Metrics smoke test complete!")