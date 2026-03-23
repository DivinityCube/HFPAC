"""
rice.py — Rice Coding for HFPAC
=================================
Rice coding is an entropy coder tuned specifically for the Laplacian
distribution that LPC residuals follow — most values near zero, falling
off exponentially for larger magnitudes.

How it works
------------
Each integer residual is encoded in three parts:

    1. Map signed → unsigned via zigzag:   0 → 0,  -1 → 1,  1 → 2,  -2 → 3 ...
       This puts the most frequent value (0) at position 0.

    2. Split by a parameter k:
         quotient  = n >> k       (upper bits)
         remainder = n &  mask    (lower k bits)

    3. Write quotient in unary (that many 1-bits followed by a 0),
       then remainder in binary (exactly k bits).

For k=3 and residual 0:  q=0, r=0 → "0" + "000" = 4 bits total
For k=3 and residual 1:  q=0, r=2 → "0" + "010" = 4 bits total  (zigzag: 1→2)
For k=3 and residual 5:  q=0, r=10 → "0"+"1010" = 5 bits total  (zigzag: 5→10)

Vectorised implementation (v5.2)
---------------------------------
encode() and choose_k() operate entirely on numpy arrays — no Python loops
over individual residuals. This makes encoding roughly 10–20× faster than
the original loop-based implementation, particularly for large frames.

The decode() function uses np.unpackbits and a sequential state machine;
fully vectorising the variable-length unary decode is complex and the
current implementation is already fast enough for real-time playback.

Why Rice is better than Huffman for HFPAC
------------------------------------------
- No tree to store (saves ~100 bytes per block = thousands of bytes per file)
- No tree-building pass (faster encode)
- No tree-walk (faster decode — arithmetic instead of pointer chasing)
- Provably near-optimal for Laplacian distributions
- Single-pass encode: just write bits directly
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Zigzag mapping — vectorised
# ---------------------------------------------------------------------------

def _zigzag_arr(arr: np.ndarray) -> np.ndarray:
    """Vectorised zigzag encode: maps signed int array to unsigned."""
    a = arr.astype(np.int32)
    return ((a << 1) ^ (a >> 31)).astype(np.uint32)


def _zigzag_decode(n: int) -> int:
    """Scalar zigzag decode: 0→0, 1→-1, 2→1, 3→-2, 4→2 ..."""
    return (n >> 1) ^ -(n & 1)


# ---------------------------------------------------------------------------
# Choose optimal k — vectorised
# ---------------------------------------------------------------------------

def choose_k(residuals: List[int], k_min: int = 0, k_max: int = 30) -> int:
    """
    Find the Rice parameter k that minimises total encoded bits.

    Uses numpy for zigzag and cost computation — no Python loop over
    individual residuals.

    Estimator improvement (v5.4)
    ────────────────────────────
    For a geometric distribution (discrete approximation to the Laplacian
    distribution that LPC residuals follow), the optimal Rice parameter
    satisfies:

        2^k ≈ mean_unsigned / log(2)

    So the closed-form estimate is:

        k_est = round( log2(mean / log(2)) )
              = round( log2(mean) − log2(log(2)) )
              = round( log2(mean) + 1.5129... )

    This is more accurate than the previous `int(log2(mean + 1))` heuristic,
    which systematically underestimated k by ~0.5.  With this tighter
    estimate only a ±1 search is needed — half the cost evaluations of the
    old ±2 search.

    k_max defaults to 30, covering 16-bit (optimal k ≤ 15) and 24-bit
    audio (optimal k ≤ ~23).  k is stored as uint8 in the format.

    Args:
        residuals — list or array of signed integer LPC residuals
        k_min, k_max — search bounds

    Returns:
        optimal k (int, k_min ≤ k ≤ k_max)
    """
    if not residuals:
        return 0

    u        = _zigzag_arr(np.asarray(residuals, dtype=np.int32))
    mean_val = float(u.mean())

    if mean_val <= 0:
        return 0

    import math
    # Closed-form optimal estimator for geometric/Laplacian residuals:
    # k ≈ round(log2(mean / log(2)))
    # log2(log(2)) ≈ −0.5129, so this adds ~0.5 to the old estimate,
    # correcting the systematic undershoot.
    k_est = max(k_min, min(k_max,
                round(math.log2(mean_val / math.log(2) + 1e-10))))

    best_k    = k_est
    best_bits = int(np.iinfo(np.int64).max)

    # ±2 search around the estimate — wider window guarantees we find the
    # true optimum even when the estimator is slightly off.
    # The improved estimator still reduces the typical search midpoint error,
    # making k_est[0] already correct more often.
    for k in range(max(k_min, k_est - 2), min(k_max + 1, k_est + 3)):
        quotients = u >> k
        cost      = int(quotients.sum()) + len(u) * (k + 1)
        if cost < best_bits:
            best_bits = cost
            best_k    = k

    return best_k


# ---------------------------------------------------------------------------
# Encode — fully vectorised
# ---------------------------------------------------------------------------

from numba import njit

@njit(cache=True)
def _encode_numba(residuals: np.ndarray, k: int):
    n = len(residuals)
    total_bits = 0
    for i in range(n):
        r = residuals[i]
        u = (r << 1) ^ (r >> 31)
        q = u >> k
        total_bits += q + 1 + k

    num_bytes = (total_bits + 7) // 8
    out = np.zeros(num_bytes, dtype=np.uint8)

    bit_pos = 0
    for i in range(n):
        r = residuals[i]
        u = (r << 1) ^ (r >> 31)
        q = u >> k
        rem = u & ((1 << k) - 1)

        for _ in range(q):
            out[bit_pos >> 3] |= (1 << (7 - (bit_pos & 7)))
            bit_pos += 1

        # The 0 terminator is already implicitly there from np.zeros
        bit_pos += 1

        for j in range(k - 1, -1, -1):
            if (rem >> j) & 1:
                out[bit_pos >> 3] |= (1 << (7 - (bit_pos & 7)))
            bit_pos += 1

    return out, total_bits

def encode(residuals: List[int], k: int) -> Tuple[bytes, int]:
    """
    Rice-encode a list of signed integer residuals.

    Algorithm
    ---------
    For each unsigned value u = zigzag(r):
        · quotient  = u >> k          → written as (quotient) 1-bits then a 0
        · remainder = u & ((1<<k)-1)  → written as k bits, MSB first

    Optimized via Numba @njit cache=True to avoid memory allocations and keep it fast.

    Args:
        residuals — list of signed integers (LPC residuals)
        k         — Rice parameter (from choose_k)

    Returns:
        (encoded_bytes, num_bits) — packed bytes and valid bit count
    """
    if len(residuals) == 0:
        return b"", 0

    arr = np.asarray(residuals, dtype=np.int32)
    out_arr, num_bits = _encode_numba(arr, k)
    return out_arr.tobytes(), num_bits


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode(encoded_bytes: bytes, k: int, num_bits: int, num_residuals: int) -> List[int]:
    """
    Rice-decode bytes back into signed integer residuals.

    Args:
        encoded_bytes — bytes from encode()
        k             — Rice parameter (same k used to encode)
        num_bits      — valid bit count (from encode())
        num_residuals — expected number of output residuals

    Returns:
        list of signed integer residuals
    """
    if not encoded_bytes or num_residuals == 0:
        return []

    bits      = np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8))[:num_bits]
    residuals = []
    pos       = 0
    n         = len(bits)

    while len(residuals) < num_residuals and pos < n:
        # Read unary quotient: count 1-bits until a 0
        quotient = 0
        while pos < n and bits[pos] == 1:
            quotient += 1
            pos      += 1
        pos += 1   # skip terminating 0

        # Read k-bit remainder
        if pos + k > n:
            break
        remainder = 0
        for i in range(k):
            remainder = (remainder << 1) | int(bits[pos])
            pos += 1

        u = (quotient << k) | remainder
        residuals.append(_zigzag_decode(u))

    return residuals


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    from lpc import compute_lpc_coefficients, encode_frame, decode_frame, FRAME_SIZE

    print("=== HFPAC rice.py smoke test ===\n")

    # Generate a test signal
    sr = 44100
    t  = np.linspace(0, FRAME_SIZE / sr, FRAME_SIZE, endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.float64)

    coeffs    = compute_lpc_coefficients(signal)
    residuals = encode_frame(signal, coeffs).tolist()

    # Choose k and encode
    k                  = choose_k(residuals)
    encoded, num_bits  = encode(residuals, k)
    decoded_residuals  = decode(encoded, k, num_bits, len(residuals))

    assert decoded_residuals == residuals, "Rice roundtrip FAILED!"

    # Reconstruct audio
    reconstructed = decode_frame(np.array(decoded_residuals, dtype=np.int32), coeffs)
    max_err       = np.max(np.abs(signal - reconstructed))

    # Compare sizes
    raw_bytes    = len(residuals) * 2        # 16-bit raw
    rice_bytes   = len(encoded)
    k_overhead   = 1                         # 1 byte to store k

    print(f"Rice parameter k:       {k}")
    print(f"Raw residuals:          {raw_bytes} bytes")
    print(f"Rice encoded:           {rice_bytes} bytes  (+{k_overhead} byte for k)")
    print(f"Compression ratio:      {raw_bytes / (rice_bytes + k_overhead):.2f}x")
    print(f"Max reconstruction err: {max_err:.4f}  (should be < 1.0)")
    print(f"\n✅ Rice encode/decode roundtrip complete!")