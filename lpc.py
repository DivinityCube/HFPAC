"""
lpc.py — Linear Predictive Coding for HFPAC
============================================
LPC is the heart of the codec. The core idea:

  "Given the last N audio samples, predict the next one."

Because audio is highly correlated (neighbouring samples are similar),
the prediction is usually very close — meaning the *residual*
(actual − predicted) is a small number. Small numbers compress well.

This is the same technique used in FLAC and Shorten.

Pipeline for a single frame:
  encode: samples → LPC coefficients + residuals
  decode: LPC coefficients + residuals → samples (perfectly reconstructed)
"""

import os
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".numba_cache"),
)

import numpy as np
from typing import Tuple
from numba import njit


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LPC_ORDER     = 12   # 12 coefficients — good balance for music/speech
FRAME_SIZE            = 1024 # samples per frame (≈23ms at 44100 Hz)
DEFAULT_LPC_PRECISION = 13   # bits of precision for integer LPC coefficients
                             # 2^13 = 8192 quantization levels — 12–15 is the
                             # sweet spot; 13 matches common FLAC implementations

# LPC mode constants (mirrored in hfpac_format.py)
LPC_FLOAT   = 0   # float32 coefficients (v3–v4.5 behaviour)
LPC_INTEGER = 1   # int16 coefficients with precision shift (v5 default)


# ---------------------------------------------------------------------------
# Step 1 — Analysis window
# ---------------------------------------------------------------------------

def _blackman_harris_window(n: int) -> np.ndarray:
    """
    Blackman-Harris window of length n.

    Chosen over Hann for LPC analysis because it has much lower sidelobes
    (−92 dB vs −31 dB), giving a more accurate spectral representation of
    the frame.  That accuracy translates directly into lower residual energy
    after LPC prediction.

    The window still tapers to near-zero at the edges, avoiding the
    discontinuity problem that makes unwindowed autocorrelation unreliable.
    """
    a = [0.35875, 0.48829, 0.14128, 0.01168]
    k = np.arange(n, dtype=np.float64)
    w = (a[0]
         - a[1] * np.cos(2 * np.pi * k / (n - 1))
         + a[2] * np.cos(4 * np.pi * k / (n - 1))
         - a[3] * np.cos(6 * np.pi * k / (n - 1)))
    return w


# ---------------------------------------------------------------------------
# Step 2 — Autocorrelation
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def autocorrelate(samples: np.ndarray, order: int) -> np.ndarray:
    """
    Compute the autocorrelation of `samples` up to lag `order`.

    Numba compiled lag-loop for maximum efficiency.
    Returns an array R of shape (order + 1,).
    """
    R = np.zeros(order + 1, dtype=samples.dtype)
    n = len(samples)
    for i in range(order + 1):
        s = 0.0
        for j in range(n - i):
            s += samples[j] * samples[j + i]
        R[i] = s
    return R


# ---------------------------------------------------------------------------
# Step 3a — Levinson-Durbin Recursion
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def levinson_durbin(R: np.ndarray, order: int) -> Tuple[np.ndarray, float]:
    """
    Solve the Yule-Walker equations using the Levinson-Durbin algorithm.

    Given autocorrelation values R[0..order], this efficiently finds the
    LPC coefficients a[1..order] that minimise the mean squared prediction
    error, without inverting a full matrix (O(order²) instead of O(order³)).

    Returns:
        coeffs  — LPC coefficients, shape (order,)
        error   — final prediction error (useful for diagnostics)
    """
    if R[0] == 0.0:
        return np.zeros(order), 0.0

    a     = np.zeros(order)
    error = R[0]

    for i in range(order):
        if i == 0:
            k = -R[1] / error
        else:
            acc = 0.0
            for j in range(i):
                acc += a[j] * R[i - j]
            k = -(R[i + 1] + acc) / error

        a_new    = a.copy()
        a_new[i] = k
        for j in range(i):
            a_new[j] = a[j] + k * a[i - 1 - j]
        a = a_new

        error *= 1.0 - k ** 2
        if error <= 0:
            break

    return a, error


# ---------------------------------------------------------------------------
# Step 3b — Burg's method  (v5.4 default)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def burg_lpc(samples: np.ndarray, order: int) -> np.ndarray:
    """
    Estimate LPC coefficients using Burg's method.

    Why Burg over Levinson-Durbin + autocorrelation?
    ─────────────────────────────────────────────────
    Levinson-Durbin fits a model to the *autocorrelation* of the frame.
    Windowing is needed to make that autocorrelation well-behaved, but
    any window attenuates signal energy at the frame edges — information
    that the predictor never sees.

    Burg's method works directly on the signal: it minimises the sum of
    forward *and* backward squared prediction errors simultaneously across
    the actual samples.  No windowing needed, no energy loss, and the
    result is always a stable (minimum-phase) filter by construction.

    In practice this means:
    · Better prediction on transients and frame boundaries
    · Lower residual energy on stationary audio too
    · More numerically stable at low frame energies (near-silence)

    Algorithm
    ─────────
    Initialise ef = eb = x (forward / backward prediction error sequences).
    For each order stage m = 0 … order-1:
        1. Reflection coeff: km = -dot(ef[m+1:], eb[m:-1]) /
                                   (0.5 * (||ef[m+1:]||² + ||eb[m:-1]||²))
        2. Update errors:
               ef[m+1:] ← ef[m+1:] + km · eb[m:-1]
               eb[m:-1] ← eb[m:-1] + km · (old ef[m+1:])
        3. Order-update AR coefficients (standard Levinson step with km).

    Args:
        samples — 1-D float64 PCM array
        order   — number of LPC coefficients

    Returns:
        coeffs — array of shape (order,), float64
    """
    x   = np.asarray(samples, dtype=np.float64)
    N   = len(x)
    ef  = x.copy()
    eb  = x.copy()
    a   = np.zeros(order, dtype=np.float64)

    for m in range(order):
        f   = ef[m + 1:]        # forward errors:   length N-1-m
        b   = eb[m: N - 1]      # backward errors:  length N-1-m

        den = 0.5 * (np.dot(f, f) + np.dot(b, b))
        if den < 1e-30:         # near-silence — stop early
            break

        km  = -np.dot(f, b) / den

        # Update error sequences (keep old f for the backward update)
        f_old       = f.copy()
        ef[m + 1:]  = f   + km * b
        eb[m: N-1]  = b   + km * f_old

        # AR coefficient order-update (Levinson recursion step)
        a_prev  = a[:m].copy()
        a[m]    = km
        for j in range(m):
            a[j] = a_prev[j] + km * a_prev[m - 1 - j]

    return a


# ---------------------------------------------------------------------------
# Step 4 — Compute LPC coefficients for a frame
# ---------------------------------------------------------------------------

def compute_lpc_coefficients(
    samples: np.ndarray,
    order:   int = DEFAULT_LPC_ORDER,
    method:  str = "autocorr",
) -> np.ndarray:
    """
    Analyse a frame of audio samples and return `order` LPC coefficients.

    Args:
        samples — 1-D array of PCM audio samples (float64)
        order   — number of LPC coefficients (default: 12)
        method  — "autocorr" (default) or "burg"

    "autocorr" (default, v5.4+):
        Applies a Blackman-Harris window before computing autocorrelation,
        then solves with Levinson-Durbin.  Blackman-Harris has −92 dB
        sidelobes vs the old Hann window's −31 dB, giving the predictor a
        much more accurate spectral picture of the frame.

    "burg":
        Burg's method — works directly on the signal without windowing,
        minimising both forward and backward prediction error simultaneously.
        Available for experimentation; will become the default in v5.5.

    Returns:
        coeffs — array of shape (order,)
    """
    samples = np.asarray(samples, dtype=np.float64)

    if method == "burg":
        return burg_lpc(samples, order)

    # Blackman-Harris windowed autocorrelation + Levinson-Durbin
    windowed  = samples * _blackman_harris_window(len(samples))
    R         = autocorrelate(windowed, order)
    coeffs, _ = levinson_durbin(R, order)
    return coeffs


# ---------------------------------------------------------------------------
# Step 4 — Encode: samples → residuals  (Numba JIT compiled)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _encode_frame_jit(samples: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Numba JIT-compiled inner loop for LPC analysis.

    @njit compiles this to native machine code on first call (cached to
    disk afterwards so subsequent runs skip recompilation).

    The loop is inherently sequential — each residual depends on the
    previous reconstructed sample — but running in native code rather than
    the Python interpreter removes essentially all overhead.

    Numba requirements (hence the separate wrapper below):
      - All arrays must be contiguous float64 / int32
      - No Python built-ins like round() — use np.round instead
        (Numba's np.round operates on scalars at C speed, unlike CPython's)
    """
    order         = len(coeffs)
    n             = len(samples)
    residuals     = np.zeros(n, dtype=np.int32)
    reconstructed = np.zeros(n, dtype=np.float64)

    for i in range(n):
        predicted = 0.0
        if i >= order:
            for j in range(order):
                predicted -= coeffs[j] * reconstructed[i - 1 - j]

        r                = np.int32(np.round(samples[i] - predicted))
        residuals[i]     = r
        reconstructed[i] = np.float64(r) + predicted

    return residuals


def encode_frame(samples: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Compute LPC residuals for one frame. Delegates to the @njit kernel.

    Args:
        samples — 1-D array of PCM samples (any numeric dtype)
        coeffs  — LPC coefficients from compute_lpc_coefficients()

    Returns:
        residuals — 1-D int32 array, same length as samples
    """
    return _encode_frame_jit(
        np.ascontiguousarray(samples, dtype=np.float64),
        np.ascontiguousarray(coeffs,  dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Step 5 — Decode: residuals → samples  (Numba JIT compiled)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _decode_frame_jit(residuals: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Numba JIT-compiled inner loop for LPC synthesis.

    Mirrors _encode_frame_jit exactly — same loop structure, same
    history buffer — guaranteeing bit-for-bit identical reconstruction.
    """
    order   = len(coeffs)
    n       = len(residuals)
    samples = np.zeros(n, dtype=np.float64)

    for i in range(n):
        predicted = 0.0
        if i >= order:
            for j in range(order):
                predicted -= coeffs[j] * samples[i - 1 - j]
        samples[i] = np.float64(residuals[i]) + predicted

    return samples


def decode_frame(residuals: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Reconstruct PCM samples from residuals and LPC coefficients.

    Args:
        residuals — 1-D int32 array from encode_frame()
        coeffs    — the same LPC coefficients used during encoding

    Returns:
        samples — 1-D float64 array of reconstructed PCM samples
    """
    return _decode_frame_jit(
        np.ascontiguousarray(residuals, dtype=np.float64),
        np.ascontiguousarray(coeffs,   dtype=np.float64),
    )



# ---------------------------------------------------------------------------
# Frame-level helpers (used by codec.py)
# ---------------------------------------------------------------------------

def split_into_frames(samples: np.ndarray, frame_size: int = FRAME_SIZE):
    """
    Split a 1-D sample array into fixed-size frames.
    The last frame is zero-padded if necessary.

    Yields (frame_index, frame_array) tuples.
    """
    n = len(samples)
    for i in range(0, n, frame_size):
        frame = samples[i: i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        yield i // frame_size, frame


# ---------------------------------------------------------------------------
# Quick smoke test (run this file directly to verify)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Integer LPC — bit-perfect lossless arithmetic
# ---------------------------------------------------------------------------
# Float LPC (v3–v4.5) uses float32 coefficients. Even though residuals are
# rounded to integers, the prediction uses floating-point which can introduce
# tiny discrepancies between encoder and decoder.
#
# Integer LPC eliminates ALL floating-point from the filter loop:
#   1. Float coefficients are negated and quantized to int16:
#      coeff_int = round(-coeff_float * 2^k)
#   2. Every prediction step uses only integer multiply + shift:
#      predicted[n] = sum(coeff_int[j] * history[n-1-j]) >> k
#                   = sum(-a[j] * history[n-1-j])  [correct sign]
#   3. Residuals are exact integers, and reconstruction is bit-perfect.
#
# v6 adds history carry-over: CONT frames seed the filter with the last
# `order` decoded samples from the previous frame rather than starting cold.
# SYNC frames always start cold (prior_history = zeros).
# ---------------------------------------------------------------------------

# Candidate LPC orders tried by select_lpc_order — all even values 2..20
LPC_ORDER_CANDIDATES = list(range(2, 21, 2))


def quantize_lpc_coefficients(
    coeffs: np.ndarray,
    precision: int = DEFAULT_LPC_PRECISION,
) -> np.ndarray:
    """
    Quantize float LPC coefficients to int16 with a given precision shift.

    Stores the NEGATED coefficients:
        coeff_int = round(-coeff_float * 2^precision)

    Why negated?
    ────────────
    The Levinson-Durbin convention gives coefficients where the correct
    prediction is:
        predicted[n] = -sum(a[j] * history[n-1-j])

    The integer JIT filter uses `+=` accumulation, so storing -a[j] makes
    the integer prediction correctly produce -sum(a[j]*history) without
    a negation step inside the hot loop.

    Backwards compatibility:
    ────────────────────────
    Files encoded before v5.4 stored the non-negated coefficients. Those
    files still decode correctly: the old wrong-sign encoder and the
    unchanged `+=` JIT decoder cancel each other out.

    With precision=13 and typical audio coefficients in [-2, 2]:
        max stored value = 2 × 8192 = 16384  (fits in int16, ±32767)

    Returns:
        int16 array of shape (order,)
    """
    scale      = float(1 << precision)
    coeffs_int = np.round(-coeffs * scale).clip(-32768, 32767).astype(np.int16)
    return coeffs_int


@njit(cache=True)
def _encode_frame_int_jit(
    samples:       np.ndarray,   # int32
    coeffs_int:    np.ndarray,   # int16
    precision:     int,
    prior_history: np.ndarray,   # int64, shape (order,) — last samples of prev frame
) -> np.ndarray:
    """
    Integer LPC analysis filter (Numba JIT compiled).

    prior_history[0] = x[last-0], prior_history[1] = x[last-1], ...
    i.e. most-recent sample at index 0, matching the j=0,1,... indexing
    of the prediction loop.

    For SYNC frames pass np.zeros(order, int64).
    For CONT frames pass the tail of the previous decoded frame.
    """
    order     = len(coeffs_int)
    n         = len(samples)
    residuals = np.zeros(n, dtype=np.int32)
    history   = np.zeros(n + order, dtype=np.int64)

    # Seed history with prior context (index 0 = most recent)
    for j in range(order):
        history[order - 1 - j] = prior_history[j]

    for i in range(n):
        hi        = i + order   # index into padded history
        predicted = np.int64(0)
        for j in range(order):
            predicted += np.int64(coeffs_int[j]) * history[hi - 1 - j]
        predicted >>= precision

        r              = np.int64(samples[i]) - predicted
        residuals[i]   = np.int32(r)
        history[hi]    = np.int64(samples[i])

    return residuals


@njit(cache=True)
def _decode_frame_int_jit(
    residuals:     np.ndarray,   # int32
    coeffs_int:    np.ndarray,   # int16
    precision:     int,
    prior_history: np.ndarray,   # int64, shape (order,)
) -> np.ndarray:
    """
    Integer LPC synthesis filter (Numba JIT compiled).

    Exact mirror of _encode_frame_int_jit — bit-perfect reconstruction
    guaranteed because both sides use identical integer arithmetic.
    """
    order   = len(coeffs_int)
    n       = len(residuals)
    history = np.zeros(n + order, dtype=np.int64)

    for j in range(order):
        history[order - 1 - j] = prior_history[j]

    for i in range(n):
        hi        = i + order
        predicted = np.int64(0)
        for j in range(order):
            predicted += np.int64(coeffs_int[j]) * history[hi - 1 - j]
        predicted >>= precision

        history[hi] = np.int64(residuals[i]) + predicted

    return history[order:].astype(np.float64)


def encode_frame_int(
    samples:       np.ndarray,
    coeffs_int:    np.ndarray,
    precision:     int        = DEFAULT_LPC_PRECISION,
    prior_history: np.ndarray = None,
) -> np.ndarray:
    """
    Compute integer LPC residuals.

    Args:
        samples       — float64 PCM samples in integer range (e.g. ±32767)
        coeffs_int    — int16 quantized coefficients from quantize_lpc_coefficients()
        precision     — the same precision used to quantize the coefficients
        prior_history — int64 array of shape (≥order,) containing the last
                        decoded samples of the previous frame, most-recent first.
                        None (default) → zeros (SYNC frame behaviour).
                        If shorter than order, zero-padded automatically.

    Returns:
        int32 residuals
    """
    order       = len(coeffs_int)
    samples_i32 = np.round(samples).astype(np.int32)
    hist        = _prepare_history(prior_history, order)
    return _encode_frame_int_jit(
        np.ascontiguousarray(samples_i32, dtype=np.int32),
        np.ascontiguousarray(coeffs_int,  dtype=np.int16),
        precision,
        hist,
    )


def decode_frame_int(
    residuals:     np.ndarray,
    coeffs_int:    np.ndarray,
    precision:     int        = DEFAULT_LPC_PRECISION,
    prior_history: np.ndarray = None,
) -> np.ndarray:
    """
    Reconstruct PCM samples from integer residuals and int16 coefficients.

    Args:
        prior_history — int64 array of shape (≥order,), most-recent first.
                        None (default) → zeros (SYNC frame behaviour).
                        If shorter than order, zero-padded automatically.

    Returns:
        float64 array of reconstructed PCM samples
    """
    order = len(coeffs_int)
    hist  = _prepare_history(prior_history, order)
    return _decode_frame_int_jit(
        np.ascontiguousarray(residuals,  dtype=np.int32),
        np.ascontiguousarray(coeffs_int, dtype=np.int16),
        precision,
        hist,
    )


def _prepare_history(prior_history, order: int) -> np.ndarray:
    """
    Return a contiguous int64 history array of exactly `order` elements.

    - None → zeros (SYNC frame)
    - Shorter than order → zero-padded at the tail (oldest positions)
    - Longer than order → truncated to first `order` elements (most recent)

    This handles adaptive-order encoding where consecutive frames may have
    different orders — the JIT function always receives an array of exactly
    `order` elements without out-of-bounds access.
    """
    if prior_history is None:
        return np.zeros(order, dtype=np.int64)
    h = np.asarray(prior_history, dtype=np.int64)
    if len(h) == order:
        return np.ascontiguousarray(h)
    if len(h) < order:
        return np.ascontiguousarray(
            np.concatenate([h, np.zeros(order - len(h), dtype=np.int64)]))
    return np.ascontiguousarray(h[:order])


def make_prior_history(decoded_samples: np.ndarray, order: int) -> np.ndarray:
    """
    Extract the tail of a decoded frame as prior history for the next frame.

    Returns an int64 array of shape (order,) where index 0 is the most
    recently decoded sample — matching the j=0 convention of the JIT filters.

    Call this after decoding a SYNC or CONT frame; pass the result as
    `prior_history` when encoding/decoding the subsequent CONT frame.
    """
    tail = np.round(decoded_samples[-order:]).astype(np.int64)
    return tail[::-1].copy()   # reverse so index 0 = most recent


def make_sync_history(
    prev_samples:  np.ndarray,
    coeffs_int:    np.ndarray,
    precision:     int = DEFAULT_LPC_PRECISION,
) -> np.ndarray:
    """
    Compute warm-up history for a SYNC frame using the previous frame's samples.

    A SYNC frame normally starts cold (history = zeros), which means the
    first `order` residuals are larger than necessary — the filter has no
    context.  This function seeds the history by running the previous
    frame's *integer samples* through the SYNC frame's own filter as a
    forward pass, producing the same history the decoder will have after
    decoding one full frame of zeros.

    Because both encoder and decoder use the same warm-up arithmetic the
    result is bit-perfect — no format change or decoder update needed.

    Why the SYNC frame's own coefficients?
    ───────────────────────────────────────
    The SYNC frame's coefficients model the signal's current spectral
    shape.  Using them to interpret the previous samples gives the filter
    a head start that matches its own prediction model.

    Args:
        prev_samples — float64 PCM samples from the frame immediately
                       before the SYNC frame (same length as frame_size)
        coeffs_int   — int16 quantized coefficients for the SYNC frame
        precision    — precision used to quantize those coefficients

    Returns:
        int64 prior_history array of shape (order,), ready to pass to
        encode_frame_int / decode_frame_int as prior_history.
    """
    order      = len(coeffs_int)
    # Decode the previous samples as if they were encoded with these
    # coefficients and cold-started — we only care about the resulting
    # reconstructed integer history, not the residuals.
    samples_i32 = np.round(prev_samples).astype(np.int32)
    hist        = np.zeros(order, dtype=np.int64)
    reconstructed = _decode_frame_int_jit(
        # Encode with SYNC coeffs to get matching residuals
        _encode_frame_int_jit(
            np.ascontiguousarray(samples_i32, dtype=np.int32),
            np.ascontiguousarray(coeffs_int,  dtype=np.int16),
            precision,
            hist,
        ),
        np.ascontiguousarray(coeffs_int, dtype=np.int16),
        precision,
        hist,
    )
    return make_prior_history(reconstructed, order)


# ---------------------------------------------------------------------------
# Pre-emphasis / de-emphasis  (v6.0.1.0)
# ---------------------------------------------------------------------------
# Audio signals have most of their energy at low frequencies — bass, vocals,
# fundamentals.  A flat spectrum is easiest for LPC to model, but a heavily
# bass-weighted spectrum pushes energy into low-order coefficients and leaves
# the predictor chasing the dominant low-frequency trend rather than the
# fine detail.
#
# A first-order pre-emphasis filter:
#     y[n] = x[n] − α · x[n−1]      (high-pass)
# boosts high frequencies and flattens the spectral envelope before LPC
# analysis, improving the predictor's resolution across the spectrum.
#
# The matching de-emphasis filter reverses this on the residuals:
#     x[n] = y[n] + α · x[n−1]      (leaky integrator)
# so the residuals stored in the file are in the original (un-emphasised)
# domain — the decoder needs no change at all.
#
# Typical α for music: 0.97.  Range 0.9–0.99.
# α = 0.0 disables pre-emphasis entirely (identity filter).
# ---------------------------------------------------------------------------

DEFAULT_PREEMPH = 0.97   # standard value used by many speech/audio codecs


def pre_emphasis(samples: np.ndarray, alpha: float = DEFAULT_PREEMPH,
                 prev_sample: float = 0.0) -> np.ndarray:
    """
    Apply a first-order pre-emphasis filter: y[n] = x[n] − α·x[n−1].

    Args:
        samples     — 1-D float64 PCM array (integer scale, e.g. ±32767)
        alpha       — emphasis coefficient (default 0.97)
        prev_sample — last sample of the previous frame, for continuity
                      across frame boundaries. Pass 0.0 for the first frame.

    Returns:
        float64 array, same shape as samples
    """
    if alpha == 0.0:
        return samples.copy()
    out    = np.empty_like(samples, dtype=np.float64)
    out[0] = samples[0] - alpha * prev_sample
    out[1:] = samples[1:] - alpha * samples[:-1]
    return out


def de_emphasis(residuals: np.ndarray, alpha: float = DEFAULT_PREEMPH,
                prev_reconstructed: float = 0.0) -> np.ndarray:
    """
    Reverse pre-emphasis on the integer residuals: x[n] = y[n] + α·x[n−1].

    This is applied *after* computing LPC residuals but *before* entropy
    coding, restoring the residuals to the original un-emphasised domain.
    The decoder is unchanged — it receives plain residuals as always.

    The de-emphasis is a causal IIR filter so it runs sample-by-sample:

        r_deemph[0] = r_emph[0] + α · prev_reconstructed
        r_deemph[n] = r_emph[n] + α · r_deemph[n−1]

    Because this operates on float64 before the final rounding to int32,
    the round-trip is still bit-perfect.

    Args:
        residuals           — float64 residual array (from LPC analysis on
                              pre-emphasised samples)
        alpha               — same coefficient used during pre-emphasis
        prev_reconstructed  — last de-emphasised residual of the previous
                              frame (for continuity). Pass 0.0 for first frame.

    Returns:
        float64 array of de-emphasised residuals (ready for rounding to int32)
    """
    if alpha == 0.0:
        return residuals.copy()
    out    = np.empty_like(residuals, dtype=np.float64)
    out[0] = residuals[0] + alpha * prev_reconstructed
    for i in range(1, len(residuals)):
        out[i] = residuals[i] + alpha * out[i - 1]
    return out

def _rice_bit_cost(residuals: np.ndarray) -> int:
    """
    Estimate the Rice-coded bit cost of an int32 residual array.

    Uses the same zigzag + optimal-k calculation as rice.choose_k / rice.encode,
    but returns only the bit count rather than producing encoded bytes.
    Inlined here to avoid a circular import (rice.py ↔ lpc.py) and to keep
    the hot path fast — no Python list conversion, pure numpy.

    Cost model:
        k     = round(log2(mean_unsigned / log(2)))  — optimal k estimator
        bits  = sum(unsigned >> k) + n*(k+1)          — unary + remainder bits
    """
    # Zigzag-encode signed → unsigned (same as rice._zigzag_arr)
    u = np.where(residuals >= 0,
                 residuals.astype(np.int64) * 2,
                 -residuals.astype(np.int64) * 2 - 1).astype(np.int64)

    mean_val = float(u.mean())
    if mean_val <= 0:
        return len(residuals)   # all zeros → 1 bit each (k=0)

    import math
    k   = max(0, min(30, round(math.log2(mean_val / math.log(2) + 1e-10))))
    # ±1 search for safety (matches choose_k behaviour)
    best = int(np.iinfo(np.int64).max)
    for ki in range(max(0, k-1), min(31, k+2)):
        cost = int((u >> ki).sum()) + len(u) * (ki + 1)
        if cost < best:
            best = cost
    return best


def select_lpc_order(
    samples:    np.ndarray,
    candidates: list = None,
    precision:  int  = DEFAULT_LPC_PRECISION,
) -> int:
    """
    Try candidate LPC orders and return the one with the lowest *total bit cost*.

    Cost model (v6.0.1.0)
    ─────────────────────
    total_bits = rice_bits(residuals) + order × 16

    The second term accounts for the int16 coefficients that must be stored
    per frame. Without it, lower orders always look cheaper because they
    produce larger residuals (more bits to Rice-encode) but fewer coefficients.

    Minimum-order guarantee
    ───────────────────────
    If the cheapest candidate is below DEFAULT_LPC_ORDER (12), we also
    evaluate order 12 and keep it as a fallback if it costs no more than
    10% above the candidate winner. This prevents adaptive encoding from
    regressing on tonal signals where very low orders appear optimal in
    isolation but cause larger Rice payloads across the file.

    Early-exit optimisation
    ───────────────────────
    Candidates are evaluated in ascending order. Once cost increases on two
    consecutive candidates we stop — the cost curve is convex for typical
    audio so further increases in order are very unlikely to help.

    Args:
        samples    — float64 PCM samples in integer range
        candidates — list of orders to try in ascending order
                     (default: LPC_ORDER_CANDIDATES, even values 2–20)
        precision  — integer LPC precision shift

    Returns:
        best LPC order (int)
    """
    if candidates is None:
        candidates = LPC_ORDER_CANDIDATES

    best_order   = candidates[0]
    best_cost    = float('inf')
    worse_streak = 0
    floor_order  = DEFAULT_LPC_ORDER   # minimum sensible order for most signals

    for order in candidates:
        coeffs     = compute_lpc_coefficients(samples, order)
        coeffs_int = quantize_lpc_coefficients(coeffs, precision)
        residuals  = encode_frame_int(samples, coeffs_int, precision)
        rice_bits  = _rice_bit_cost(residuals)
        total_cost = rice_bits + order * 16

        if total_cost < best_cost:
            best_cost    = total_cost
            best_order   = order
            worse_streak = 0
        else:
            worse_streak += 1
            if worse_streak >= 2:
                break

    # If best order is below the floor, compare against the floor order
    # and use the floor if it's within 10% — avoids order-2 wins on pure
    # tones that are costly across longer files due to larger Rice payloads
    if best_order < floor_order and floor_order in candidates:
        c_floor   = compute_lpc_coefficients(samples, floor_order)
        ci_floor  = quantize_lpc_coefficients(c_floor, precision)
        res_floor = encode_frame_int(samples, ci_floor, precision)
        cost_floor = _rice_bit_cost(res_floor) + floor_order * 16
        if cost_floor <= best_cost * 1.10:
            best_order = floor_order

    return best_order


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== HFPAC lpc.py smoke test ===\n")

    sr = 44100
    t  = np.linspace(0, FRAME_SIZE / sr, FRAME_SIZE, endpoint=False)
    test_signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.float64)

    # Float LPC roundtrip
    coeffs = compute_lpc_coefficients(test_signal, order=DEFAULT_LPC_ORDER)
    residuals = encode_frame(test_signal, coeffs)
    reconstructed = decode_frame(residuals, coeffs)
    max_error = np.max(np.abs(test_signal - reconstructed))
    print(f"Float LPC max error:    {max_error:.4f}  (< 1.0 expected)")

    # Integer LPC roundtrip — SYNC (no history)
    ci  = quantize_lpc_coefficients(coeffs)
    res = encode_frame_int(test_signal, ci)
    rec = decode_frame_int(res, ci)
    err = np.max(np.abs(np.round(test_signal) - rec))
    print(f"Int LPC SYNC error:     {err:.4f}  (0.0 = bit-perfect)")
    assert err == 0.0

    # Integer LPC roundtrip — CONT (with history)
    hist = make_prior_history(rec, DEFAULT_LPC_ORDER)
    t2   = np.linspace(FRAME_SIZE/sr, 2*FRAME_SIZE/sr, FRAME_SIZE, endpoint=False)
    sig2 = (np.sin(2*np.pi*440*t2)*32767).astype(np.float64)
    c2   = compute_lpc_coefficients(sig2)
    ci2  = quantize_lpc_coefficients(c2)
    res2 = encode_frame_int(sig2, ci2, prior_history=hist)
    rec2 = decode_frame_int(res2, ci2, prior_history=hist)
    err2 = np.max(np.abs(np.round(sig2) - rec2))
    print(f"Int LPC CONT error:     {err2:.4f}  (0.0 = bit-perfect)")
    assert err2 == 0.0

    # Adaptive order selection
    best = select_lpc_order(test_signal)
    print(f"Adaptive order chosen:  {best}  (from {LPC_ORDER_CANDIDATES})")

    print(f"\n✅  All smoke test assertions passed.")