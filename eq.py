import numpy as np
from numba import njit
import math

EQ_center_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

def make_peaking_eq(f0, fs, Q, dbGain):
    if dbGain == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
    A = math.pow(10, dbGain / 40.0)
    w0 = 2 * math.pi * f0 / fs
    if w0 >= math.pi: # Nyquist limit guard
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
    alpha = math.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / A

    return np.array([b0/a0, b1/a0, b2/a0, a1/a0, a2/a0], dtype=np.float64)

def get_eq_coeffs(fs, gains):
    coeffs = np.zeros((10, 5), dtype=np.float64)
    Q = 1.414
    for i, f0 in enumerate(EQ_center_freqs):
        coeffs[i] = make_peaking_eq(f0, fs, Q, gains[i])
    return coeffs

@njit
def apply_biquad_cascade(samples, coeffs, state):
    out = np.empty_like(samples)
    N_bands = coeffs.shape[0]
    for i in range(len(samples)):
        x = samples[i]
        for b in range(N_bands):
            b0 = coeffs[b, 0]
            b1 = coeffs[b, 1]
            b2 = coeffs[b, 2]
            a1 = coeffs[b, 3]
            a2 = coeffs[b, 4]
            
            w0 = x - a1 * state[b, 0] - a2 * state[b, 1]
            y = b0 * w0 + b1 * state[b, 0] + b2 * state[b, 1]
            
            state[b, 1] = state[b, 0]
            state[b, 0] = w0
            x = y
        out[i] = x
    return out

@njit
def process_stereo_eq(block, coeffs, state):
    out = np.empty_like(block)
    channels = block.shape[1]
    for ch in range(channels):
        out[:, ch] = apply_biquad_cascade(block[:, ch], coeffs, state[ch])
    return out
