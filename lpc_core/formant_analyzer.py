"""
lpc_core/formant_analyzer.py
----------------------------
Extracts vocal-tract formant frequencies (F1, F2, F3, ...) and computes
the LPC spectral envelope H(z) = 1 / A(z).
"""

import numpy as np
from scipy.signal import freqz

def extract_formants_from_lpc(a, fs=16000, max_bandwidth=450.0):
    """
    Extracts formant frequencies and bandwidths from LPC filter coefficients
    by finding the roots of the polynomial A(z).
    
    A(z) = 1 + a_1*z^{-1} + ... + a_p*z^{-p} = 0
    
    Parameters
    ----------
    a : np.ndarray
        LPC coefficients [1.0, a_1, ..., a_p].
    fs : int
        Sampling frequency in Hz.
    max_bandwidth : float
        Maximum allowable bandwidth for a valid vocal tract formant (default 450 Hz).
        
    Returns
    -------
    formants : list of float
        Sorted list of formant center frequencies (F1, F2, F3, ...).
    bandwidths : list of float
        Corresponding formant bandwidths in Hz.
    """
    # Find roots of the LPC polynomial
    roots = np.roots(a)
    
    # Keep roots in the upper half of the z-plane (positive frequencies)
    # and strictly inside the unit circle (stable resonances)
    roots = roots[np.imag(roots) >= 0]
    
    # Calculate frequencies and bandwidths
    angles = np.angle(roots)
    freqs = angles * (fs / (2.0 * np.pi))
    bandwidths = - (fs / np.pi) * np.log(np.maximum(np.abs(roots), 1e-12))
    
    # Filter candidates:
    # 1. Frequency strictly between 90 Hz and fs/2 - 150 Hz
    # 2. Bandwidth below threshold
    valid_mask = (freqs > 90.0) & (freqs < (fs / 2.0 - 150.0)) & (bandwidths < max_bandwidth)
    
    valid_freqs = freqs[valid_mask]
    valid_bws = bandwidths[valid_mask]
    
    # Sort ascending
    sort_idx = np.argsort(valid_freqs)
    sorted_freqs = valid_freqs[sort_idx].tolist()
    sorted_bws = valid_bws[sort_idx].tolist()
    
    return sorted_freqs, sorted_bws

def compute_lpc_spectrum(a, fs=16000, n_fft=512):
    """
    Computes the frequency response of the all-pole filter H(z) = 1 / A(z).
    This represents the smoothed vocal tract spectral envelope.
    
    Parameters
    ----------
    a : np.ndarray
        LPC coefficients [1.0, a_1, ..., a_p].
    fs : int
        Sampling frequency in Hz.
    n_fft : int
        Number of FFT points.
        
    Returns
    -------
    freqs : np.ndarray
        Frequency axis from 0 to fs/2 in Hz.
    mag_db : np.ndarray
        Magnitude in decibels (dB): 20 * log10(|H(f)|).
    """
    # In scipy freqz: H(w) = B(w) / A(w). Here B(w) = 1.0, A(w) = a.
    w, h = freqz([1.0], a, worN=n_fft, fs=fs)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-10))
    return w, mag_db

def track_formants(lpc_results, max_formants=4):
    """
    Tracks formant trajectories over all frames in the utterance.
    
    Returns
    -------
    formant_tracks : dict with keys 'F1', 'F2', 'F3', 'F4', each an array of length num_frames
    times : np.ndarray of frame center timestamps
    """
    lpc_coeffs = lpc_results['lpc_coeffs']
    fs = lpc_results['fs']
    times = lpc_results['frame_times']
    num_frames = len(lpc_coeffs)
    
    tracks = {f"F{i+1}": np.full(num_frames, np.nan) for i in range(max_formants)}
    
    for i in range(num_frames):
        a = lpc_coeffs[i]
        freqs, _ = extract_formants_from_lpc(a, fs=fs)
        for j in range(min(len(freqs), max_formants)):
            tracks[f"F{j+1}"][i] = freqs[j]
            
    return tracks, times

def extract_formants(a, fs=16000):
    """Convenience alias for extract_formants_from_lpc."""
    return extract_formants_from_lpc(a, fs)
