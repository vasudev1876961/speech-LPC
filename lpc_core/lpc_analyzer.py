"""
lpc_core/lpc_analyzer.py
------------------------
Framing, windowing, autocorrelation computation, and Levinson-Durbin algorithm.
Estimates the all-pole vocal-tract filter model from speech.
"""

import numpy as np

def frame_signal(signal, frame_len, hop_len):
    """
    Splits 1D signal into overlapping frames.
    
    Parameters
    ----------
    signal : np.ndarray
        1D speech signal array.
    frame_len : int
        Number of samples per frame (e.g., 400 for 25ms at 16kHz).
    hop_len : int
        Hop step between consecutive frames (e.g., 160 for 10ms at 16kHz).
        
    Returns
    -------
    frames : np.ndarray
        2D array of shape (num_frames, frame_len).
    indices : np.ndarray
        Start sample index for each frame.
    """
    total_samples = len(signal)
    if total_samples < frame_len:
        # Pad with zeros if shorter than one frame
        pad_width = frame_len - total_samples
        signal = np.pad(signal, (0, pad_width), mode='constant')
        total_samples = len(signal)
        
    num_frames = 1 + int(np.floor((total_samples - frame_len) / hop_len))
    indices = np.arange(num_frames) * hop_len
    
    # Vectorized sliding window extraction
    shape = (num_frames, frame_len)
    strides = (signal.strides[0] * hop_len, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    
    return frames.copy(), indices

def compute_autocorrelation(frame, order):
    """
    Computes biased autocorrelation coefficients R[0], R[1], ..., R[order]
    for a windowed speech frame.
    
    R[k] = sum_{n=0}^{N - 1 - k} x[n] * x[n + k]
    """
    N = len(frame)
    R = np.zeros(order + 1, dtype=np.float64)
    for k in range(order + 1):
        R[k] = np.dot(frame[:N - k], frame[k:])
    return R

def levinson_durbin(R, order):
    """
    Solves the Yule-Walker equations using Levinson-Durbin recursion.
    Toeplitz system: R * a = [E, 0, 0, ..., 0]^T
    
    Parameters
    ----------
    R : np.ndarray
        Autocorrelation sequence of length order + 1: [R[0], R[1], ..., R[p]].
    order : int
        LPC order (p).
        
    Returns
    -------
    a : np.ndarray
        LPC filter coefficients vector: [1.0, a_1, a_2, ..., a_p] (length p + 1).
    k : np.ndarray
        Reflection coefficients (PARCOR) of length p.
    E : float
        Final minimum prediction error energy.
    """
    # Guard against zero or near-silent frames
    if R[0] < 1e-12:
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        k = np.zeros(order, dtype=np.float64)
        return a, k, 0.0

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    k = np.zeros(order, dtype=np.float64)
    
    E = float(R[0])
    
    # Iterative step-up recursion
    for i in range(1, order + 1):
        # Calculate reflection coefficient k[i-1]
        sum_val = 0.0
        for j in range(1, i):
            sum_val += a[j] * R[i - j]
            
        ki = -(R[i] + sum_val) / (E + 1e-15)
        k[i - 1] = ki
        
        # Update filter coefficients
        a_prev = a.copy()
        a[i] = ki
        for j in range(1, i):
            a[j] = a_prev[j] + ki * a_prev[i - j]
            
        # Update prediction error energy
        E = E * (1.0 - ki * ki)
        
        # Stability safeguard: if filter becomes unstable (|ki| >= 1)
        if E <= 0 or abs(ki) >= 1.0:
            E = max(E, 1e-10)
            break
            
    return a, k, E

def extract_lpc_coefficients(signal, fs=16000, order=16, frame_ms=25.0, hop_ms=10.0, window_type="hamming"):
    """
    Performs frame-by-frame LPC analysis on the speech signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input speech signal.
    fs : int
        Sampling frequency in Hz (default 16000).
    order : int
        LPC model order (default 16).
        Rule of thumb: p = fs/1000 + 4 => 16 + 4 = 16 to 20 for adult speech at 16kHz.
    frame_ms : float
        Frame length in milliseconds (default 25.0ms -> 400 samples).
    hop_ms : float
        Frame hop step in milliseconds (default 10.0ms -> 160 samples).
    window_type : str
        Window function ("hamming", "hanning", or "rectangular").
        
    Returns
    -------
    dict with:
        'lpc_coeffs': np.ndarray, shape (num_frames, order + 1)
        'reflection_coeffs': np.ndarray, shape (num_frames, order)
        'prediction_energies': np.ndarray, shape (num_frames,)
        'frame_times': np.ndarray, shape (num_frames,) in seconds
        'indices': np.ndarray, start sample indices
        'frame_len': int
        'hop_len': int
        'fs': int
        'order': int
    """
    frame_len = int(round(frame_ms * 1e-3 * fs))
    hop_len = int(round(hop_ms * 1e-3 * fs))
    
    frames, indices = frame_signal(signal, frame_len, hop_len)
    num_frames = frames.shape[0]
    
    # Create window
    if window_type == "hamming":
        window = np.hamming(frame_len)
    elif window_type == "hanning":
        window = np.hanning(frame_len)
    else:
        window = np.ones(frame_len)
        
    lpc_coeffs = np.zeros((num_frames, order + 1), dtype=np.float64)
    reflection_coeffs = np.zeros((num_frames, order), dtype=np.float64)
    pred_energies = np.zeros(num_frames, dtype=np.float64)
    
    for i in range(num_frames):
        # Window the frame to taper edges and prevent spectral leakage
        w_frame = frames[i] * window
        # Autocorrelation
        R = compute_autocorrelation(w_frame, order)
        # Levinson-Durbin
        a, k, E = levinson_durbin(R, order)
        lpc_coeffs[i] = a
        reflection_coeffs[i] = k
        pred_energies[i] = E
        
    frame_times = (indices + frame_len / 2.0) / fs
    
    return {
        "lpc_coeffs": lpc_coeffs,
        "reflection_coeffs": reflection_coeffs,
        "prediction_energies": pred_energies,
        "frame_times": frame_times,
        "indices": indices,
        "frame_len": frame_len,
        "hop_len": hop_len,
        "fs": fs,
        "order": order
    }
