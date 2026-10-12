"""
lpc_core/preprocessor.py
------------------------
Audio loading, conditioning, normalization, and pre-emphasis filtering.
"""

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly

def load_and_preprocess_audio(filepath, target_fs=16000, apply_preemph=True, preemph_coeff=0.97):
    """
    Loads a WAV file, converts to mono, resamples to target_fs, removes DC bias,
    normalizes amplitude, and optionally applies pre-emphasis.
    
    Parameters
    ----------
    filepath : str
        Path to input .wav file.
    target_fs : int
        Desired sampling frequency in Hz (default 16000).
    apply_preemph : bool
        Whether to apply pre-emphasis filter y[n] = x[n] - a * x[n-1].
    preemph_coeff : float
        Pre-emphasis coefficient, typically 0.95 - 0.98 (default 0.97).
        
    Returns
    -------
    processed_signal : np.ndarray
        1D float array normalized between -1.0 and 1.0.
    raw_signal : np.ndarray
        1D float array before pre-emphasis (normalized).
    fs : int
        Final sampling rate (target_fs).
    """
    fs, data = wavfile.read(filepath)
    
    # 1. Convert to floating point
    if data.dtype == np.int16:
        signal = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        signal = (data.astype(np.float64) - 128.0) / 128.0
    else:
        signal = data.astype(np.float64)
        
    # 2. Convert Stereo to Mono
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
        
    # 3. Resample if necessary to target_fs
    if fs != target_fs:
        from math import gcd
        g = gcd(int(target_fs), int(fs))
        up = int(target_fs // g)
        down = int(fs // g)
        signal = resample_poly(signal, up, down)
        fs = target_fs
        
    # 4. Remove DC offset
    signal = signal - np.mean(signal)
    
    # 5. Amplitude normalization (peak normalize to [-0.95, 0.95] to prevent clipping)
    peak = np.max(np.abs(signal))
    if peak > 1e-6:
        signal = signal / peak * 0.95
        
    raw_signal = np.copy(signal)
    
    # 6. Pre-emphasis: flattens the glottal roll-off (-6dB/octave) to boost high-frequency formants
    if apply_preemph:
        processed_signal = pre_emphasis(signal, alpha=preemph_coeff)
    else:
        processed_signal = signal
        
    return processed_signal, raw_signal, fs

def pre_emphasis(signal, alpha=0.97):
    """
    High-pass pre-emphasis filter: y[n] = x[n] - alpha * x[n-1].
    Justification: Speech signals naturally have lower energy at higher frequencies
    due to glottal source spectral tilt (~ -6 dB/octave). Pre-emphasis boosts
    high-frequency formant resonances (F2, F3) for accurate LPC model fitting.
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def de_emphasis(signal, alpha=0.97):
    """
    Inverses pre-emphasis filter: x[n] = y[n] + alpha * x[n-1].
    """
    from scipy.signal import lfilter
    return lfilter([1.0], [1.0, -alpha], signal)

def save_wav_file(filepath, signal, fs=16000):
    """
    Saves a floating-point numpy array as a standard 16-bit PCM WAV file.
    """
    # Safe clipping
    clipped = np.clip(signal, -1.0, 1.0)
    int16_audio = np.int16(clipped * 32767)
    wavfile.write(filepath, fs, int16_audio)
