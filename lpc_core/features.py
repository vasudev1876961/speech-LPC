"""
lpc_core/features.py
--------------------
Feature extraction from LPC coefficients and LPC residual:
- Prediction Gain
- Pitch / F0 estimation from residual autocorrelation
- Residual energy contour
- Spectral flatness
- Reflection coefficients (PARCOR) statistics
"""

import numpy as np

def estimate_pitch_from_residual(residual, fs=16000, fmin=60.0, fmax=400.0, frame_ms=40.0, hop_ms=10.0):
    """
    Estimates fundamental frequency (F0 / pitch) contour using autocorrelation
    of the LPC residual signal.
    
    Why from the residual?
    Inverse filtering removes formant resonances, leaving behind predominantly
    the glottal excitation pulse train. This makes pitch period peaks in the
    autocorrelation function significantly sharper and less prone to formant pitch doubling.
    """
    frame_len = int(round(frame_ms * 1e-3 * fs))
    hop_len = int(round(hop_ms * 1e-3 * fs))
    
    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)
    
    num_frames = 1 + int(np.floor((len(residual) - frame_len) / hop_len))
    f0_contour = np.full(num_frames, np.nan)
    times = (np.arange(num_frames) * hop_len + frame_len / 2.0) / fs
    
    for i in range(num_frames):
        start = i * hop_len
        frame = residual[start : start + frame_len]
        
        # Energy check (unvoiced / silence detection)
        if np.std(frame) < 1e-4:
            continue
            
        # Autocorrelation of residual frame
        r = np.correlate(frame, frame, mode='full')
        r = r[len(frame) - 1 :] # Non-negative lags
        
        if len(r) > max_lag:
            search_region = r[min_lag : max_lag + 1]
            peak_lag_rel = np.argmax(search_region)
            peak_lag = min_lag + peak_lag_rel
            
            # Voicing threshold: normalized autocorrelation peak > 0.25
            norm_peak = search_region[peak_lag_rel] / (r[0] + 1e-12)
            if norm_peak > 0.22:
                f0 = fs / peak_lag
                f0_contour[i] = f0
                
    return f0_contour, times

def compute_prediction_gain(raw_signal, residual, frame_len=400, hop_len=160):
    """
    Computes Prediction Gain in dB across frames:
    G = 10 * log10( Energy(raw) / Energy(residual) )
    Higher gain indicates strong vocal-tract formant resonance (voiced vowels).
    """
    num_frames = 1 + int(np.floor((len(raw_signal) - frame_len) / hop_len))
    gain_db = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_len
        s_raw = raw_signal[start : start + frame_len]
        s_res = residual[start : start + frame_len]
        
        e_raw = np.mean(s_raw ** 2) + 1e-12
        e_res = np.mean(s_res ** 2) + 1e-12
        
        gain_db[i] = 10.0 * np.log10(e_raw / e_res)
        
    return gain_db

def compute_spectral_flatness(signal, frame_len=400, hop_len=160):
    """
    Computes Wiener spectral flatness: geometric mean / arithmetic mean of power spectrum.
    Values close to 1.0 indicate white-like excitation (residual);
    values close to 0.0 indicate peaky resonant speech.
    """
    num_frames = 1 + int(np.floor((len(signal) - frame_len) / hop_len))
    flatness = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_len
        frame = signal[start : start + frame_len] * np.hanning(frame_len)
        ps = np.abs(np.fft.rfft(frame)) ** 2 + 1e-15
        
        geo_mean = np.exp(np.mean(np.log(ps)))
        arith_mean = np.mean(ps)
        flatness[i] = geo_mean / (arith_mean + 1e-15)
        
    return flatness

def extract_lpc_features(raw_signal, residual, lpc_results):
    """
    Aggregates all key features into a comprehensive dictionary for downstream analytical modules.
    """
    fs = lpc_results['fs']
    frame_len = lpc_results['frame_len']
    hop_len = lpc_results['hop_len']
    
    f0, f0_times = estimate_pitch_from_residual(residual, fs=fs)
    gain_db = compute_prediction_gain(raw_signal, residual, frame_len, hop_len)
    res_flatness = compute_spectral_flatness(residual, frame_len, hop_len)
    raw_flatness = compute_spectral_flatness(raw_signal, frame_len, hop_len)
    
    # RMS Energy of residual
    num_frames = len(gain_db)
    res_rms = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_len
        res_rms[i] = np.sqrt(np.mean(residual[start : start + frame_len] ** 2))
        
    frame_times = (np.arange(num_frames) * hop_len + frame_len / 2.0) / fs
    
    return {
        "f0": f0,
        "f0_times": f0_times,
        "frame_times": frame_times,
        "prediction_gain_db": gain_db,
        "mean_prediction_gain_db": float(np.mean(gain_db[gain_db > 0])) if np.any(gain_db > 0) else 0.0,
        "residual_rms": res_rms,
        "mean_residual_rms": float(np.mean(res_rms)),
        "residual_spectral_flatness": float(np.mean(res_flatness)),
        "speech_spectral_flatness": float(np.mean(raw_flatness)),
    }
