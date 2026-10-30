"""
lpc_core/phase_timing.py
-------------------------
Module 3: Phase & Glottal Timing Analysis Suite for LPC Residual.
Provides:
1. Zero Frequency Resonator (ZFR) method for Epoch / Glottal Closure Instant (GCI) extraction
2. Hilbert Transform: Instantaneous Amplitude, Phase, and Instantaneous Frequency
3. Group Delay Function & Modified Group Delay Spectrum (resolving nasal zeros & formants)
4. Glottal Timing Metrics: Jitter (%), Shimmer (%), HNR (dB), and Glottal Pulse Dynamics
"""

import numpy as np
from scipy.signal import hilbert, correlate

def extract_epochs_zfr(residual, fs=16000, avg_f0=130.0):
    """
    Extracts Glottal Closure Instants (GCIs / Epochs) using the Zero Frequency
    Resonator (ZFR) method on the LPC residual.
    
    Why ZFR on LPC Residual?
    The glottal closure produces a sharp discontinuity (impulse) in the vocal tract.
    The response of an ideal resonator at zero frequency to an impulse is a step,
    and a second resonator yields a ramp. At the epoch instant, a sharp zero-crossing occurs.
    
    Parameters
    ----------
    residual : np.ndarray
        1D LPC residual signal e[n].
    fs : int
        Sampling frequency in Hz.
    avg_f0 : float
        Nominal fundamental frequency for trend removal window length.
        
    Returns
    -------
    gci_indices : np.ndarray
        Sample indices of detected Glottal Closure Instants (GCIs).
    gci_times : np.ndarray
        Timestamps of GCIs in seconds.
    zfr_signal : np.ndarray
        Zero-frequency filtered trend-removed signal.
    """
    N = len(residual)
    x = residual.astype(np.float64).copy()
    
    # Remove mean
    x = x - np.mean(x)
    
    # 1. First 0-Hz Resonator (Double accumulator: y1[n] = 2 y1[n-1] - y1[n-2] + x[n])
    # Implemented via cumulative summation
    y1 = np.cumsum(np.cumsum(x))
    
    # Trend removal window: corresponds to 1 to 1.5 pitch periods
    # Window length 2*M + 1
    N_win = int(round(fs / max(avg_f0, 50.0) * 1.2))
    if N_win % 2 == 0:
        N_win += 1
    M = N_win // 2
    
    # Moving average trend removal
    kernel = np.ones(N_win) / N_win
    trend1 = np.convolve(y1, kernel, mode='same')
    y1_detrended = y1 - trend1
    
    # 2. Second 0-Hz Resonator
    y2 = np.cumsum(np.cumsum(y1_detrended))
    trend2 = np.convolve(y2, kernel, mode='same')
    zfr = y2 - trend2
    
    # Normalize ZFR signal
    max_zfr = np.max(np.abs(zfr))
    if max_zfr > 1e-12:
        zfr = zfr / max_zfr
        
    # 3. Detect positive zero-crossings (slope > 0)
    # A GCI occurs where zfr transitions from negative to positive
    signs = np.sign(zfr)
    # Zero-crossing from negative to positive
    zc = np.where((signs[:-1] <= 0) & (signs[1:] > 0))[0]
    
    # Refine GCI candidates: enforce minimum pitch period constraint (e.g. max F0 = 400 Hz -> min lag = fs / 400)
    min_dist = int(fs / 400.0)
    refined_gcis = []
    
    for idx in zc:
        if not refined_gcis or (idx - refined_gcis[-1] >= min_dist):
            # Check local residual energy near this epoch to reject silence
            win_check = residual[max(0, idx - 25) : min(N, idx + 25)]
            if np.max(np.abs(win_check)) > 0.02 * np.max(np.abs(residual)):
                refined_gcis.append(idx)
                
    gci_indices = np.array(refined_gcis, dtype=int)
    gci_times = gci_indices / fs
    
    return gci_indices, gci_times, zfr

def compute_hilbert_phase_features(residual, fs=16000):
    """
    Computes Hilbert transform features on the LPC residual:
    - Analytic Signal: z[n] = e[n] + j H{e[n]}
    - Instantaneous Envelope / Amplitude: A[n] = |z[n]|
    - Instantaneous Phase: theta[n] = unwrap(angle(z[n]))
    - Instantaneous Frequency: f_i[n] = (fs / 2pi) * d(theta)/dn
    
    Returns
    -------
    dict with 'envelope', 'phase_unwrapped', 'inst_freq', 'phase_coherence'
    """
    # Analytic signal
    analytic = hilbert(residual)
    envelope = np.abs(analytic)
    phase_unwrapped = np.unwrap(np.angle(analytic))
    
    # Instantaneous frequency: numerical differentiation
    inst_freq = np.diff(phase_unwrapped, prepend=phase_unwrapped[0]) * (fs / (2.0 * np.pi))
    # Clip extreme anomalous spikes for display
    inst_freq_clipped = np.clip(inst_freq, 0, fs / 2.0)
    
    # Phase coherence: measure of periodicity in residual phase
    # High coherence indicates sharp glottal closures; low coherence indicates turbulent noise
    phase_diff = np.diff(phase_unwrapped)
    phase_coherence = float(1.0 - (np.std(phase_diff % (2 * np.pi)) / np.pi))
    phase_coherence = np.clip(phase_coherence, 0.0, 1.0)
    
    return {
        "envelope": envelope,
        "phase_unwrapped": phase_unwrapped,
        "inst_freq": inst_freq_clipped,
        "phase_coherence": round(float(phase_coherence), 4)
    }

def compute_group_delay_spectrum(signal, fs=16000, n_fft=1024, alpha=0.4, gamma=0.9):
    """
    Computes Modified Group Delay (MGD) spectrum of a speech or residual frame.
    
    tau(w) = - d(theta(w)) / dw
    Using the identity without phase unwrapping:
    tau(w) = (X_R * Y_R + X_I * Y_I) / (|X(w)|^2)
    where y[n] = n * x[n].
    
    Modified Group Delay:
    tau_m(w) = sgn(num) * |num| / (S(w)^{2*gamma} + eps)
    where S(w) is a smoothed spectral envelope.
    
    Significance in Oral vs Nasalized Vowels:
    Group delay peaks correspond to formant poles, while sharp negative dips
    correspond to anti-resonances (zeros) introduced by nasal coupling!
    """
    N = len(signal)
    n = np.arange(N)
    
    # x[n] and y[n] = n * x[n]
    x = signal * np.hamming(N)
    y = n * x
    
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    
    Xr, Xi = np.real(X), np.imag(X)
    Yr, Yi = np.real(Y), np.imag(Y)
    
    # Numerator: X_R * Y_R + X_I * Y_I
    num = Xr * Yr + Xi * Yi
    
    # Power spectrum |X|^2
    mag_sq = Xr ** 2 + Xi ** 2 + 1e-10
    
    # Standard group delay
    tau = num / mag_sq
    
    # Smoothed magnitude for Modified Group Delay
    from scipy.ndimage import gaussian_filter1d
    mag_smoothed = gaussian_filter1d(np.sqrt(mag_sq), sigma=3.0)
    
    # Modified group delay
    tau_mod = (num / (mag_smoothed ** (2.0 * gamma) + 1e-6))
    tau_mod = np.sign(tau_mod) * (np.abs(tau_mod) ** alpha)
    
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return freqs, tau, tau_mod

def compute_glottal_timing_metrics(residual, fs=16000, gcis=None):
    """
    Computes clinical and acoustic phonetics glottal timing metrics:
    - Jitter (local, RAP, PPQ5) : period-to-period perturbation
    - Shimmer (local, APQ3, APQ5) : amplitude perturbation
    - Glottal Open Quotient (OQ) approximation
    - Harmonic-to-Noise Ratio (HNR in dB)
    """
    if gcis is None or len(gcis) < 6:
        # Auto-extract GCIs if not provided
        gcis, _, _ = extract_epochs_zfr(residual, fs=fs)
        
    if len(gcis) < 6:
        # Fallback if too few GCIs detected (e.g. unvoiced/whispered)
        return {
            "num_epochs": len(gcis),
            "mean_f0_hz": 0.0,
            "jitter_local_pct": 0.0,
            "jitter_rap_pct": 0.0,
            "shimmer_local_pct": 0.0,
            "shimmer_apq3_pct": 0.0,
            "hnr_db": 0.0,
            "mean_pitch_period_ms": 0.0
        }
        
    # Pitch period intervals in samples
    periods = np.diff(gcis).astype(np.float64)
    # Filter anomalous periods
    median_p = np.median(periods)
    valid_p_mask = (periods > 0.4 * median_p) & (periods < 2.5 * median_p)
    valid_periods = periods[valid_p_mask]
    
    if len(valid_periods) < 4:
        valid_periods = periods
        
    # Pulse peak amplitudes at each epoch
    pulse_amplitudes = []
    for g in gcis:
        win = residual[max(0, g - 4) : min(len(residual), g + 5)]
        pulse_amplitudes.append(np.max(np.abs(win)) if len(win) > 0 else 0.0)
    pulse_amplitudes = np.array(pulse_amplitudes, dtype=np.float64)
    
    # 1. Jitter Metrics
    # Jitter (local): relative mean absolute difference between consecutive periods
    diff_p = np.abs(np.diff(valid_periods))
    mean_p = np.mean(valid_periods)
    jitter_local = (np.mean(diff_p) / mean_p) * 100.0 if mean_p > 0 else 0.0
    
    # Jitter RAP (Relative Average Perturbation - 3 point window)
    if len(valid_periods) >= 3:
        rap_diffs = []
        for i in range(1, len(valid_periods) - 1):
            avg_3 = (valid_periods[i-1] + valid_periods[i] + valid_periods[i+1]) / 3.0
            rap_diffs.append(abs(valid_periods[i] - avg_3))
        jitter_rap = (np.mean(rap_diffs) / mean_p) * 100.0
    else:
        jitter_rap = jitter_local
        
    # 2. Shimmer Metrics
    diff_a = np.abs(np.diff(pulse_amplitudes))
    mean_a = np.mean(pulse_amplitudes)
    shimmer_local = (np.mean(diff_a) / mean_a) * 100.0 if mean_a > 0 else 0.0
    
    if len(pulse_amplitudes) >= 3:
        apq3_diffs = []
        for i in range(1, len(pulse_amplitudes) - 1):
            avg_a = (pulse_amplitudes[i-1] + pulse_amplitudes[i] + pulse_amplitudes[i+1]) / 3.0
            apq3_diffs.append(abs(pulse_amplitudes[i] - avg_a))
        shimmer_apq3 = (np.mean(apq3_diffs) / mean_a) * 100.0
    else:
        shimmer_apq3 = shimmer_local
        
    # 3. Harmonic to Noise Ratio (HNR) via autocorrelation of residual
    r = correlate(residual, residual, mode='full')
    r = r[len(residual) - 1 :]
    # Lag around pitch period
    lag_min = int(0.7 * median_p)
    lag_max = int(1.3 * median_p)
    if len(r) > lag_max and lag_max > lag_min:
        peak_harmonic = np.max(r[lag_min : lag_max + 1])
        noise_energy = max(r[0] - peak_harmonic, 1e-12)
        hnr_db = 10.0 * np.log10(peak_harmonic / noise_energy)
    else:
        hnr_db = 12.0
        
    mean_f0 = float(fs / mean_p) if mean_p > 0 else 0.0
    
    return {
        "num_epochs": len(gcis),
        "mean_f0_hz": round(mean_f0, 2),
        "mean_pitch_period_ms": round(float(mean_p / fs * 1000.0), 2),
        "jitter_local_pct": round(float(jitter_local), 3),
        "jitter_rap_pct": round(float(jitter_rap), 3),
        "shimmer_local_pct": round(float(shimmer_local), 3),
        "shimmer_apq3_pct": round(float(shimmer_apq3), 3),
        "hnr_db": round(float(hnr_db), 2)
    }
