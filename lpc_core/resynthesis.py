"""
lpc_core/resynthesis.py
------------------------
LPC Synthesis & Vocoder Engine:
1. Exact LPC Resynthesis from Residual: 1 / A(z) synthesis filtering
2. Parametric LPC Vocoder: Pitch-pulse train (voiced) + Gaussian noise (unvoiced)
3. Speech Morphing & Voice Transformation:
   - Pitch shifting (independent of vocal tract)
   - Formant warping / Vocal tract scaling (simulating anatomical length changes)
   - Whisperization (unvoiced turbulent excitation)
"""

import numpy as np
from scipy.signal import lfilter
from .preprocessor import de_emphasis

def lpc_resynthesize_residual(residual, lpc_results, de_emph=True, de_emph_coeff=0.97):
    """
    Resynthesizes speech by filtering the LPC residual e[n] through the time-varying
    all-pole synthesis filter:
    H(z) = 1 / A(z)
    
    x_hat[n] = e[n] - sum_{k=1}^p a_k * x_hat[n - k]
    
    Parameters
    ----------
    residual : np.ndarray
        The prediction error / excitation signal e[n].
    lpc_results : dict
        Analysis results containing 'lpc_coeffs', 'indices', 'frame_len', 'order'.
    de_emph : bool
        Whether to apply de-emphasis filter after synthesis.
        
    Returns
    -------
    reconstructed_signal : np.ndarray
        Synthesized speech signal.
    """
    lpc_coeffs = lpc_results['lpc_coeffs']
    indices = lpc_results['indices']
    frame_len = lpc_results['frame_len']
    order = lpc_results['order']
    num_frames = len(lpc_coeffs)
    num_samples = len(residual)
    
    # Map each sample to nearest analysis frame center
    frame_centers = indices + (frame_len // 2)
    sample_indices = np.arange(num_samples)
    nearest_frame = np.clip(
        np.searchsorted(frame_centers, sample_indices, side='right') - 1,
        0,
        num_frames - 1
    )
    
    # Time-varying IIR all-pole synthesis:
    # x[n] = e[n] - sum_{k=1}^p a_k * x[n-k]
    # We maintain a state vector of past p synthesized samples
    synth = np.zeros(num_samples, dtype=np.float64)
    state = np.zeros(order, dtype=np.float64) # [x[n-1], x[n-2], ..., x[n-p]]
    
    for n in range(num_samples):
        frame_idx = nearest_frame[n]
        a_k = lpc_coeffs[frame_idx, 1:] # length p
        
        # Current sample
        val = residual[n] - np.dot(a_k, state)
        
        # Stability safeguard against runaway resonance
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        elif np.abs(val) > 10.0:
            val = np.sign(val) * 10.0
            
        synth[n] = val
        
        # Update delay line: shift state right, insert val at state[0]
        state[1:] = state[:-1]
        state[0] = val
        
    # De-emphasis if requested
    if de_emph:
        synth = de_emphasis(synth, alpha=de_emph_coeff)
        
    # Normalize
    peak = np.max(np.abs(synth))
    if peak > 1e-6:
        synth = synth / peak * 0.90
        
    return synth

def generate_synthetic_excitation(pitch_contour, times, total_samples, fs=16000, gain_contour=None):
    """
    Generates synthetic source excitation for parametric LPC vocoding:
    - Voiced segments: Periodic Rosenberg glottal pulse train at pitch F0
    - Unvoiced segments: White Gaussian noise
    """
    excitation = np.zeros(total_samples, dtype=np.float64)
    num_frames = len(pitch_contour)
    
    # Interpolate pitch to sample level
    sample_times = np.arange(total_samples) / fs
    frame_times = times
    
    # Handle NaNs in pitch contour
    valid_mask = ~np.isnan(pitch_contour) & (pitch_contour > 40.0)
    
    if np.sum(valid_mask) > 1:
        # Interpolate pitch
        f0_interp = np.interp(sample_times, frame_times[valid_mask], pitch_contour[valid_mask])
    else:
        f0_interp = np.zeros(total_samples)
        
    # Interpolate voicing indicator
    voicing_frame = valid_mask.astype(float)
    voicing_interp = np.interp(sample_times, frame_times, voicing_frame) > 0.5
    
    # Generate excitation sample-by-sample
    phase = 0.0
    noise = np.random.normal(0.0, 0.25, total_samples)
    
    for n in range(total_samples):
        if voicing_interp[n] and f0_interp[n] > 40.0:
            period = fs / f0_interp[n]
            n_mod = phase % period
            tn = n_mod / period
            
            # Rosenberg glottal flow derivative pulse
            if tn < 0.4:
                pulse = 0.5 * (1.0 - np.cos(np.pi * tn / 0.4))
            elif tn < 0.6:
                pulse = np.cos(np.pi * (tn - 0.4) / (2.0 * 0.2))
            else:
                pulse = 0.0
                
            # Add slight unvoiced aspiration breathiness (5%)
            excitation[n] = pulse + 0.05 * noise[n]
            phase += 1.0
        else:
            # Unvoiced turbulent noise
            excitation[n] = noise[n]
            phase = 0.0
            
    # Apply energy gain contour if provided
    if gain_contour is not None and len(gain_contour) == num_frames:
        gain_interp = np.interp(sample_times, frame_times, gain_contour)
        excitation = excitation * gain_interp
        
    return excitation

def lpc_resynthesize_vocoder(pitch_contour, times, lpc_results, fs=16000, target_energy=None):
    """
    Parametric LPC Vocoder: Reconstructs speech using only LPC coefficients
    and estimated pitch F0 (no residual required). Used in low-bitrate speech coders.
    """
    total_samples = lpc_results['indices'][-1] + lpc_results['frame_len']
    
    # Calculate energy per frame to preserve volume contour
    lpc_coeffs = lpc_results['lpc_coeffs']
    num_frames = len(lpc_coeffs)
    
    if target_energy is None:
        target_energy = np.ones(num_frames)
        
    excitation = generate_synthetic_excitation(
        pitch_contour,
        times,
        total_samples,
        fs=fs,
        gain_contour=target_energy
    )
    
    # Resynthesize with synthetic excitation
    synth = lpc_resynthesize_residual(excitation, lpc_results, de_emph=True)
    return synth

def warp_lpc_formants(a, scale_factor=1.0, fs=16000):
    r"""
    Scales vocal-tract formant resonances by rotating polynomial roots in the z-plane:
    z_k = r_k * e^{j \omega_k}  -->  z_k' = r_k * e^{j (\omega_k * scale_factor)}
    
    - scale_factor > 1.0: Formants shifted higher (simulates shorter vocal tract / female or child voice)
    - scale_factor < 1.0: Formants shifted lower (simulates longer vocal tract / deep baritone voice)
    
    Parameters
    ----------
    a : np.ndarray
        LPC coefficients [1.0, a_1, ..., a_p].
    scale_factor : float
        Frequency scaling multiplier (e.g., 0.85 to 1.25).
    fs : int
        Sampling frequency in Hz.
        
    Returns
    -------
    a_warped : np.ndarray
        New LPC polynomial coefficients with shifted formant peaks.
    """
    if abs(scale_factor - 1.0) < 1e-4:
        return a.copy()
        
    roots = np.roots(a)
    warped_roots = []
    
    for r in roots:
        mag = np.abs(r)
        ang = np.angle(r)
        
        # Enforce stability: keep magnitude strictly inside unit circle
        mag = min(mag, 0.995)
        
        # Scale angle while clamping between 0 and pi
        new_ang = np.clip(ang * scale_factor, -0.98 * np.pi, 0.98 * np.pi)
        warped_roots.append(mag * np.exp(1j * new_ang))
        
    # Reconstruct polynomial from warped roots: P(z) = prod(z - r_i)
    new_poly = np.poly(warped_roots)
    a_warped = np.real(new_poly)
    
    # Normalize leading coefficient to 1.0
    if a_warped[0] != 0:
        a_warped = a_warped / a_warped[0]
        
    return a_warped

def modify_speech(signal, lpc_results, pitch_scale=1.0, formant_scale=1.0, whisper=False, fs=16000):
    """
    Comprehensive Speech Modification & Voice Morphing Studio:
    Allows independent transformation of:
    1. Pitch F0 (glottal source frequency)
    2. Formants (vocal tract geometry / speaker identity)
    3. Voicing mode (whispered vs natural)
    
    Parameters
    ----------
    signal : np.ndarray
        Input speech signal.
    lpc_results : dict
        Computed LPC analysis dictionary.
    pitch_scale : float
        Pitch shift factor (e.g., 1.2 for 20% higher pitch, 0.8 for 20% lower).
    formant_scale : float
        Vocal tract formant scaling (e.g., 1.15 for female/child vocal tract, 0.88 for deep voice).
    whisper : bool
        If True, replaces vocal cord vibration with unvoiced turbulent aspiration noise.
    fs : int
        Sampling frequency.
        
    Returns
    -------
    modified_audio : np.ndarray
        Synthesized transformed speech audio.
    """
    coeffs = lpc_results['lpc_coeffs'].copy()
    num_frames = len(coeffs)
    total_samples = len(signal)
    
    # Step 1: Formant warping on LPC coefficients if scale != 1.0
    if abs(formant_scale - 1.0) > 1e-3:
        warped_coeffs = np.zeros_like(coeffs)
        for i in range(num_frames):
            warped_coeffs[i] = warp_lpc_formants(coeffs[i], scale_factor=formant_scale, fs=fs)
        lpc_results_mod = lpc_results.copy()
        lpc_results_mod['lpc_coeffs'] = warped_coeffs
    else:
        lpc_results_mod = lpc_results
        
    # Step 2: Excitation generation
    if whisper:
        # Whispered speech: Pure noise excitation shaped by vocal tract filter
        noise = np.random.normal(0.0, 0.25, total_samples)
        modified_audio = lpc_resynthesize_residual(noise, lpc_results_mod, de_emph=True)
    else:
        # Voiced modification
        from .features import estimate_pitch_from_residual
        from .inverse_filter import compute_lpc_residual
        
        residual = compute_lpc_residual(signal, lpc_results, method="sample_varying")
        f0_contour, times = estimate_pitch_from_residual(residual, fs=fs)
        
        if abs(pitch_scale - 1.0) > 1e-3:
            # Scale pitch contour
            f0_scaled = f0_contour * pitch_scale
            # Measure frame energy
            frame_len = lpc_results['frame_len']
            hop_len = lpc_results['hop_len']
            energy_contour = []
            for idx in lpc_results['indices']:
                frm = signal[idx : idx + frame_len]
                energy_contour.append(np.std(frm) + 1e-4)
            energy_contour = np.array(energy_contour)
            
            modified_audio = lpc_resynthesize_vocoder(
                f0_scaled,
                times,
                lpc_results_mod,
                fs=fs,
                target_energy=energy_contour
            )
        else:
            # Resynthesize with original residual through warped/unwarped filter
            modified_audio = lpc_resynthesize_residual(residual, lpc_results_mod, de_emph=True)
            
    return modified_audio
