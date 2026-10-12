"""
lpc_core/inverse_filter.py
--------------------------
Calculates the LPC residual signal using inverse filtering:
A(z) = 1 + sum_{k=1}^p a_k * z^{-k}
e[n] = x[n] + sum_{k=1}^p a_k * x[n - k]
"""

import numpy as np
from scipy.signal import lfilter

def inverse_filter_frame(frame, lpc_coeff):
    """
    Applies the inverse filter A(z) to a single speech frame.
    
    Parameters
    ----------
    frame : np.ndarray
        1D array of speech samples.
    lpc_coeff : np.ndarray
        LPC filter coefficients [1.0, a_1, a_2, ..., a_p].
        
    Returns
    -------
    residual_frame : np.ndarray
        Prediction error / residual signal for the frame.
    """
    # Inverse filter: Numerator = lpc_coeff, Denominator = [1.0]
    return lfilter(lpc_coeff, [1.0], frame)

def compute_lpc_residual(signal, lpc_results, method="sample_varying"):
    """
    Computes the full continuous LPC residual signal across the entire utterance.
    
    Parameters
    ----------
    signal : np.ndarray
        Input speech signal (1D array of length N).
    lpc_results : dict
        Dictionary returned by extract_lpc_coefficients containing:
        - 'lpc_coeffs': shape (num_frames, p + 1)
        - 'indices': start sample index of each frame
        - 'frame_len': frame length in samples
        - 'hop_len': hop length in samples
        - 'order': LPC order p
    method : str
        'sample_varying' (recommended: exact time-alignment and smooth interpolation)
        or 'overlap_add' (classic frame-by-frame filtering with OLA reconstruction).
        
    Returns
    -------
    residual : np.ndarray
        Prediction error signal e[n], exactly aligned and same length as input signal.
    """
    lpc_coeffs = lpc_results['lpc_coeffs']
    indices = lpc_results['indices']
    frame_len = lpc_results['frame_len']
    hop_len = lpc_results['hop_len']
    order = lpc_results['order']
    num_frames = len(lpc_coeffs)
    num_samples = len(signal)
    
    if method == "sample_varying":
        # Sample-by-sample filtering with frame-center mapping
        # Maps each sample n to the closest analysis frame
        frame_centers = indices + (frame_len // 2)
        
        # For each sample n, find nearest frame index
        sample_indices = np.arange(num_samples)
        # Search sorted insertion points
        nearest_frame = np.clip(
            np.searchsorted(frame_centers, sample_indices, side='right') - 1,
            0,
            num_frames - 1
        )
        
        # Vectorized FIR computation
        # e[n] = x[n] + sum_{k=1}^p a_k[nearest_frame[n]] * x[n - k]
        # Pad signal at the beginning with p zeros
        padded_x = np.pad(signal, (order, 0), mode='constant')
        
        # Extract sliding history matrix: shape (num_samples, p)
        # For each n, history is [x[n-1], x[n-2], ..., x[n-p]]
        history = np.zeros((num_samples, order), dtype=np.float64)
        for k in range(1, order + 1):
            history[:, k - 1] = padded_x[order - k : order - k + num_samples]
            
        # Active a coefficients for each sample: shape (num_samples, p)
        active_a = lpc_coeffs[nearest_frame, 1:]
        
        # Predicted sample: x_hat[n] = - sum_{k=1}^p a_k * x[n-k]
        predicted = - np.sum(active_a * history, axis=1)
        
        # Residual error: e[n] = x[n] - x_hat[n] = x[n] + sum(a_k * x[n-k])
        residual = signal - predicted
        
    elif method == "overlap_add":
        # Frame-by-frame inverse filtering with Overlap-Add (OLA)
        residual = np.zeros(num_samples, dtype=np.float64)
        norm_weights = np.zeros(num_samples, dtype=np.float64)
        
        win = np.hanning(frame_len)
        
        for i in range(num_frames):
            start = indices[i]
            end = min(start + frame_len, num_samples)
            actual_len = end - start
            
            frame_seg = signal[start:end]
            if actual_len < frame_len:
                frame_seg = np.pad(frame_seg, (0, frame_len - actual_len), mode='constant')
                
            res_frame = lfilter(lpc_coeffs[i], [1.0], frame_seg)
            
            # Accumulate with window
            residual[start:end] += res_frame[:actual_len] * win[:actual_len]
            norm_weights[start:end] += win[:actual_len]
            
        # Normalize overlap
        mask = norm_weights > 1e-4
        residual[mask] /= norm_weights[mask]
        residual[~mask] = signal[~mask]
        
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'sample_varying' or 'overlap_add'.")
        
    return residual
