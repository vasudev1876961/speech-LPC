"""
lpc_core/advanced_lpc.py
-------------------------
Advanced LPC parameterizations, alternative estimation algorithms, and representations:
1. Burg's Lattice Method (reflection coefficients computed directly from forward & backward errors)
2. Line Spectral Frequencies / Pairs (LSF / LSP) - industry standard for speech coding (AMR-WB, GSM)
3. Linear Predictive Cepstral Coefficients (LPCC) - cepstral recursion for speech recognition
4. Quefrency-weighted Cepstral Distance & LPC Spectral Distance
"""

import numpy as np
from scipy.signal import freqz

def burg_method(frame, order):
    """
    Computes LPC coefficients using Burg's lattice filter method.
    
    Advantages of Burg's method:
    - Guaranteed minimum-phase (stable) all-pole filter (all reflection coeffs |k_i| <= 1).
    - Minimizes both forward and backward prediction errors simultaneously.
    - Does not window the signal as heavily, yielding superior resolution on short frames.
    
    Parameters
    ----------
    frame : np.ndarray
        1D speech frame of length N.
    order : int
        LPC prediction order p.
        
    Returns
    -------
    a : np.ndarray
        LPC coefficients [1.0, a_1, ..., a_p] of length order + 1.
    k : np.ndarray
        Reflection coefficients (PARCOR) of length order.
    E : float
        Residual error energy.
    """
    N = len(frame)
    if N <= order:
        raise ValueError(f"Frame length {N} must be greater than order {order}")
        
    # Initialize forward and backward prediction errors
    f = frame.astype(np.float64).copy()
    b = frame.astype(np.float64).copy()
    
    # Initial error energy
    E = np.dot(f, f) / N
    if E < 1e-12:
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        return a, np.zeros(order), 0.0
        
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    k = np.zeros(order, dtype=np.float64)
    
    # Store intermediate prediction polynomials
    a_prev = np.array([1.0], dtype=np.float64)
    
    for m in range(1, order + 1):
        # Calculate reflection coefficient k_m
        num = -2.0 * np.dot(b[:-1], f[1:])
        den = np.dot(f[1:], f[1:]) + np.dot(b[:-1], b[:-1])
        
        if den < 1e-12:
            km = 0.0
        else:
            km = float(num / den)
            km = np.clip(km, -0.99999, 0.99999) # Enforce strict stability
            
        k[m - 1] = km
        
        # Update filter coefficients via Levinson recursion
        a_curr = np.zeros(m + 1, dtype=np.float64)
        a_curr[0] = 1.0
        for j in range(1, m):
            a_curr[j] = a_prev[j] + km * a_prev[m - j]
        a_curr[m] = km
        
        a_prev = a_curr.copy()
        
        # Update forward and backward errors for next stage
        f_next = f[1:] + km * b[:-1]
        b_next = b[:-1] + km * f[1:]
        f = f_next
        b = b_next
        
        # Update error energy
        E = E * (1.0 - km * km)
        
    a = np.zeros(order + 1, dtype=np.float64)
    a[:len(a_prev)] = a_prev
    return a, k, float(E)

def lpc_to_lsf(a):
    r"""
    Converts LPC polynomial coefficients A(z) into Line Spectral Frequencies (LSF / LSP).
    
    Mathematical Formulation:
    Symmetric polynomial:     P(z) = A(z) + z^{-(p+1)} A(z^{-1})
    Antisymmetric polynomial: Q(z) = A(z) - z^{-(p+1)} A(z^{-1})
    
    The roots of P(z) and Q(z) all lie on the unit circle (e^{j \omega_i}) and interlace:
    0 < \omega_{P1} < \omega_{Q1} < \omega_{P2} < \omega_{Q2} < ... < \pi
    
    Parameters
    ----------
    a : np.ndarray
        LPC coefficients [1.0, a_1, ..., a_p].
        
    Returns
    -------
    lsf : np.ndarray
        Normalized Line Spectral Frequencies in radians [0, pi], sorted ascending.
    """
    order = len(a) - 1
    
    # Form P(z) and Q(z) coefficients
    # P(z) = A(z) + z^{-(p+1)} A(1/z)
    # Q(z) = A(z) - z^{-(p+1)} A(1/z)
    a_flip = a[::-1]
    
    p_poly = np.zeros(order + 2, dtype=np.float64)
    q_poly = np.zeros(order + 2, dtype=np.float64)
    
    p_poly[:order + 1] += a
    p_poly[1:] += a_flip
    
    q_poly[:order + 1] += a
    q_poly[1:] -= a_flip
    
    # Find roots of P and Q
    roots_p = np.roots(p_poly)
    roots_q = np.roots(q_poly)
    
    # Extract angles on upper half of unit circle [0, pi]
    angles_p = np.angle(roots_p[np.imag(roots_p) >= 0])
    angles_q = np.angle(roots_q[np.imag(roots_q) >= 0])
    
    # Filter out trivial roots at z = -1 (angle = pi) or z = 1 (angle = 0)
    angles_p = angles_p[(angles_p > 1e-4) & (angles_p < np.pi - 1e-4)]
    angles_q = angles_q[(angles_q > 1e-4) & (angles_q < np.pi - 1e-4)]
    
    # Combine and sort all LSF angles
    lsf = np.sort(np.concatenate([angles_p, angles_q]))
    
    # If numerical root finding missed any due to multiplicity, truncate or pad to order
    if len(lsf) > order:
        lsf = lsf[:order]
    elif len(lsf) < order:
        # Fallback: estimate from phase spectrum
        w = np.linspace(0.05 * np.pi, 0.95 * np.pi, order)
        lsf = np.sort(w)
        
    return lsf

def lpc_to_lpcc(a, num_cep=16, gain=1.0):
    """
    Converts LPC filter coefficients a = [1.0, a_1, ..., a_p] into
    LPC Cepstral Coefficients (LPCC) using the exact recursive formula:
    
    c_0 = ln(gain)
    c_1 = -a_1
    c_m = -a_m - sum_{k=1}^{m-1} (1 - k/m) * a_k * c_{m-k}  (for 1 < m <= p)
    c_m = - sum_{k=1}^p (1 - k/m) * a_k * c_{m-k}           (for m > p)
    
    Parameters
    ----------
    a : np.ndarray
        LPC coefficients [1.0, a_1, ..., a_p].
    num_cep : int
        Number of cepstral coefficients to return.
    gain : float
        Prediction gain.
        
    Returns
    -------
    lpcc : np.ndarray
        Array of cepstral coefficients c_1, c_2, ..., c_{num_cep}.
    """
    p = len(a) - 1
    # Coefficients index 1 to p: a_k
    a_k = a[1:]
    
    c = np.zeros(num_cep + 1, dtype=np.float64)
    c[0] = np.log(max(gain, 1e-6))
    
    for m in range(1, num_cep + 1):
        sum_val = 0.0
        for k in range(1, m):
            if (m - k) <= p:
                sum_val += (k / float(m)) * c[k] * a_k[m - k - 1]
                
        if m <= p:
            c[m] = -a_k[m - 1] - sum_val
        else:
            c[m] = -sum_val
            
    return c[1:] # Return c_1 ... c_{num_cep}

def lpcc_distance(lpcc1, lpcc2, weights="bandpass"):
    """
    Computes Euclidean or Quefrency-weighted distance between two LPCC vectors.
    
    Parameters
    ----------
    lpcc1, lpcc2 : np.ndarray
        1D arrays of cepstral coefficients.
    weights : str
        'euclidean', 'quefrency' (multiplied by index), or 'bandpass' (raised sine window).
        
    Returns
    -------
    dist : float
        Scalar distance metric.
    """
    K = len(lpcc1)
    if weights == "bandpass":
        # Raised sine window w_m = 1 + (K/2)*sin(pi*m/K) to de-emphasize low and high quefrencies
        m = np.arange(1, K + 1)
        w = 1.0 + (K / 2.0) * np.sin(np.pi * m / K)
    elif weights == "quefrency":
        w = np.arange(1, K + 1, dtype=np.float64)
    else:
        w = np.ones(K, dtype=np.float64)
        
    diff = lpcc1 - lpcc2
    return float(np.sqrt(np.sum(w * (diff ** 2))))

def lpc_spectral_distance(a1, a2, fs=16000, n_fft=512):
    """
    Computes Root-Mean-Square Spectral Distance (dB) between two LPC filter responses:
    SD = sqrt( (1 / pi) * integral_0^pi | 20 log10 |H1(w)/H2(w)| |^2 dw )
    """
    w1, h1 = freqz([1.0], a1, worN=n_fft, fs=fs)
    w2, h2 = freqz([1.0], a2, worN=n_fft, fs=fs)
    
    db1 = 20.0 * np.log10(np.maximum(np.abs(h1), 1e-10))
    db2 = 20.0 * np.log10(np.maximum(np.abs(h2), 1e-10))
    
    diff_db = db1 - db2
    return float(np.sqrt(np.mean(diff_db ** 2)))
