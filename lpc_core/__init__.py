"""
LPC Core Module
---------------
Linear Predictive Coding (LPC) and Inverse Filtering toolkit for speech analysis.
Designed for phonetic and acoustic differentiation projects.
"""

from .preprocessor import load_and_preprocess_audio, pre_emphasis, de_emphasis
from .lpc_analyzer import extract_lpc_coefficients, levinson_durbin, frame_signal
from .inverse_filter import compute_lpc_residual, inverse_filter_frame
from .formant_analyzer import extract_formants, compute_lpc_spectrum
from .features import extract_lpc_features

__all__ = [
    "load_and_preprocess_audio",
    "pre_emphasis",
    "de_emphasis",
    "extract_lpc_coefficients",
    "levinson_durbin",
    "frame_signal",
    "compute_lpc_residual",
    "inverse_filter_frame",
    "extract_formants",
    "compute_lpc_spectrum",
    "extract_lpc_features",
]
