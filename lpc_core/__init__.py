"""
LPC Core Module
---------------
Linear Predictive Coding (LPC) Speech Analysis, Resynthesis,
Glottal Phase & Timing, and Acoustic Classification Suite.
Covers the entire modular engineering pipeline:
  Module 1: Preprocessing & Conditioning
  Module 2: LPC Estimation, Inverse Filtering & Vocoding
  Module 3: Phase & Glottal Timing (ZFR Epochs, Hilbert, Group Delay)
  Module 4: Acoustic Discrimination, DTW & Minimal Pair Classification
"""

from .preprocessor import (
    load_and_preprocess_audio,
    pre_emphasis,
    de_emphasis,
    save_wav_file
)

from .lpc_analyzer import (
    extract_lpc_coefficients,
    levinson_durbin,
    compute_autocorrelation,
    frame_signal
)

from .inverse_filter import (
    compute_lpc_residual,
    inverse_filter_frame
)

from .formant_analyzer import (
    extract_formants_from_lpc,
    compute_lpc_spectrum,
    track_formants
)
# Alias for backwards compatibility
extract_formants = extract_formants_from_lpc

from .features import (
    extract_lpc_features,
    estimate_pitch_from_residual,
    compute_prediction_gain,
    compute_spectral_flatness
)

from .advanced_lpc import (
    burg_method,
    lpc_to_lsf,
    lpc_to_lpcc,
    lpcc_distance,
    lpc_spectral_distance
)

from .resynthesis import (
    lpc_resynthesize_residual,
    lpc_resynthesize_vocoder,
    generate_synthetic_excitation,
    warp_lpc_formants,
    modify_speech
)

from .phase_timing import (
    extract_epochs_zfr,
    compute_hilbert_phase_features,
    compute_group_delay_spectrum,
    compute_glottal_timing_metrics
)

from .classifier import (
    compute_dtw,
    extract_acoustic_features,
    OralNasalClassifier,
    run_corpus_classification_benchmark
)

from .visualizer import (
    plot_waveform_and_residual,
    plot_spectrogram_comparison,
    plot_lpc_spectral_envelope,
    plot_word_comparison,
    plot_zplane_poles,
    plot_dtw_alignment,
    plot_phase_timing,
    plot_group_delay
)

__all__ = [
    # Preprocessor
    "load_and_preprocess_audio",
    "pre_emphasis",
    "de_emphasis",
    "save_wav_file",
    # LPC Analyzer
    "extract_lpc_coefficients",
    "levinson_durbin",
    "compute_autocorrelation",
    "frame_signal",
    # Inverse Filter
    "compute_lpc_residual",
    "inverse_filter_frame",
    # Formants
    "extract_formants_from_lpc",
    "extract_formants",
    "compute_lpc_spectrum",
    "track_formants",
    # Features
    "extract_lpc_features",
    "estimate_pitch_from_residual",
    "compute_prediction_gain",
    "compute_spectral_flatness",
    # Advanced LPC
    "burg_method",
    "lpc_to_lsf",
    "lpc_to_lpcc",
    "lpcc_distance",
    "lpc_spectral_distance",
    # Resynthesis & Vocoder
    "lpc_resynthesize_residual",
    "lpc_resynthesize_vocoder",
    "generate_synthetic_excitation",
    "warp_lpc_formants",
    "modify_speech",
    # Phase & Timing
    "extract_epochs_zfr",
    "compute_hilbert_phase_features",
    "compute_group_delay_spectrum",
    "compute_glottal_timing_metrics",
    # Classifier & DTW
    "compute_dtw",
    "extract_acoustic_features",
    "OralNasalClassifier",
    "run_corpus_classification_benchmark",
    # Visualizer
    "plot_waveform_and_residual",
    "plot_spectrogram_comparison",
    "plot_lpc_spectral_envelope",
    "plot_word_comparison",
    "plot_zplane_poles",
    "plot_dtw_alignment",
    "plot_phase_timing",
    "plot_group_delay"
]
