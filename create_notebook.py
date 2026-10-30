"""
create_notebook.py
------------------
Programmatically generates the interactive Jupyter Notebook:
'LPC_Analysis_Walkthrough.ipynb'

Covers all 4 project stages across 10 interactive DSP steps:
- Stage 1: Audio Loading, Pre-emphasis & Conditioning
- Stage 2: LPC Levinson-Durbin & Burg Lattice Modeling
- Stage 3: Acoustic Inverse Filtering & Residual Audio Extraction
- Stage 4: Formant Tracking & Z-Plane Complex Pole Geometry
- Stage 5: LPC Resynthesis & Reconstruction SNR Verification
- Stage 6: Line Spectral Frequencies (LSF) & Cepstral LPCC Recursion
- Stage 7: Zero Frequency Resonator (ZFR) & GCI Glottal Epochs
- Stage 8: Hilbert Phase Dynamics & Modified Group Delay (MGD)
- Stage 9: Dynamic Time Warping (DTW) Minimal Pair Trajectory Alignment
- Stage 10: AI Minimal Pair Random Forest Classifier & Deliverables Export
"""

import json
import os

def create_notebook():
    cells = []

    def md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Title & Introduction
    md(r"""
# 🎙️ Complete Linear Predictive Coding (LPC) Speech & Voice Analysis Suite
## Acoustic Phonetics Project: Minimal Pair Analysis ('कहा' vs 'कहाँ')
### 4-Stage Acoustic Signal Processing & Computational Phonetics Pipeline:
- **Stage 1 (Audio Acquisition & Conditioning)**: Standardized 16 kHz conditioning, DC bias removal, and high-pass pre-emphasis filter.
- **Stage 2 (LPC Modeling & Resynthesis)**: Levinson-Durbin Yule-Walker solver, Burg lattice method, inverse filtering, formant tracking, LSF/LPCC, and residual resynthesis.
- **Stage 3 (Glottal Phase & Temporal Dynamics)**: Zero Frequency Resonator (ZFR) Glottal Closure Instants (GCIs), Hilbert analytic envelope, and Modified Group Delay (MGD) phase spectrum.
- **Stage 4 (Pattern Recognition & DTW)**: Acoustic discrimination, Dynamic Time Warping (DTW), and explainable AI Random Forest classification.

---
""")

    # Step 0: Imports
    md(r"""
## 0. Import LPC Core DSP Suite
Initializes all 44 modular functions across the four architectural modules.
""")

    code(r"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from IPython.display import Audio, display

import lpc_core
from lpc_core.preprocessor import load_and_preprocess_audio, pre_emphasis, de_emphasis, save_wav_file
from lpc_core.lpc_analyzer import extract_lpc_coefficients, levinson_durbin, compute_autocorrelation
from lpc_core.inverse_filter import compute_lpc_residual
from lpc_core.formant_analyzer import extract_formants_from_lpc, compute_lpc_spectrum
from lpc_core.features import extract_lpc_features, estimate_pitch_from_residual
from lpc_core.advanced_lpc import burg_method, lpc_to_lsf, lpc_to_lpcc, lpcc_distance
from lpc_core.resynthesis import lpc_resynthesize_residual, lpc_resynthesize_vocoder, modify_speech
from lpc_core.phase_timing import (
    extract_epochs_zfr,
    compute_hilbert_phase_features,
    compute_group_delay_spectrum,
    compute_glottal_timing_metrics
)
from lpc_core.classifier import (
    extract_acoustic_features,
    compute_dtw,
    OralNasalClassifier,
    run_corpus_classification_benchmark
)
from lpc_core.visualizer import (
    plot_waveform_and_residual,
    plot_spectrogram_comparison,
    plot_lpc_spectral_envelope,
    plot_zplane_poles,
    plot_dtw_alignment,
    plot_phase_timing,
    plot_group_delay
)

print("[OK] LPC Advanced Suite Environment Initialized Successfully! (44 DSP modules)")
""")

    # Step 1: Preprocessing
    md(r"""
## 1. Stage 1: Audio Loading, Conditioning & Pre-emphasis
Prepares the speech signal with mono conversion, 16 kHz resampling, DC offset removal, and high-pass pre-emphasis filter ($y[n] = x[n] - 0.97 x[n-1]$) to flatten glottal spectral tilt.
""")

    code(r"""
audio_file = "data/kaha.wav"
proc_signal, raw_signal, fs = load_and_preprocess_audio(
    audio_file,
    target_fs=16000,
    apply_preemph=True,
    preemph_coeff=0.97
)

print(f"Audio File:      {audio_file}")
print(f"Sampling Rate:   {fs} Hz")
print(f"Total Samples:   {len(raw_signal):,}")
print(f"Duration:        {len(raw_signal) / fs:.3f} seconds")

print("\nListen to Original Audio:")
display(Audio(raw_signal, rate=fs))
""")

    # Step 2: LPC Modeling
    md(r"""
## 2. Stage 2: Frame-by-Frame LPC Modeling (Levinson-Durbin)
Estimates the all-pole filter coefficients $a = [1.0, a_1, \dots, a_p]$ using 25 ms Hamming windows with 10 ms hop (order $p = 16$).
""")

    code(r"""
lpc_order = 16
lpc_res = extract_lpc_coefficients(
    proc_signal,
    fs=fs,
    order=lpc_order,
    frame_ms=25.0,
    hop_ms=10.0,
    window_type="hamming"
)
lpc_coeffs = lpc_res['lpc_coeffs']

print(f"Number of frames:        {lpc_coeffs.shape[0]}")
print(f"Coefficients per frame:  {lpc_coeffs.shape[1]} (Leading 1.0 for A(z))")
mid_frame = len(lpc_coeffs) // 2
print(f"Mid-frame coefficients:  {np.round(lpc_coeffs[mid_frame][:6], 4)}")
""")

    # Step 3: Inverse Filtering
    md(r"""
## 3. Stage 2: Acoustic Inverse Filtering & Residual Extraction
Inverse filtering strips the vocal tract resonances by applying $A(z) = 1 + \sum_{k=1}^p a_k z^{-k}$, isolating the glottal prediction error / excitation signal $e[n]$.
""")

    code(r"""
residual = compute_lpc_residual(proc_signal, lpc_res, method="sample_varying")
norm_res = residual / (np.max(np.abs(residual)) + 1e-9) * 0.90

print(f"Residual signal computed! Total samples: {len(residual):,}")
fig1 = plot_waveform_and_residual(raw_signal, residual, fs, title="'kaha' (कहा)")
plt.show()

print("Listen to LPC Residual Audio (Excitation pulses with formants stripped away):")
display(Audio(norm_res, rate=fs))
""")

    # Step 4: Resynthesis
    md(r"""
## 4. Stage 2: Speech Resynthesis & Quality Evaluation
Resynthesizes speech by feeding the residual back through the time-varying synthesis filter $H(z) = 1 / A(z)$ and applying de-emphasis.
""")

    code(r"""
resynth_speech = lpc_resynthesize_residual(residual, lpc_res, de_emph=True)

# Calculate Reconstruction Signal-to-Noise Ratio (SNR)
min_l = min(len(raw_signal), len(resynth_speech))
sig_p = np.mean(raw_signal[:min_l] ** 2)
err_p = np.mean((raw_signal[:min_l] - resynth_speech[:min_l]) ** 2) + 1e-12
snr_db = 10.0 * np.log10(sig_p / err_p)

print(f"Reconstruction Signal-to-Noise Ratio (SNR): {snr_db:.2f} dB (High fidelity)")
print("Listen to LPC Resynthesized Audio:")
display(Audio(resynth_speech, rate=fs))
""")

    # Step 5: Formants & Z-Plane Poles
    md(r"""
## 5. Stage 2: Formant Envelopes & Z-Plane Complex Poles
Computes vocal tract formant frequencies ($F_1 - F_4$) from the polynomial roots and plots pole-zero geometry on the unit circle.
""")

    code(r"""
mid_idx = len(lpc_coeffs) // 2
formants, bws = extract_formants_from_lpc(lpc_coeffs[mid_idx], fs=fs)
for i, (f, b) in enumerate(zip(formants[:4], bws[:4])):
    print(f"  -> Formant F{i+1}: {int(f)} Hz (Bandwidth: {int(b)} Hz)")

# Z-plane pole plot
fig_poles = plot_zplane_poles(lpc_coeffs[mid_idx], fs=fs, title="'kaha' All-Pole Vocal Tract Filter in Z-Plane")
plt.show()
""")

    # Step 6: LSF and LPCC
    md(r"""
## 6. Stage 2: Line Spectral Frequencies (LSF) & Cepstral LPCC
Calculates Line Spectral Frequencies (interlaced zeros of symmetric/anti-symmetric polynomials) and LPCC cepstral recursion.
""")

    code(r"""
# Line Spectral Frequencies (LSF)
lsf = lpc_to_lsf(lpc_coeffs[mid_idx])
print("Line Spectral Frequencies (rad):", np.round(lsf, 4))
print("LSF strictly monotonic:", bool(np.all(np.diff(lsf) > 0)))

# LPC Cepstral Coefficients (LPCC)
lpcc = lpc_to_lpcc(lpc_coeffs[mid_idx], num_cep=12)
print("LPCC Vector (12 cepstra):", np.round(lpcc, 4))
""")

    # Step 7: Glottal Epoch Extraction (ZFR)
    md(r"""
## 7. Stage 3: Glottal Closure Instant (GCI / Epoch) Extraction via ZFR
Passes the LPC residual through the Zero Frequency Resonator (ZFR) to detect exact glottal closure timings.
""")

    code(r"""
gcis, gci_times, zfr = extract_epochs_zfr(residual, fs=fs)
timing_metrics = compute_glottal_timing_metrics(residual, fs=fs, gcis=gcis)

print(f"Total Glottal Closure Epochs: {len(gcis)}")
print(f"Mean Pitch F0:                {timing_metrics['mean_f0_hz']} Hz")
print(f"Local Jitter:                 {timing_metrics['jitter_local_pct']}%")
print(f"Local Shimmer:                {timing_metrics['shimmer_local_pct']}%")
print(f"Harmonic-to-Noise Ratio:      {timing_metrics['hnr_db']} dB")

fig_zfr = plot_phase_timing(residual, zfr, gcis, fs=fs, title="'kaha'")
plt.show()
""")

    # Step 8: Group Delay Spectrum
    md(r"""
## 8. Stage 3: Modified Group Delay (MGD) Spectrum
Resolves acoustic formant poles and nasal anti-resonances (zeros) in the phase domain.
""")

    code(r"""
mid_sample = len(residual) // 2
vowel_frame = residual[mid_sample : mid_sample + 400]
mgd_freqs, _, tau_mod = compute_group_delay_spectrum(vowel_frame, fs=fs, n_fft=1024)

fig_mgd = plot_group_delay(mgd_freqs, tau_mod, fs=fs, title="Modified Group Delay Spectrum: 'kaha'")
plt.show()
""")

    # Step 9: Dynamic Time Warping
    md(r"""
## 9. Stage 4: Dynamic Time Warping (DTW) Alignment
Aligns feature trajectories between the minimal pair 'kaha' (oral) and 'kahaan' (nasalized).
""")

    code(r"""
f1_data = extract_acoustic_features("data/kaha.wav")
f2_data = extract_acoustic_features("data/kahaan.wav")

dtw_dist, cost_mat, path = compute_dtw(f1_data['lpcc_matrix'], f2_data['lpcc_matrix'])
print(f"Normalized DTW Alignment Distance: {dtw_dist:.4f}")

fig_dtw = plot_dtw_alignment(cost_mat, path, name1="kaha", name2="kahaan", dist=dtw_dist)
plt.show()
""")

    # Step 10: Machine Learning Classification & Deliverables
    md(r"""
## 10. Stage 4: Automated AI Minimal Pair Classification & Deliverables
Runs the trained Random Forest classifier to distinguish oral vs nasalized vowels with confidence, explainability, and exports all deliverables.
""")

    code(r"""
clf = OralNasalClassifier()

res_oral = clf.predict("data/kaha.wav")
print(f"Result for 'kaha.wav':   {res_oral['predicted_class']} (Confidence: {res_oral['confidence']*100:.1f}%)")

res_nasal = clf.predict("data/kahaan.wav")
print(f"Result for 'kahaan.wav': {res_nasal['predicted_class']} (Confidence: {res_nasal['confidence']*100:.1f}%)")
print("\nAcoustic Evidence for 'kahaan.wav':")
for r in res_nasal['acoustic_explanation']:
    print(f"  * {r}")

# Export Deliverables
out_dir = "outputs/kaha"
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, "lpc_coefficients.npy"), lpc_coeffs)
np.save(os.path.join(out_dir, "lpc_residual.npy"), residual)
save_wav_file(os.path.join(out_dir, "lpc_residual.wav"), norm_res, fs=fs)
save_wav_file(os.path.join(out_dir, "lpc_resynthesized.wav"), resynth_speech, fs=fs)

print("\n📁 All Deliverables Saved Successfully in:", out_dir)
print("  1. lpc_coefficients.npy")
print("  2. lpc_residual.npy")
print("  3. lpc_residual.wav")
print("  4. lpc_resynthesized.wav")
""")

    # Build notebook JSON structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    target_path = "LPC_Analysis_Walkthrough.ipynb"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"[OK] Generated '{target_path}' successfully with {len(cells)} cells!")

if __name__ == "__main__":
    create_notebook()
