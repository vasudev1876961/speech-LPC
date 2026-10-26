"""
run_lpc.py
----------
Main execution pipeline for the LPC Speech Analysis Suite.
Processes an input speech WAV file, performs LPC coefficient estimation,
applies inverse filtering to extract the LPC residual, resynthesizes speech,
extracts glottal timing/phase metrics (Module 3), and classifies the sound (Module 4).

Outputs:
  1. lpc_coefficients.npy (p + 1 vocal tract coefficients)
  2. lpc_residual.npy (prediction error signal e[n])
  3. lpc_residual.wav (playable residual audio)
  4. lpc_resynthesized.wav (synthesized reconstructed speech audio)
  5. Visual analysis plots (waveform, spectrogram, envelope, z-plane poles)
  6. Phase & glottal timing deliverables (ZFR epochs, Hilbert, group delay)
  7. Comprehensive summary report (lpc_summary.json)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from lpc_core.preprocessor import load_and_preprocess_audio, save_wav_file
from lpc_core.lpc_analyzer import extract_lpc_coefficients
from lpc_core.inverse_filter import compute_lpc_residual
from lpc_core.formant_analyzer import extract_formants_from_lpc
from lpc_core.features import extract_lpc_features
from lpc_core.resynthesis import lpc_resynthesize_residual
from lpc_core.advanced_lpc import lpc_to_lpcc, lpc_to_lsf
from lpc_core.visualizer import (
    plot_waveform_and_residual,
    plot_spectrogram_comparison,
    plot_lpc_spectral_envelope,
    plot_zplane_poles
)
from lpc_core.phase_timing import (
    extract_epochs_zfr,
    compute_hilbert_phase_features,
    compute_group_delay_spectrum,
    compute_glottal_timing_metrics
)
from lpc_core.classifier import OralNasalClassifier

def process_file(
    input_wav,
    output_dir=None,
    order=16,
    frame_ms=25.0,
    hop_ms=10.0,
    pre_emphasis=True,
    full_pipeline=True,
    verbose=True
):
    """
    Runs the complete LPC processing pipeline on a single audio file.
    """
    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Audio file not found: {input_wav}")
        
    base_name = os.path.splitext(os.path.basename(input_wav))[0]
    if output_dir is None:
        output_dir = os.path.join("outputs", base_name)
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print(f"  LINEAR PREDICTIVE CODING (LPC) ADVANCED PIPELINE")
        print(f"  Input File:     {input_wav}")
        print(f"  LPC Order (p):  {order}")
        print(f"  Frame Length:   {frame_ms} ms")
        print(f"  Hop Length:     {hop_ms} ms")
        print(f"  Full Pipeline:  {full_pipeline}")
        print(f"  Output Folder:  {output_dir}")
        print("=" * 70)

    # -------------------------------------------------------------
    # Step 1 & 2: Load and Prepare Audio
    # -------------------------------------------------------------
    if verbose:
        print("\n[Step 1 & 2] Loading audio, normalizing, and pre-processing...")
    proc_signal, raw_signal, fs = load_and_preprocess_audio(
        input_wav,
        target_fs=16000,
        apply_preemph=pre_emphasis,
        preemph_coeff=0.97
    )
    duration = len(raw_signal) / fs
    if verbose:
        print(f"  -> Samples: {len(raw_signal):,} | Sample Rate: {fs} Hz | Duration: {duration:.3f} s")

    # -------------------------------------------------------------
    # Step 3 & 4: Frame Audio and Calculate LPC Coefficients
    # -------------------------------------------------------------
    if verbose:
        print("\n[Step 3 & 4] Framing and estimating LPC filter coefficients (Levinson-Durbin)...")
    lpc_results = extract_lpc_coefficients(
        proc_signal,
        fs=fs,
        order=order,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        window_type="hamming"
    )
    lpc_coeffs = lpc_results['lpc_coeffs']
    num_frames = lpc_coeffs.shape[0]
    if verbose:
        print(f"  -> Processed {num_frames} frames. LPC Coefficients matrix shape: {lpc_coeffs.shape}")

    # -------------------------------------------------------------
    # Step 5 & 6: Inverse Filtering to Calculate LPC Residual
    # -------------------------------------------------------------
    if verbose:
        print("\n[Step 5 & 6] Applying inverse filtering to extract LPC residual e[n]...")
    residual = compute_lpc_residual(proc_signal, lpc_results, method="sample_varying")
    assert len(residual) == len(proc_signal), "Residual length must match original signal!"
    if verbose:
        print(f"  -> Residual extracted successfully! Length: {len(residual):,} samples")

    # -------------------------------------------------------------
    # Step 7: LPC Resynthesis (Reconstruction)
    # -------------------------------------------------------------
    if verbose:
        print("\n[Step 7] Resynthesizing speech from residual and 1/A(z) synthesis filter...")
    resynth_audio = lpc_resynthesize_residual(residual, lpc_results, de_emph=True)
    min_len = min(len(raw_signal), len(resynth_audio))
    sig_power = np.mean(raw_signal[:min_len] ** 2)
    err_power = np.mean((raw_signal[:min_len] - resynth_audio[:min_len]) ** 2) + 1e-12
    snr_db = 10.0 * np.log10(sig_power / err_power)
    if verbose:
        print(f"  -> Resynthesis complete! Reconstruction SNR: {snr_db:.2f} dB")

    # -------------------------------------------------------------
    # Step 8: Feature Extraction (Module 2 & Advanced LPC)
    # -------------------------------------------------------------
    if verbose:
        print("\n[Step 8] Extracting formants, LPCC, LSF, and residual dynamics...")
    features = extract_lpc_features(proc_signal, residual, lpc_results)
    
    mid_frame_idx = num_frames // 2
    mid_formants, mid_bws = extract_formants_from_lpc(lpc_coeffs[mid_frame_idx], fs=fs)
    formants_str = ", ".join([f"F{i+1}={int(f)}Hz" for i, f in enumerate(mid_formants[:4])])
    
    # LPCC and LSF for representative mid-frame
    mid_lpcc = lpc_to_lpcc(lpc_coeffs[mid_frame_idx], num_cep=12)
    mid_lsf = lpc_to_lsf(lpc_coeffs[mid_frame_idx])
    
    if verbose:
        print(f"  -> Mid-frame ({lpc_results['frame_times'][mid_frame_idx]:.2f}s) Formants: {formants_str}")
        print(f"  -> Mean Prediction Gain: {features['mean_prediction_gain_db']:.2f} dB")
        print(f"  -> Residual RMS Energy:  {features['mean_residual_rms']:.4f}")

    # -------------------------------------------------------------
    # Step 9: Module 3 Phase & Glottal Timing (if full_pipeline)
    # -------------------------------------------------------------
    timing_data = {}
    if full_pipeline:
        if verbose:
            print("\n[Step 9 - Module 3] Running Phase & Glottal Timing Analysis (ZFR Epochs, Hilbert, Group Delay)...")
        pt_dir = os.path.join(output_dir, "phase_timing")
        os.makedirs(pt_dir, exist_ok=True)
        
        gcis, gci_times, zfr = extract_epochs_zfr(residual, fs=fs)
        hilbert_data = compute_hilbert_phase_features(residual, fs=fs)
        timing_metrics = compute_glottal_timing_metrics(residual, fs=fs, gcis=gcis)
        timing_metrics['phase_coherence'] = hilbert_data['phase_coherence']
        timing_data = timing_metrics
        
        np.save(os.path.join(pt_dir, "gci_indices.npy"), gcis)
        np.save(os.path.join(pt_dir, "zfr_signal.npy"), zfr)
        with open(os.path.join(pt_dir, "glottal_timing_metrics.json"), "w") as f:
            json.dump(timing_metrics, f, indent=2)
            
        if verbose:
            print(f"  -> Epochs detected:     {len(gcis)}")
            print(f"  -> Mean F0 / Jitter:    {timing_metrics['mean_f0_hz']} Hz | Jitter: {timing_metrics['jitter_local_pct']}% | Shimmer: {timing_metrics['shimmer_local_pct']}%")

    # -------------------------------------------------------------
    # Step 10: Module 4 Classification (if full_pipeline)
    # -------------------------------------------------------------
    classification_data = {}
    if full_pipeline:
        if verbose:
            print("\n[Step 10 - Module 4] Running Minimal Pair Acoustic Classification...")
        clf = OralNasalClassifier()
        clf_result = clf.predict(input_wav)
        classification_data = {
            "predicted_class": clf_result['predicted_class'],
            "confidence": clf_result['confidence'],
            "probabilities": clf_result['probabilities'],
            "acoustic_explanation": clf_result['acoustic_explanation']
        }
        if verbose:
            print(f"  -> Predicted: {clf_result['predicted_class']} (Confidence: {clf_result['confidence']*100:.1f}%)")

    # -------------------------------------------------------------
    # Step 11: Save Outputs & Generate Visualizations
    # -------------------------------------------------------------
    if verbose:
        print("\n[Step 11] Saving deliverables and rendering publication-grade plots...")
        
    coeff_file = os.path.join(output_dir, "lpc_coefficients.npy")
    np.save(coeff_file, lpc_coeffs)
    
    res_npy_file = os.path.join(output_dir, "lpc_residual.npy")
    np.save(res_npy_file, residual)
    
    res_wav_file = os.path.join(output_dir, "lpc_residual.wav")
    norm_residual = residual / (np.max(np.abs(residual)) + 1e-9) * 0.90
    save_wav_file(res_wav_file, norm_residual, fs=fs)
    
    resynth_wav_file = os.path.join(output_dir, "lpc_resynthesized.wav")
    save_wav_file(resynth_wav_file, resynth_audio, fs=fs)
    
    # Save JSON summary report
    summary = {
        "file_name": os.path.basename(input_wav),
        "fs": fs,
        "duration_seconds": round(duration, 4),
        "total_samples": len(raw_signal),
        "lpc_order": order,
        "frame_ms": frame_ms,
        "hop_ms": hop_ms,
        "num_frames": num_frames,
        "coefficients_shape": list(lpc_coeffs.shape),
        "reconstruction_snr_db": round(float(snr_db), 2),
        "mean_prediction_gain_db": round(features["mean_prediction_gain_db"], 2),
        "mean_residual_rms": round(features["mean_residual_rms"], 4),
        "residual_spectral_flatness": round(features["residual_spectral_flatness"], 4),
        "speech_spectral_flatness": round(features["speech_spectral_flatness"], 4),
        "representative_formants_hz": [round(f, 1) for f in mid_formants[:4]],
        "mid_frame_lpcc": [round(float(c), 4) for c in mid_lpcc],
        "mid_frame_lsf_rad": [round(float(w), 4) for w in mid_lsf],
        "glottal_timing": timing_data,
        "classification": classification_data,
        "outputs": {
            "lpc_coefficients_npy": coeff_file,
            "lpc_residual_npy": res_npy_file,
            "lpc_residual_wav": res_wav_file,
            "lpc_resynthesized_wav": resynth_wav_file
        }
    }
    json_path = os.path.join(output_dir, "lpc_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    # Visualizations
    p1 = os.path.join(plots_dir, "waveform_and_residual.png")
    f1 = plot_waveform_and_residual(raw_signal, residual, fs, title=f"'{base_name}'", save_path=p1)
    plt.close(f1)
    
    p2 = os.path.join(plots_dir, "spectrogram_comparison.png")
    f2 = plot_spectrogram_comparison(raw_signal, residual, fs, title=f"'{base_name}' Spectrogram", save_path=p2)
    plt.close(f2)
    
    mid_start = lpc_results['indices'][mid_frame_idx]
    mid_frame_samples = proc_signal[mid_start : mid_start + lpc_results['frame_len']]
    p3 = os.path.join(plots_dir, "lpc_spectral_envelope.png")
    f3 = plot_lpc_spectral_envelope(
        mid_frame_samples,
        lpc_coeffs[mid_frame_idx],
        fs,
        frame_time=lpc_results['frame_times'][mid_frame_idx],
        save_path=p3
    )
    plt.close(f3)
    
    p4 = os.path.join(plots_dir, "zplane_poles.png")
    f4 = plot_zplane_poles(lpc_coeffs[mid_frame_idx], fs=fs, title=f"'{base_name}' Z-Plane Poles", save_path=p4)
    plt.close(f4)

    if verbose:
        print(f"  [+] Saved LPC Coefficients:   {coeff_file}")
        print(f"  [+] Saved LPC Residual:       {res_npy_file}")
        print(f"  [+] Saved Residual WAV:       {res_wav_file}")
        print(f"  [+] Saved Resynthesized WAV:  {resynth_wav_file}")
        print(f"  [+] Saved Summary Report:     {json_path}")
        print(f"  [+] Saved Publication Plots:  {plots_dir}")
        print("\n" + "=" * 70)
        print("  SUCCESS: Full Pipeline Complete! All deliverables ready.")
        print("=" * 70 + "\n")

    return {
        "fs": fs,
        "speech": raw_signal,
        "residual": residual,
        "resynthesized": resynth_audio,
        "lpc_coeffs": lpc_coeffs,
        "feat": features,
        "summary": summary
    }

def main():
    parser = argparse.ArgumentParser(description="LPC Feature & Residual Extractor for Speech Processing")
    parser.add_argument("--input", "-i", type=str, default="data/kaha.wav", help="Input WAV file path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--order", "-p", type=int, default=16, help="LPC order (default 16)")
    parser.add_argument("--frame_ms", type=float, default=25.0, help="Frame duration in ms (default 25)")
    parser.add_argument("--hop_ms", type=float, default=10.0, help="Hop duration in ms (default 10)")
    parser.add_argument("--no_preemph", action="store_true", help="Disable pre-emphasis filter")
    parser.add_argument("--no_pipeline", action="store_true", help="Disable full modular pipeline")
    
    args = parser.parse_args()
    process_file(
        args.input,
        output_dir=args.output,
        order=args.order,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        pre_emphasis=not args.no_preemph,
        full_pipeline=not args.no_pipeline,
        verbose=True
    )

if __name__ == "__main__":
    main()
