"""
run_lpc.py
----------
Main execution script for Member 2 (LPC Analysis Module).
Processes an input speech WAV file, performs LPC coefficient estimation,
applies inverse filtering to extract the LPC residual, and saves all outputs:
  1. lpc_coefficients.npy
  2. lpc_residual.npy
  3. lpc_residual.wav
  4. Visual analysis graphs (plots/)
  5. Comprehensive analysis report (lpc_summary.json)
"""

import os
import sys
import json
import argparse
import numpy as np

from lpc_core.preprocessor import load_and_preprocess_audio, save_wav_file
from lpc_core.lpc_analyzer import extract_lpc_coefficients
from lpc_core.inverse_filter import compute_lpc_residual
from lpc_core.formant_analyzer import extract_formants_from_lpc
from lpc_core.features import extract_lpc_features
from lpc_core.visualizer import (
    plot_waveform_and_residual,
    plot_spectrogram_comparison,
    plot_lpc_spectral_envelope
)

def process_file(
    input_wav,
    output_dir=None,
    order=16,
    frame_ms=25.0,
    hop_ms=10.0,
    pre_emphasis=True,
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
        print("=" * 65)
        print(f"  LINEAR PREDICTIVE CODING (LPC) PIPELINE - MEMBER 2")
        print(f"  Input File:     {input_wav}")
        print(f"  LPC Order (p):  {order}")
        print(f"  Frame Length:   {frame_ms} ms")
        print(f"  Hop Length:     {hop_ms} ms")
        print(f"  Output Folder:  {output_dir}")
        print("=" * 65)

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
        print(f"  -> Leading coefficient for each frame: 1.0 (filter polynomial A(z))")

    # -------------------------------------------------------------
    # Step 5 & 6: Inverse Filtering to Calculate LPC Residual
    # -------------------------------------------------------------
    if verbose:
        print("\n[Step 5 & 6] Applying inverse filtering to extract LPC residual e[n]...")
    residual = compute_lpc_residual(proc_signal, lpc_results, method="sample_varying")
    
    # Check length alignment
    assert len(residual) == len(proc_signal), "Residual length must match original signal!"
    if verbose:
        print(f"  -> Residual extracted successfully! Length: {len(residual):,} samples")

    # -------------------------------------------------------------
    # Step 7: Feature Extraction for Member 3 & 4
    # -------------------------------------------------------------
    if verbose:
        print("\n[Step 7] Extracting formants, prediction gain, and residual features...")
    features = extract_lpc_features(proc_signal, residual, lpc_results)
    
    # Extract representative formants from a mid-utterance voiced frame
    mid_frame_idx = num_frames // 2
    mid_formants, mid_bws = extract_formants_from_lpc(lpc_coeffs[mid_frame_idx], fs=fs)
    formants_str = ", ".join([f"F{i+1}={int(f)}Hz" for i, f in enumerate(mid_formants[:4])])
    if verbose:
        print(f"  -> Mid-frame ({lpc_results['frame_times'][mid_frame_idx]:.2f}s) Formants: {formants_str}")
        print(f"  -> Mean Prediction Gain: {features['mean_prediction_gain_db']:.2f} dB")
        print(f"  -> Residual RMS Energy:  {features['mean_residual_rms']:.4f}")

    # -------------------------------------------------------------
    # Step 8: Save Outputs (NPY, WAV, JSON, PLOTS)
    # -------------------------------------------------------------
    if verbose:
        print("\n[Step 8] Saving project outputs for Member 3 & Member 4...")
        
    coeff_file = os.path.join(output_dir, "lpc_coefficients.npy")
    np.save(coeff_file, lpc_coeffs)
    
    res_npy_file = os.path.join(output_dir, "lpc_residual.npy")
    np.save(res_npy_file, residual)
    
    res_wav_file = os.path.join(output_dir, "lpc_residual.wav")
    # Normalize residual audio to safe range for listening
    norm_residual = residual / (np.max(np.abs(residual)) + 1e-9) * 0.90
    save_wav_file(res_wav_file, norm_residual, fs=fs)
    
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
        "mean_prediction_gain_db": round(features["mean_prediction_gain_db"], 2),
        "mean_residual_rms": round(features["mean_residual_rms"], 4),
        "residual_spectral_flatness": round(features["residual_spectral_flatness"], 4),
        "speech_spectral_flatness": round(features["speech_spectral_flatness"], 4),
        "representative_formants_hz": [round(f, 1) for f in mid_formants[:4]],
        "outputs": {
            "lpc_coefficients_npy": coeff_file,
            "lpc_residual_npy": res_npy_file,
            "lpc_residual_wav": res_wav_file
        }
    }
    json_path = os.path.join(output_dir, "lpc_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    # Generate and save plots
    p1 = os.path.join(plots_dir, "waveform_and_residual.png")
    plot_waveform_and_residual(raw_signal, residual, fs, title=f"'{base_name}'", save_path=p1)
    
    p2 = os.path.join(plots_dir, "spectrogram_comparison.png")
    plot_spectrogram_comparison(raw_signal, residual, fs, title=f"'{base_name}' Spectrogram", save_path=p2)
    
    mid_start = lpc_results['indices'][mid_frame_idx]
    mid_frame_samples = proc_signal[mid_start : mid_start + lpc_results['frame_len']]
    p3 = os.path.join(plots_dir, "lpc_spectral_envelope.png")
    plot_lpc_spectral_envelope(
        mid_frame_samples,
        lpc_coeffs[mid_frame_idx],
        fs,
        frame_time=lpc_results['frame_times'][mid_frame_idx],
        save_path=p3
    )

    if verbose:
        print(f"  [+] Saved: {coeff_file}")
        print(f"  [+] Saved: {res_npy_file}")
        print(f"  [+] Saved: {res_wav_file}")
        print(f"  [+] Saved: {json_path}")
        print(f"  [+] Plots generated in: {plots_dir}")
        print("\n" + "=" * 65)
        print("  SUCCESS: LPC Part Completed! All artifacts ready for Member 3.")
        print("=" * 65 + "\n")

    return {
        "fs": fs,
        "speech": raw_signal,
        "residual": residual,
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
    
    args = parser.parse_args()
    process_file(
        args.input,
        output_dir=args.output,
        order=args.order,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        pre_emphasis=not args.no_preemph,
        verbose=True
    )

if __name__ == "__main__":
    main()
