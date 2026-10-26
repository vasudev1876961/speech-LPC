"""
phase_timing_analysis.py
-------------------------
CLI Tool for Module 3: Phase & Glottal Timing Analysis Suite.
Processes the LPC residual to extract:
1. Glottal Closure Instants (GCIs / Epochs) using Zero Frequency Resonator (ZFR)
2. Hilbert Transform Instantaneous Amplitude, Phase, and Instantaneous Frequency
3. Modified Group Delay (MGD) Spectrum (resolving nasal zeros & formants)
4. Jitter (local, RAP), Shimmer (local, APQ3), and Harmonic-to-Noise Ratio (HNR)
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from lpc_core.preprocessor import load_and_preprocess_audio
from lpc_core.lpc_analyzer import extract_lpc_coefficients
from lpc_core.inverse_filter import compute_lpc_residual
from lpc_core.phase_timing import (
    extract_epochs_zfr,
    compute_hilbert_phase_features,
    compute_group_delay_spectrum,
    compute_glottal_timing_metrics
)
from lpc_core.visualizer import plot_phase_timing, plot_group_delay

def run_phase_timing_analysis(audio_path, output_dir=None, order=16, verbose=True):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")
        
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    if output_dir is None:
        output_dir = os.path.join("outputs", base_name)
        
    pt_dir = os.path.join(output_dir, "phase_timing")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    if verbose:
        print("\n" + "="*70)
        print(f"  MODULE 3: PHASE & GLOTTAL TIMING ANALYSIS")
        print(f"  Input:         {audio_path}")
        print(f"  Output Folder: {pt_dir}")
        print("="*70)
        
    # 1. Preprocess & LPC Residual
    proc_sig, raw_sig, fs = load_and_preprocess_audio(audio_path, target_fs=16000, apply_preemph=True)
    lpc_res = extract_lpc_coefficients(proc_sig, fs=fs, order=order, frame_ms=25.0, hop_ms=10.0)
    residual = compute_lpc_residual(proc_sig, lpc_res, method="sample_varying")
    
    # 2. Zero Frequency Resonator (ZFR) Epoch / GCI Extraction
    if verbose:
        print("\n[+] Extracting Glottal Closure Instants (GCIs) via Zero Frequency Resonator (ZFR)...")
    gcis, gci_times, zfr = extract_epochs_zfr(residual, fs=fs)
    if verbose:
        print(f"  -> Detected {len(gcis)} Glottal Closure Epochs across the utterance.")
        
    # 3. Hilbert Transform Instantaneous Phase & Frequency
    if verbose:
        print("[+] Computing Hilbert Transform Instantaneous Phase & Amplitude Envelope...")
    hilbert_data = compute_hilbert_phase_features(residual, fs=fs)
    if verbose:
        print(f"  -> Mean Phase Coherence Index: {hilbert_data['phase_coherence']:.4f}")
        
    # 4. Glottal Timing & Perturbation Metrics (Jitter, Shimmer, HNR)
    if verbose:
        print("[+] Calculating Glottal Timing & Perturbation Metrics (Jitter, Shimmer, HNR)...")
    timing_metrics = compute_glottal_timing_metrics(residual, fs=fs, gcis=gcis)
    timing_metrics['phase_coherence'] = hilbert_data['phase_coherence']
    
    if verbose:
        print(f"  -> Mean F0:             {timing_metrics['mean_f0_hz']} Hz")
        print(f"  -> Mean Pitch Period:   {timing_metrics['mean_pitch_period_ms']} ms")
        print(f"  -> Local Jitter:        {timing_metrics['jitter_local_pct']}%")
        print(f"  -> Local Shimmer:       {timing_metrics['shimmer_local_pct']}%")
        print(f"  -> Harmonic-to-Noise:   {timing_metrics['hnr_db']} dB")
        
    # 5. Modified Group Delay (MGD) on mid-vowel region
    if verbose:
        print("[+] Computing Modified Group Delay (MGD) Spectrum for acoustic zeros...")
    mid_sample = len(residual) // 2
    vowel_frame = residual[mid_sample : mid_sample + 400]
    freqs, tau, tau_mod = compute_group_delay_spectrum(vowel_frame, fs=fs, n_fft=1024)
    
    # Save artifacts
    np.save(os.path.join(pt_dir, "gci_indices.npy"), gcis)
    np.save(os.path.join(pt_dir, "gci_times.npy"), gci_times)
    np.save(os.path.join(pt_dir, "zfr_signal.npy"), zfr)
    np.save(os.path.join(pt_dir, "hilbert_envelope.npy"), hilbert_data['envelope'])
    
    with open(os.path.join(pt_dir, "glottal_timing_metrics.json"), "w") as f:
        json.dump(timing_metrics, f, indent=2)
        
    # Generate Visualizations
    epoch_fig_path = os.path.join(plots_dir, "epoch_and_zfr_analysis.png")
    fig_epoch = plot_phase_timing(residual, zfr, gcis, fs=fs, title=base_name, save_path=epoch_fig_path)
    plt.close(fig_epoch)
    
    mgd_fig_path = os.path.join(plots_dir, "group_delay_spectrum.png")
    fig_mgd = plot_group_delay(freqs, tau_mod, fs=fs, title=f"Modified Group Delay: {base_name}", save_path=mgd_fig_path)
    plt.close(fig_mgd)
    
    if verbose:
        print(f"\n[OK] Module 3 Deliverables saved to: {pt_dir}")
        print(f"  -> Epoch Analysis Plot:       {epoch_fig_path}")
        print(f"  -> Group Delay Plot:          {mgd_fig_path}")
        print(f"  -> Timing Metrics JSON:       {os.path.join(pt_dir, 'glottal_timing_metrics.json')}")
        
    return {
        "gcis": gcis,
        "gci_times": gci_times,
        "zfr": zfr,
        "hilbert": hilbert_data,
        "timing_metrics": timing_metrics,
        "mgd_freqs": freqs,
        "mgd_tau": tau_mod,
        "plots": [epoch_fig_path, mgd_fig_path]
    }

def main():
    parser = argparse.ArgumentParser(description="Module 3: Phase & Glottal Timing Analysis CLI")
    parser.add_argument("--input", "-i", type=str, default="data/kaha.wav", help="Input WAV file")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--order", "-p", type=int, default=16, help="LPC order (default: 16)")
    args = parser.parse_args()
    
    run_phase_timing_analysis(args.input, output_dir=args.output_dir, order=args.order)

if __name__ == "__main__":
    main()
