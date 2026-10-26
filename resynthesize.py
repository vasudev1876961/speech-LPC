"""
resynthesize.py
---------------
CLI Tool for LPC Speech Resynthesis & Vocoding.
Reconstructs speech from LPC all-pole filter coefficients and excitation:
1. Residual Resynthesis: High-fidelity reconstruction using e[n]
2. Parametric LPC Vocoder: Reconstructed using pitch pulses + white noise
3. Voice Morphing: Pitch scaling, vocal tract formant warping, whisperization
"""

import os
import argparse
import numpy as np

from lpc_core.preprocessor import load_and_preprocess_audio, save_wav_file
from lpc_core.lpc_analyzer import extract_lpc_coefficients
from lpc_core.inverse_filter import compute_lpc_residual
from lpc_core.features import estimate_pitch_from_residual
from lpc_core.resynthesis import (
    lpc_resynthesize_residual,
    lpc_resynthesize_vocoder,
    modify_speech
)

def main():
    parser = argparse.ArgumentParser(description="LPC Speech Resynthesis & Voice Morphing Studio")
    parser.add_argument("--input", "-i", type=str, default="data/kaha.wav", help="Path to input audio file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output reconstructed WAV file")
    parser.add_argument("--mode", "-m", type=str, choices=["residual", "vocoder", "morph"], default="residual",
                        help="Resynthesis mode: 'residual' (exact e[n]), 'vocoder' (pulse/noise), or 'morph'")
    parser.add_argument("--pitch_scale", type=float, default=1.0, help="Pitch shift factor (e.g. 1.25 for higher, 0.8 for lower)")
    parser.add_argument("--formant_scale", type=float, default=1.0, help="Formant warp factor (e.g. 1.15 for shorter tract/female, 0.85 for deep voice)")
    parser.add_argument("--whisper", action="store_true", help="Whisperize the speech (turbulent noise excitation)")
    parser.add_argument("--order", "-p", type=int, default=16, help="LPC order (default: 16)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
        
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    if args.output is None:
        out_dir = os.path.join("outputs", base_name, "resynthesis")
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_{args.mode}"
        if args.whisper:
            suffix += "_whisper"
        if args.pitch_scale != 1.0:
            suffix += f"_pitch{args.pitch_scale:.2f}"
        if args.formant_scale != 1.0:
            suffix += f"_formant{args.formant_scale:.2f}"
        args.output = os.path.join(out_dir, f"{base_name}{suffix}.wav")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
    print("\n" + "="*65)
    print("  LPC RESYNTHESIS & VOCODER ENGINE")
    print(f"  Input:         {args.input}")
    print(f"  Mode:          {args.mode}")
    print(f"  LPC Order:     {args.order}")
    print(f"  Pitch Scale:   {args.pitch_scale}x")
    print(f"  Formant Scale: {args.formant_scale}x")
    print(f"  Whisperize:    {args.whisper}")
    print(f"  Output:        {args.output}")
    print("="*65)
    
    proc_sig, raw_sig, fs = load_and_preprocess_audio(args.input, target_fs=16000, apply_preemph=True)
    lpc_res = extract_lpc_coefficients(proc_sig, fs=fs, order=args.order, frame_ms=25.0, hop_ms=10.0)
    
    if args.mode == "residual":
        print("[+] Extracting LPC residual e[n] via inverse filtering...")
        residual = compute_lpc_residual(proc_sig, lpc_res, method="sample_varying")
        print("[+] Resynthesizing speech through 1 / A(z) all-pole filter...")
        synth = lpc_resynthesize_residual(residual, lpc_res, de_emph=True)
        
    elif args.mode == "vocoder":
        print("[+] Estimating pitch contour F0 from residual...")
        residual = compute_lpc_residual(proc_sig, lpc_res, method="sample_varying")
        f0_contour, times = estimate_pitch_from_residual(residual, fs=fs)
        print("[+] Synthesizing speech using parametric pulse/noise source...")
        synth = lpc_resynthesize_vocoder(f0_contour, times, lpc_res, fs=fs)
        
    elif args.mode == "morph":
        print("[+] Performing voice morphing transformation...")
        synth = modify_speech(
            proc_sig,
            lpc_res,
            pitch_scale=args.pitch_scale,
            formant_scale=args.formant_scale,
            whisper=args.whisper,
            fs=fs
        )
        
    save_wav_file(args.output, synth, fs=fs)
    print(f"\n[OK] Resynthesized audio saved to: {args.output}")
    
    # Calculate SNR between original and resynthesized (for residual mode)
    if args.mode == "residual":
        min_l = min(len(raw_sig), len(synth))
        sig_pow = np.mean(raw_sig[:min_l] ** 2)
        err_pow = np.mean((raw_sig[:min_l] - synth[:min_l]) ** 2) + 1e-12
        snr_db = 10.0 * np.log10(sig_pow / err_pow)
        print(f"  -> Reconstruction Signal-to-Noise Ratio (SNR): {snr_db:.2f} dB")

if __name__ == "__main__":
    main()
