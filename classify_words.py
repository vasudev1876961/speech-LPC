"""
classify_words.py
------------------
CLI Tool for Module 4: Minimal Pair Acoustic Classification & DTW Studio.
Provides:
1. Classification of any input WAV into Oral vs Nasalized vowel (/a/ vs /ã/)
2. Detailed acoustic phonetics diagnostic report explaining the prediction
3. Dynamic Time Warping (DTW) trajectory alignment between minimal pairs
4. Benchmark evaluation across the full dataset corpus
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from lpc_core.classifier import (
    OralNasalClassifier,
    extract_acoustic_features,
    compute_dtw,
    run_corpus_classification_benchmark
)
from lpc_core.visualizer import plot_dtw_alignment

def main():
    parser = argparse.ArgumentParser(description="Module 4: Acoustic Minimal Pair Classifier & DTW Studio")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input WAV file to classify")
    parser.add_argument("--compare", "-c", nargs=2, default=None, metavar=("FILE1", "FILE2"),
                        help="Compare two utterances using Dynamic Time Warping (DTW)")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Run classification evaluation benchmark across all corpus files")
    parser.add_argument("--out_dir", "-o", type=str, default="outputs/classification", help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # Option 1: Benchmark across entire corpus
    # -------------------------------------------------------------
    if args.benchmark or (args.input is None and args.compare is None):
        print("\n" + "="*75)
        print("  MODULE 4: CORPUS MINIMAL PAIR CLASSIFICATION BENCHMARK")
        print("="*75)
        
        results = run_corpus_classification_benchmark(data_dir="data")
        
        print(f"\n{'File Name':<20} | {'Predicted Class':<22} | {'Conf':<7} | {'Ground Truth':<12} | {'Status'}")
        print("-" * 75)
        
        correct_count = 0
        total_eval = 0
        for r in results:
            pred = r['predicted_class']
            conf = f"{r['confidence']*100:.1f}%"
            gt = r['ground_truth']
            status = "N/A"
            if r['correct'] is not None:
                total_eval += 1
                if r['correct']:
                    correct_count += 1
                    status = "CORRECT"
                else:
                    status = "MISMATCH"
            print(f"{r['audio_file']:<20} | {pred:<22} | {conf:<7} | {gt:<12} | {status}")
            
        print("-" * 75)
        if total_eval > 0:
            acc = (correct_count / total_eval) * 100.0
            print(f"Overall Minimal Pair Discrimination Accuracy: {acc:.1f}% ({correct_count}/{total_eval})\n")
            
        benchmark_json = os.path.join(args.out_dir, "corpus_classification_benchmark.json")
        with open(benchmark_json, "w") as f:
            clean_results = [{k: v for k, v in r.items() if k != "feat_dict"} for r in results]
            json.dump(clean_results, f, indent=2)
        print(f"[OK] Full benchmark report saved: {benchmark_json}")
        if args.input is None and args.compare is None:
            return
            
    # -------------------------------------------------------------
    # Option 2: Classify single audio file
    # -------------------------------------------------------------
    if args.input:
        print("\n" + "="*70)
        print(f"  MODULE 4: ACOUSTIC PHONETICS CLASSIFICATION")
        print(f"  File: {args.input}")
        print("="*70)
        
        clf = OralNasalClassifier()
        res = clf.predict(args.input)
        
        print(f"\n>>> PREDICTION: {res['predicted_class']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"    Oral Probability:      {res['probabilities']['Oral']*100:.1f}%")
        print(f"    Nasalized Probability: {res['probabilities']['Nasalized']*100:.1f}%\n")
        
        print("Key Acoustic Phonetics Measurements:")
        print(f"  - Formant F1 Center Frequency:   {res['key_metrics']['F1_frequency_hz']} Hz")
        print(f"  - Formant F2 Center Frequency:   {res['key_metrics']['F2_frequency_hz']} Hz")
        print(f"  - Formant F1 Bandwidth:          {res['key_metrics']['F1_bandwidth_hz']} Hz (Broadened in nasal vowels)")
        print(f"  - Low-Frequency Nasal Pole Ratio:{res['key_metrics']['nasal_energy_ratio']:.3f} (Elevated in nasal vowels)")
        print(f"  - Anti-Resonance Zero Depth:     {res['key_metrics']['anti_resonance_depth']:.2f}")
        print(f"  - Prediction Gain:               {res['key_metrics']['prediction_gain_db']:.2f} dB")
        print(f"  - Residual RMS Energy:           {res['key_metrics']['residual_rms']:.4f}\n")
        
        print("Acoustic Phonetics Rationale:")
        for reason in res['acoustic_explanation']:
            print(f"  * {reason}")
            
        # Save output JSON
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        out_json = os.path.join(args.out_dir, f"classification_{base_name}.json")
        save_dict = {k: v for k, v in res.items() if k != "feat_dict"}
        with open(out_json, "w") as f:
            json.dump(save_dict, f, indent=2)
        print(f"\n[OK] Classification report saved: {out_json}")
        
    # -------------------------------------------------------------
    # Option 3: Compare two utterances using DTW
    # -------------------------------------------------------------
    if args.compare:
        file1, file2 = args.compare
        name1 = os.path.splitext(os.path.basename(file1))[0]
        name2 = os.path.splitext(os.path.basename(file2))[0]
        
        print("\n" + "="*70)
        print(f"  DYNAMIC TIME WARPING (DTW) ACOUSTIC ALIGNMENT")
        print(f"  Word 1: {name1} ({file1})")
        print(f"  Word 2: {name2} ({file2})")
        print("="*70)
        
        f1 = extract_acoustic_features(file1)
        f2 = extract_acoustic_features(file2)
        
        # DTW on LPCC trajectories
        dist, cost_mat, path = compute_dtw(f1['lpcc_matrix'], f2['lpcc_matrix'], metric="euclidean")
        print(f"\n[+] Normalized DTW Alignment Distance: {dist:.4f}")
        print(f"  -> Path steps: {len(path)} frames aligned.")
        
        # Plot DTW alignment
        dtw_plot_path = os.path.join(args.out_dir, f"dtw_alignment_{name1}_vs_{name2}.png")
        fig = plot_dtw_alignment(cost_mat, path, name1=name1, name2=name2, dist=dist, save_path=dtw_plot_path)
        plt.close(fig)
        print(f"[OK] DTW alignment heatmap saved: {dtw_plot_path}")

if __name__ == "__main__":
    main()
