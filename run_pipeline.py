"""
run_pipeline.py
---------------
Master Pipeline Orchestrator for the Complete LPC Speech Analysis Suite.
Executes the end-to-end multi-module workflow across all corpus files:
  1. Module 1: Preprocessing & Audio Conditioning
  2. Module 2: LPC Estimation, Inverse Filtering, Formants, LPCC, LSF, Resynthesis
  3. Module 3: Glottal Phase & Timing (ZFR Epochs, Hilbert, Group Delay, Jitter/Shimmer)
  4. Module 4: Dynamic Time Warping (DTW) & Minimal Pair Classification
  5. Multi-Voice Comparisons across minimal pairs, genders, and vowels
"""

import os
import sys
import json
import time
import numpy as np

from run_lpc import process_file
from compare_words import compare_pair
from phase_timing_analysis import run_phase_timing_analysis
from lpc_core.classifier import run_corpus_classification_benchmark

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

COMPARISON_PAIRS = [
    ("data/kaha.wav", "data/kahaan.wav", "kaha", "kahaan"),
    ("data/baas.wav", "data/baans.wav", "baas", "baans"),
    ("data/saas.wav", "data/saans.wav", "saas", "saans"),
    ("data/kaha.wav", "data/kaha_female.wav", "kaha_male", "kaha_female"),
    ("data/hello.wav", "data/hello_female.wav", "hello_male", "hello_female"),
    ("data/vowel_i.wav", "data/vowel_u.wav", "vowel_i", "vowel_u"),
    ("data/kaha.wav", "data/whisper_kaha.wav", "kaha_voiced", "kaha_whisper")
]

def run_full_suite():
    start_time = time.time()
    print("=" * 80)
    print("  [>>>] EXECUTING COMPLETE END-TO-END LPC SPEECH ANALYSIS SUITE")
    print("=" * 80)
    
    # 1. Verify data files exist
    data_dir = "data"
    wav_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
    print(f"\n[Phase 1] Found {len(wav_files)} audio utterances in '{data_dir}/':")
    for f in sorted(wav_files):
        print(f"  - {f}")
        
    # 2. Process all audio files through Module 2 & 3 pipeline
    print("\n" + "=" * 80)
    print("  [Phase 2] Running LPC Analysis, Resynthesis, & Phase-Timing across all files")
    print("=" * 80)
    processed_summaries = {}
    
    for f in sorted(wav_files):
        in_path = os.path.join(data_dir, f)
        base = os.path.splitext(f)[0]
        print(f"\n>>> Processing: {f} ...")
        res = process_file(in_path, full_pipeline=True, verbose=False)
        processed_summaries[base] = res['summary']
        print(f"    [OK] LPC Order: {res['summary']['lpc_order']} | SNR: {res['summary']['reconstruction_snr_db']} dB | Pred Gain: {res['summary']['mean_prediction_gain_db']} dB")
        
    # 3. Run all benchmark comparisons
    print("\n" + "=" * 80)
    print("  [Phase 3] Running Multi-Voice & Minimal Pair Comparative Analyses")
    print("=" * 80)
    comparison_reports = {}
    
    for f1, f2, n1, n2 in COMPARISON_PAIRS:
        if os.path.exists(f1) and os.path.exists(f2):
            print(f"\n>>> Comparing '{n1}' vs '{n2}' ...")
            rep = compare_pair(f1, f2, name1=n1, name2=n2, out_dir="outputs/comparison")
            comparison_reports[f"{n1}_vs_{n2}"] = rep
            print(f"    [OK] LPC Cosine Dist: {rep['mean_lpc_cosine_distance']:.4f} | Res RMS Ratio: {rep['residual_rms_ratio']:.2f}")
            
    # 4. Run Module 4 Minimal Pair Classification Benchmark
    print("\n" + "=" * 80)
    print("  [Phase 4] Evaluating Module 4 AI Minimal Pair Classifier")
    print("=" * 80)
    clf_results = run_corpus_classification_benchmark(data_dir=data_dir)
    
    correct = sum(1 for r in clf_results if r.get('correct') is True)
    evaluated = sum(1 for r in clf_results if r.get('correct') is not None)
    accuracy = (correct / evaluated * 100.0) if evaluated > 0 else 0.0
    print(f"\n>>> Classifier Benchmark Result: {accuracy:.1f}% Accuracy ({correct}/{evaluated} minimal pairs matched)")
    
    # 5. Build Master Summary Report
    total_time = time.time() - start_time
    summary_report_path = "outputs/PIPELINE_RUN_SUMMARY.md"
    
    report_md = f"""# 🎙️ LPC Suite: Complete Pipeline Run Summary

**Execution Timestamp:** {time.strftime("%Y-%m-%d %H:%M:%S")}  
**Total Runtime:** {total_time:.2f} seconds  
**Total Audio Utterances Processed:** {len(wav_files)}  
**Minimal Pair Classification Accuracy:** {accuracy:.1f}% ({correct}/{evaluated})  

---

## 1. Processed Corpus Deliverables

| Audio File | Duration (s) | LPC Order | Reconstruction SNR | Prediction Gain | Residual RMS | Epochs | Prediction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for base, sm in sorted(processed_summaries.items()):
        dur = sm['duration_seconds']
        p = sm['lpc_order']
        snr = sm['reconstruction_snr_db']
        gain = sm['mean_prediction_gain_db']
        rms = sm['mean_residual_rms']
        epochs = sm.get('glottal_timing', {}).get('num_epochs', 'N/A')
        pred = sm.get('classification', {}).get('predicted_class', 'N/A')
        report_md += f"| `{sm['file_name']}` | {dur:.2f}s | p={p} | {snr:.1f} dB | {gain:.2f} dB | {rms:.4f} | {epochs} | {pred} |\n"
        
    report_md += """
---

## 2. Minimal Pair & Speaker Voice Comparisons

| Comparison Pair | LPC Cosine Dist | Residual Energy Ratio | Formant F1 Shift | Acoustic Contrast |
| :--- | :--- | :--- | :--- | :--- |
"""
    for pair_name, rep in comparison_reports.items():
        cos_d = rep['mean_lpc_cosine_distance']
        ratio = rep['residual_rms_ratio']
        f1_1 = rep['word1_vowel_formants_hz'][0] if rep['word1_vowel_formants_hz'] else 0
        f1_2 = rep['word2_vowel_formants_hz'][0] if rep['word2_vowel_formants_hz'] else 0
        report_md += f"| `{pair_name}` | {cos_d:.4f} | {ratio:.2f} | {int(f1_1)} Hz → {int(f1_2)} Hz | {rep['acoustic_interpretation'][:75]}... |\n"
        
    report_md += f"""
---

## 3. Subsystem Deliverables & Production Artifacts Overview

1. **Acoustic Audio Corpus Subsystem**: 14 mono 16 kHz standardized recordings across 3 minimal pairs and speaker contrasts.
2. **Module 1 (LPC Modeling & Vocoder Subsystem)**: `lpc_coefficients.npy`, `lpc_residual.npy`, `lpc_residual.wav`, and `lpc_resynthesized.wav`.
3. **Module 2 (Glottal Phase & Temporal Dynamics Subsystem)**: Zero Frequency Resonator (ZFR) Epochs `gci_indices.npy`, Hilbert envelope, Group delay spectrum, Jitter ({np.mean([sm.get('glottal_timing',{}).get('jitter_local_pct',0) for sm in processed_summaries.values()]):.2f}%), Shimmer.
4. **Module 3 (Pattern Recognition & DTW Alignment Subsystem)**: Acoustic phonetic classifier achieving **{accuracy:.1f}% accuracy** on Hindi oral vs nasal minimal pairs, DTW alignment paths.
"""

    with open(summary_report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"\n[SUCCESS] Master pipeline run summary generated at: {summary_report_path}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_full_suite()
