# 🎙️ LPC Suite: Complete Pipeline Run Summary

**Execution Timestamp:** 2026-10-30 22:49:28  
**Total Runtime:** 92.16 seconds  
**Total Audio Utterances Processed:** 14  
**Minimal Pair Classification Accuracy:** 100.0% (9/9)  

---

## 1. Processed Corpus Deliverables

| Audio File | Duration (s) | LPC Order | Reconstruction SNR | Prediction Gain | Residual RMS | Epochs | Prediction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baans.wav` | 0.75s | p=16 | 25.6 dB | 16.49 dB | 0.0173 | 1 | Nasalized Vowel (/ã/) |
| `baas.wav` | 0.75s | p=16 | 25.6 dB | 14.88 dB | 0.0149 | 5 | Oral Vowel (/a/) |
| `hello.wav` | 0.70s | p=16 | 25.6 dB | 13.99 dB | 0.0095 | 71 | Oral Vowel (/a/) |
| `hello_female.wav` | 0.70s | p=16 | 25.6 dB | 12.98 dB | 0.0166 | 113 | Oral Vowel (/a/) |
| `kaha.wav` | 0.70s | p=16 | 25.6 dB | 11.62 dB | 0.0158 | 50 | Oral Vowel (/a/) |
| `kaha_deep_male.wav` | 0.70s | p=16 | 25.6 dB | 12.36 dB | 0.0146 | 7 | Oral Vowel (/a/) |
| `kaha_female.wav` | 0.70s | p=16 | 25.6 dB | 10.89 dB | 0.0227 | 90 | Oral Vowel (/a/) |
| `kahaan.wav` | 0.74s | p=16 | 25.6 dB | 12.85 dB | 0.0139 | 59 | Nasalized Vowel (/ã/) |
| `saans.wav` | 0.78s | p=16 | 25.6 dB | 16.48 dB | 0.0204 | 58 | Nasalized Vowel (/ã/) |
| `saas.wav` | 0.78s | p=16 | 25.6 dB | 15.15 dB | 0.0222 | 54 | Oral Vowel (/a/) |
| `vowel_a.wav` | 0.65s | p=16 | 25.6 dB | 13.66 dB | 0.0142 | 77 | Oral Vowel (/a/) |
| `vowel_i.wav` | 0.65s | p=16 | 25.6 dB | 13.91 dB | 0.0094 | 77 | Oral Vowel (/a/) |
| `vowel_u.wav` | 0.65s | p=16 | 25.6 dB | 20.12 dB | 0.0045 | 77 | Oral Vowel (/a/) |
| `whisper_kaha.wav` | 0.70s | p=16 | 25.6 dB | 4.79 dB | 0.0414 | 39 | Oral Vowel (/a/) |

---

## 2. Minimal Pair & Speaker Voice Comparisons

| Comparison Pair | LPC Cosine Dist | Residual Energy Ratio | Formant F1 Shift | Acoustic Contrast |
| :--- | :--- | :--- | :--- | :--- |
| `kaha_vs_kahaan` | 0.0476 | 0.88 | 617 Hz → 273 Hz | Minimal pair contrast (Oral vs Nasalized): LPC cosine distance is 0.0476. T... |
| `baas_vs_baans` | 0.0321 | 1.16 | 613 Hz → 250 Hz | Minimal pair contrast (Oral vs Nasalized): LPC cosine distance is 0.0321. T... |
| `saas_vs_saans` | 0.0610 | 0.92 | 591 Hz → 252 Hz | Minimal pair contrast (Oral vs Nasalized): LPC cosine distance is 0.0610. T... |
| `kaha_male_vs_kaha_female` | 0.0503 | 1.44 | 617 Hz → 793 Hz | Speaker voice comparison (Male vs Female): The female speaker exhibits a su... |
| `hello_male_vs_hello_female` | 0.0487 | 1.74 | 403 Hz → 428 Hz | Speaker voice comparison (Male vs Female): The female speaker exhibits a su... |
| `vowel_i_vs_vowel_u` | 0.0686 | 0.47 | 240 Hz → 260 Hz | Vowel formant topology contrast: Distinct vocal tract geometries yield clea... |
| `kaha_voiced_vs_kaha_whisper` | 0.2568 | 2.62 | 617 Hz → 749 Hz | Excitation source contrast: Voiced speech produces quasi-periodic glottal i... |

---

## 3. Subsystem Deliverables & Production Artifacts Overview

1. **Acoustic Audio Corpus Subsystem**: 14 mono 16 kHz standardized recordings across 3 minimal pairs and speaker contrasts.
2. **Module 1 (LPC Modeling & Vocoder Subsystem)**: `lpc_coefficients.npy`, `lpc_residual.npy`, `lpc_residual.wav`, and `lpc_resynthesized.wav`.
3. **Module 2 (Glottal Phase & Temporal Dynamics Subsystem)**: Zero Frequency Resonator (ZFR) Epochs `gci_indices.npy`, Hilbert envelope, Group delay spectrum, Jitter (15.32%), Shimmer.
4. **Module 3 (Pattern Recognition & DTW Alignment Subsystem)**: Acoustic phonetic classifier achieving **100.0% accuracy** on Hindi oral vs nasal minimal pairs, DTW alignment paths.
