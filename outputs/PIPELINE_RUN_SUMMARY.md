# 🎙️ LPC Suite: Complete Pipeline Run Summary

**Execution Timestamp:** 2026-10-26 12:25:17  
**Total Runtime:** 73.33 seconds  
**Total Audio Utterances Processed:** 14  
**Minimal Pair Classification Accuracy:** 100.0% (9/9)  

---

## 1. Processed Corpus Deliverables

| Audio File | Duration (s) | LPC Order | Reconstruction SNR | Prediction Gain | Residual RMS | Epochs | Prediction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baans.wav` | 0.75s | p=16 | 25.6 dB | 8.58 dB | 0.0854 | 55 | Nasalized Vowel (/ã/) |
| `baas.wav` | 0.75s | p=16 | 25.6 dB | 9.39 dB | 0.0680 | 51 | Oral Vowel (/a/) |
| `hello.wav` | 0.70s | p=16 | 25.6 dB | 4.71 dB | 0.0666 | 63 | Oral Vowel (/a/) |
| `hello_female.wav` | 0.70s | p=16 | 25.6 dB | 5.14 dB | 0.0687 | 68 | Oral Vowel (/a/) |
| `kaha.wav` | 0.70s | p=16 | 25.6 dB | 3.99 dB | 0.0707 | 56 | Oral Vowel (/a/) |
| `kaha_deep_male.wav` | 0.70s | p=16 | 25.6 dB | 3.48 dB | 0.0653 | 40 | Oral Vowel (/a/) |
| `kaha_female.wav` | 0.70s | p=16 | 25.6 dB | 4.38 dB | 0.0763 | 57 | Oral Vowel (/a/) |
| `kahaan.wav` | 0.74s | p=16 | 25.6 dB | 3.19 dB | 0.1015 | 64 | Nasalized Vowel (/ã/) |
| `saans.wav` | 0.78s | p=16 | 25.6 dB | 9.46 dB | 0.0790 | 50 | Nasalized Vowel (/ã/) |
| `saas.wav` | 0.78s | p=16 | 25.6 dB | 9.96 dB | 0.0754 | 51 | Oral Vowel (/a/) |
| `vowel_a.wav` | 0.65s | p=16 | 25.6 dB | 4.01 dB | 0.0965 | 64 | Oral Vowel (/a/) |
| `vowel_i.wav` | 0.65s | p=16 | 25.6 dB | 7.21 dB | 0.0731 | 70 | Oral Vowel (/a/) |
| `vowel_u.wav` | 0.65s | p=16 | 25.6 dB | 4.20 dB | 0.0825 | 69 | Oral Vowel (/a/) |
| `whisper_kaha.wav` | 0.70s | p=16 | 25.6 dB | 4.86 dB | 0.0436 | 43 | Oral Vowel (/a/) |

---

## 2. Minimal Pair & Speaker Voice Comparisons

| Comparison Pair | LPC Cosine Dist | Residual Energy Ratio | Formant F1 Shift | Acoustic Contrast |
| :--- | :--- | :--- | :--- | :--- |
| `kaha_vs_kahaan` | 0.0605 | 1.44 | 1232 Hz → 1083 Hz | Minimal pair contrast (Oral vs Nasalized): LPC cosine distance is 0.0605. T... |
| `baas_vs_baans` | 0.1029 | 1.26 | 1256 Hz → 1211 Hz | Minimal pair contrast (Oral vs Nasalized): LPC cosine distance is 0.1029. T... |
| `saas_vs_saans` | 0.0806 | 1.05 | 1223 Hz → 2734 Hz | Minimal pair contrast (Oral vs Nasalized): LPC cosine distance is 0.0806. T... |
| `kaha_male_vs_kaha_female` | 0.1104 | 1.08 | 1232 Hz → 979 Hz | Speaker voice comparison (Male vs Female): The female speaker exhibits a su... |
| `hello_male_vs_hello_female` | 0.1014 | 1.03 | 961 Hz → 1133 Hz | Speaker voice comparison (Male vs Female): The female speaker exhibits a su... |
| `vowel_i_vs_vowel_u` | 0.3744 | 1.13 | 2228 Hz → 839 Hz | Vowel formant topology contrast: Distinct vocal tract geometries yield clea... |
| `kaha_voiced_vs_kaha_whisper` | 0.2459 | 0.62 | 1232 Hz → 824 Hz | Excitation source contrast: Voiced speech produces quasi-periodic glottal i... |

---

## 3. Subsystem Deliverables & Production Artifacts Overview

1. **Acoustic Audio Corpus Subsystem**: 14 mono 16 kHz standardized recordings across 3 minimal pairs and speaker contrasts.
2. **Module 1 (LPC Modeling & Vocoder Subsystem)**: `lpc_coefficients.npy`, `lpc_residual.npy`, `lpc_residual.wav`, and `lpc_resynthesized.wav`.
3. **Module 2 (Glottal Phase & Temporal Dynamics Subsystem)**: Zero Frequency Resonator (ZFR) Epochs `gci_indices.npy`, Hilbert envelope, Group delay spectrum, Jitter (20.11%), Shimmer.
4. **Module 3 (Pattern Recognition & DTW Alignment Subsystem)**: Acoustic phonetic classifier achieving **100.0% accuracy** on Hindi oral vs nasal minimal pairs, DTW alignment paths.
