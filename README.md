# 🎙️ Linear Predictive Coding (LPC) Speech & Voice Analysis Suite
### Advanced Modular Acoustic Phonetics & DSP Engineering Architecture

This repository contains the complete, production-grade **Linear Predictive Coding (LPC)** speech analysis, inverse filtering, glottal phase-timing, resynthesis, and minimal-pair classification suite.

Specifically engineered to analyze and distinguish Hindi acoustic minimal pairs:
- **"कहा" (kaha)** [oral vowel /a/] vs **"कहाँ" (kahaan)** [nasalized vowel /ã/]
- **"बास" (baas)** [oral vowel /aː/] vs **"बाँस" (baans)** [nasalized vowel /ãː/]
- **"सास" (saas)** [oral vowel /aː/] vs **"साँस" (saans)** [nasalized vowel /ãː/]
- Cross-speaker comparisons: **Male vs Female voice** (pitch $F_0$ & vocal tract length)
- Vowel topology: **/i/ vs /u/**
- Excitation contrast: **Voiced speech vs Whispered speech**

---

## 🏛️ System Architecture & Subsystem Modules

```
[Module 1: Audio Conditioning & Pre-emphasis]
  Records & conditions speech -> 16 kHz Mono, normalized, pre-emphasized (α = 0.97)
         │
         ▼
[Module 2: LPC Modeling, Resynthesis & Vocoder Engine]
  1. Autocorrelation & Levinson-Durbin Yule-Walker solver (p = 16)
  2. Burg's lattice filter method (guaranteed minimum-phase stability)
  3. Inverse Filtering: e[n] = x[n] + ∑ a_k x[n-k]
  4. Formant extraction (F1-F4) & Z-plane pole-zero geometry
  5. High-fidelity LPC Resynthesis (SNR > 25 dB) & Parametric Vocoder
  6. Line Spectral Frequencies (LSF) & Cepstral Coefficients (LPCC)
         │
         ├─── Residual e[n] ──────────────┐
         ▼                                ▼
[Module 3: Glottal Phase Dynamics]  [Module 4: Pattern Recognition & DTW]
  1. Zero Frequency Resonator (ZFR)   1. Dynamic Time Warping (DTW) trajectory alignment
     Epoch / GCI extraction           2. Acoustic feature extraction (nasal murmur, F1 broadening)
  2. Hilbert instantaneous envelope,  3. AI Ensemble Random Forest Classifier (100% Accuracy)
     phase & instantaneous frequency  4. Full corpus benchmarking & confusion matrix
  3. Modified Group Delay (MGD)
  4. Jitter (%), Shimmer (%), HNR
```

---

## 📂 Repository Structure

```
LPC/
│
├── data/                                 # Standardized Audio Recordings (16 kHz, Mono)
│   ├── kaha.wav                          # Minimal pair 1: oral vowel /a/ (Male)
│   ├── kahaan.wav                        # Minimal pair 1: nasalized vowel /ã/ (Male)
│   ├── baas.wav                          # Minimal pair 2: oral vowel /aː/ ("बास")
│   ├── baans.wav                         # Minimal pair 2: nasalized vowel /ãː/ ("बाँस")
│   ├── saas.wav                          # Minimal pair 3: oral vowel /aː/ ("सास")
│   ├── saans.wav                         # Minimal pair 3: nasalized vowel /ãː/ ("साँस")
│   ├── kaha_female.wav                   # Female voice (F0 ~ 225 Hz, shorter vocal tract)
│   ├── kaha_deep_male.wav                # Deep baritone voice (F0 ~ 95 Hz)
│   ├── hello.wav                         # Male greeting speech
│   ├── hello_female.wav                  # Female greeting speech
│   ├── vowel_a.wav                       # Cardinal open vowel /a/
│   ├── vowel_i.wav                       # Cardinal high-front vowel /i/
│   ├── vowel_u.wav                       # Cardinal high-back rounded vowel /u/
│   └── whisper_kaha.wav                  # Whispered "kaha" (unvoiced turbulent excitation)
│
├── lpc_core/                             # Core DSP & ML Library (44 functions)
│   ├── __init__.py                       # Package exports
│   ├── preprocessor.py                   # Conditioning, resampling, pre/de-emphasis
│   ├── lpc_analyzer.py                   # Framing, autocorrelation, Levinson-Durbin
│   ├── advanced_lpc.py                   # Burg's method, LSF/LSP, LPCC recursion
│   ├── inverse_filter.py                 # Time-varying inverse filtering (A(z))
│   ├── resynthesis.py                    # LPC vocoder, residual resynthesis, voice morphing
│   ├── formant_analyzer.py               # Formant tracking (F1-F4) from roots of A(z)
│   ├── features.py                       # Prediction gain, pitch F0, spectral flatness
│   ├── phase_timing.py                   # Module 3: ZFR Epochs/GCIs, Hilbert, Group Delay
│   ├── classifier.py                     # Module 4: DTW, acoustic features, Random Forest
│   └── visualizer.py                     # High-res publication plots & Z-plane pole graphs
│
├── outputs/                              # Generated Deliverables
│   ├── kaha/, kahaan/, ...               # Coefficients (.npy), residual (.npy, .wav), resynth (.wav)
│   ├── comparison/                       # Multi-panel plots & spectral envelope overlays
│   ├── classification/                   # DTW alignment plots & corpus benchmark JSON
│   └── PIPELINE_RUN_SUMMARY.md           # Master pipeline execution summary
│
├── run_lpc.py                            # CLI single-file processor & full-pipeline runner
├── resynthesize.py                       # CLI tool for speech resynthesis & voice morphing
├── phase_timing_analysis.py              # CLI tool for glottal phase & timing analysis
├── classify_words.py                     # CLI tool for AI classification & DTW alignment
├── compare_words.py                      # CLI tool for acoustic minimal pair comparisons
├── run_pipeline.py                       # Master orchestrator executing entire corpus pipeline
├── generate_test_audio.py                # Source-filter speech synthesis tool
├── app.py                                # Interactive Streamlit Web Application (6 workspaces)
├── LPC_Analysis_Walkthrough.ipynb        # 10-step Jupyter Notebook for interactive DSP demonstration
├── tests/                                # Comprehensive Unit Test Suite
│   ├── __init__.py
│   └── test_lpc_suite.py                 # 11 unit tests covering all DSP formulations
├── TECHNICAL_REPORT.md                   # Comprehensive Research Whitepaper & Technical Specification
├── ENGINEERING_MANUAL.md                 # Engineering Reference Manual & Technical Architecture Specifications
├── requirements.txt                      # Project dependencies
└── README.md                             # This file
```

---

## 🚀 How to Run

### 1. Run Master End-to-End Pipeline
Executes analysis, inverse filtering, resynthesis, phase-timing, and AI classification across all 14 files in one command:
```bash
python run_pipeline.py
```
Generates the complete summary report at `outputs/PIPELINE_RUN_SUMMARY.md`.

---

### 2. Process Any Single WAV File
```bash
python run_lpc.py --input data/kaha.wav --order 16 --frame_ms 25 --hop_ms 10
```
Outputs produced:
- `outputs/kaha/lpc_coefficients.npy` (Vocal tract filter matrix)
- `outputs/kaha/lpc_residual.npy` (Time-aligned prediction error)
- `outputs/kaha/lpc_residual.wav` (Playable 16-bit PCM excitation audio)
- `outputs/kaha/lpc_resynthesized.wav` (Reconstructed speech audio, SNR > 25 dB)
- `outputs/kaha/lpc_summary.json` (Full parameter report)
- `outputs/kaha/plots/` (Waveform, spectrogram, LPC envelope, and Z-plane poles)

---

### 3. Speech Resynthesis & Voice Morphing Studio
```bash
# High-fidelity residual resynthesis
python resynthesize.py --input data/kaha.wav --mode residual

# Parametric LPC vocoder (pitch pulse + noise)
python resynthesize.py --input data/kaha.wav --mode vocoder

# Voice morphing: raise pitch by 25% and simulate shorter female vocal tract
python resynthesize.py --input data/kaha.wav --mode morph --pitch_scale 1.25 --formant_scale 1.15

# Whisperize speech (unvoiced turbulent excitation)
python resynthesize.py --input data/kaha.wav --mode morph --whisper
```

---

### 4. Glottal Phase & Temporal Dynamics Analysis
```bash
python phase_timing_analysis.py --input data/kaha.wav
```
Extracts and saves:
- `outputs/kaha/phase_timing/gci_indices.npy` (Glottal Closure Instant timestamps)
- `outputs/kaha/phase_timing/zfr_signal.npy` (Zero Frequency Resonator trend-removed signal)
- `outputs/kaha/phase_timing/glottal_timing_metrics.json` (Jitter, Shimmer, HNR, Phase Coherence)
- `outputs/kaha/plots/epoch_and_zfr_analysis.png`
- `outputs/kaha/plots/group_delay_spectrum.png`

---

### 5. Acoustic Minimal Pair Classification & DTW Alignment
```bash
# Classify an audio file (predicts Oral vs Nasalized with acoustic explanation)
python classify_words.py --input data/kaha.wav
python classify_words.py --input data/kahaan.wav

# Dynamic Time Warping (DTW) alignment between two utterances
python classify_words.py --compare data/kaha.wav data/kahaan.wav

# Run full corpus benchmark (evaluates entire dataset with 100% accuracy)
python classify_words.py --benchmark
```

---

### 6. Acoustic Minimal Pair & Voice Comparisons
```bash
# Compare minimal pair: oral vs nasalized vowel
python compare_words.py --file1 data/kaha.wav --file2 data/kahaan.wav --name1 kaha --name2 kahaan

# Run all 7 benchmark voice comparisons in batch
python compare_words.py --all
```

---

### 7. Launch Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Featuring 6 specialized workspaces:
1. **🎙️ Module 1 Studio**: Original vs residual playback, waveform/spectrogram, interactive frame inspector, Z-plane complex poles.
2. **🔊 Speech Resynthesis & Vocoder Studio**: Residual vs vocoder resynthesis, interactive pitch shifter, vocal tract formant warper, and whisperizer.
3. **⏱️ Module 2 Studio**: ZFR Epoch (GCI) detection, Hilbert envelope, Modified Group Delay spectrum, Jitter/Shimmer clinical dashboard.
4. **🧠 Module 3 Studio**: Real-time Oral vs Nasalized AI classification with diagnostic radar, DTW alignment heatmap, and feature importance rankings.
5. **👥 Acoustic Phonetics Comparative Studio**: Side-by-side comparison across 8 benchmark pairs with spectral envelope overlays.
6. **📖 System Architecture & DSP Technical Specifications**: Mathematical derivations, acoustic phonetics physics, computational complexity profiling, and whitepaper viewer.

---

### 8. Run Unit Test Suite
```bash
python tests/test_lpc_suite.py
```
Executes 11 unit tests verifying mathematical invertibility, Levinson-Durbin stability, Burg's method, LSF monotonicity, LPCC recursion, resynthesis SNR, ZFR epochs, Hilbert phase, DTW alignment, and classifier accuracy.

---

## 📦 System Deliverables & Data Matrix

| Deliverable | Description | Format | Target Pipeline Subsystem |
| :--- | :--- | :--- | :--- |
| `lpc_coefficients.npy` | Vocal tract all-pole filter matrix | NumPy float64 | Classifier & DTW Engine |
| `lpc_residual.npy` | Prediction error excitation signal $e[n]$ | NumPy float64 | Phase & Timing Engine |
| `lpc_residual.wav` | Playable acoustic residual audio | 16-bit PCM WAV | Audio Inspection & Listening |
| `lpc_resynthesized.wav`| Reconstructed speech audio (SNR > 25 dB) | 16-bit PCM WAV | Synthesis & Quality Benchmark |
| `gci_indices.npy` | Glottal Closure Instant timestamps | NumPy int64 | Temporal Feature Analysis |
| `glottal_timing_metrics.json` | Jitter (%), Shimmer (%), HNR (dB) | JSON | Acoustic Diagnostics |
| `lpc_summary.json` | Complete LPC, LSF, LPCC & formant summary | JSON | Acoustic Parameter Store |
| `comparison_*.png` | Multi-panel comparison graphs | High-res PNG | Comparative Analysis |
| `PIPELINE_RUN_SUMMARY.md` | Master pipeline execution report | Markdown | Performance & Accuracy Audit |

---

## 📄 Research Whitepaper & Technical Documentation
- **[TECHNICAL_REPORT.md](file:///c:/Users/vasud/OneDrive/Desktop/LPC/TECHNICAL_REPORT.md)**: Formal research whitepaper covering theoretical derivations, acoustic phonetics of velopharyngeal coupling, full corpus measurements, and experimental benchmarks.
- **[ENGINEERING_MANUAL.md](file:///c:/Users/vasud/OneDrive/Desktop/LPC/ENGINEERING_MANUAL.md)**: In-depth engineering reference manual and comprehensive technical specifications.
