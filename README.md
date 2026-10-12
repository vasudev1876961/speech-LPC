# 🎙️ Linear Predictive Coding (LPC) Speech Analysis Suite
### Member 2: LPC Modeling, Inverse Filtering, and Residual Extraction

This repository contains the complete, production-grade **Linear Predictive Coding (LPC)** implementation for an acoustic phonetics speech recognition/differentiation project.

Specifically tailored to analyze and distinguish Hindi minimal pairs like **"कहा" (kaha)** vs **"कहाँ" (kahaan)**.

---

## 📂 Repository Structure

```
LPC/
│
├── data/                                 # Audio recordings (16 kHz, Mono)
│   ├── kaha.wav                          # Minimal pair 1: oral vowel /a/
│   ├── kahaan.wav                        # Minimal pair 2: nasalized vowel /ã/
│   └── hello.wav                         # Test greeting speech
│
├── outputs/                              # Generated deliverables for Members 3 & 4
│   ├── kaha/
│   │   ├── lpc_coefficients.npy          # Estimated filter weights (num_frames, 17)
│   │   ├── lpc_residual.npy              # Numerical residual signal (num_samples,)
│   │   ├── lpc_residual.wav              # Playable residual audio
│   │   ├── lpc_summary.json              # Acoustic summary & formant metrics
│   │   └── plots/                        # Waveform, Spectrogram & Formant Envelope
│   ├── kahaan/
│   └── comparison/                       # Side-by-side comparison report & graphs
│
├── lpc_core/                             # Core Python DSP library
│   ├── __init__.py
│   ├── preprocessor.py                   # Load, resample, normalize, pre-emphasis
│   ├── lpc_analyzer.py                   # Framing, autocorrelation, Levinson-Durbin
│   ├── inverse_filter.py                 # Time-varying inverse filtering (A(z))
│   ├── formant_analyzer.py               # Formant tracking (F1-F4) from polynomial roots
│   ├── features.py                       # Prediction gain, pitch, residual RMS
│   └── visualizer.py                     # High-res publication-quality plots
│
├── run_lpc.py                            # CLI execution script for single WAV files
├── compare_words.py                      # Comparative script for minimal pairs
├── app.py                                # Interactive Streamlit Web Application
├── generate_test_audio.py                # Source-filter speech synthesis tool
├── LPC_Analysis_Walkthrough.ipynb        # Runnable Jupyter Notebook for demos & grading
├── PROJECT_EXPLANATION_AND_VIVA.md       # Plain-English guide & viva question answers
└── README.md                             # This file
```

---

## 🚀 How to Run

### 1. Process Any Audio File (CLI)
To run LPC analysis on any WAV file:
```bash
python run_lpc.py --input data/kaha.wav --order 16 --frame_ms 25 --hop_ms 10
```
This automatically extracts and saves:
- `outputs/kaha/lpc_coefficients.npy`
- `outputs/kaha/lpc_residual.npy`
- `outputs/kaha/lpc_residual.wav`
- `outputs/kaha/plots/waveform_and_residual.png`
- `outputs/kaha/plots/spectrogram_comparison.png`
- `outputs/kaha/plots/lpc_spectral_envelope.png`

---

### 2. Compare Two Words (e.g. 'kaha' vs 'kahaan')
To run the automated acoustic comparison:
```bash
python compare_words.py --file1 data/kaha.wav --file2 data/kahaan.wav --name1 kaha --name2 kahaan
```
This outputs:
- Mean LPC Cosine Distance
- Residual RMS energy comparison
- Formant shifts between oral and nasalized vowels
- Multi-panel comparison graphs in `outputs/comparison/`

---

### 3. Launch the Interactive Web Dashboard
Run the Streamlit application to explore the audio, listen to the residual, and inspect plots interactively:
```bash
streamlit run app.py
```

Features in the Web App:
- 🔊 Play original speech vs LPC residual side-by-side
- 🎚️ Interactive sliders for LPC order $p$, frame size, and hop size
- 📊 Interactive inspection of formant peaks and spectral envelopes for each frame
- 💾 One-click download buttons for Member 3 deliverables

---

### 4. Run the Jupyter Notebook
Open and run `LPC_Analysis_Walkthrough.ipynb` in VS Code or Jupyter:
```bash
jupyter notebook LPC_Analysis_Walkthrough.ipynb
```

---

## 📦 Deliverables for Member 3 & Member 4

| File | Purpose | Consumers |
| :--- | :--- | :--- |
| `lpc_coefficients.npy` | Frame-by-frame vocal tract filter coefficients $(p+1=17)$ | Member 4 (Feature classification, DTW) |
| `lpc_residual.npy` | Numerical prediction error $e[n]$ with exact sample alignment | Member 3 (Phase, timing, glottal pulse shape) |
| `lpc_residual.wav` | 16-bit PCM playable audio of the residual | Human inspection / listening tests |
| `lpc_summary.json` | Formant resonances, prediction gain, and residual metrics | Member 4 |

---

## 📚 Viva & Presentation Prep
Read [PROJECT_EXPLANATION_AND_VIVA.md](file:///c:/Users/vasud/OneDrive/Desktop/LPC/PROJECT_EXPLANATION_AND_VIVA.md) for clear, simple English explanations of:
- Why we predict samples
- Why the residual (error) is useful
- What inverse filtering is
- How oral vowels differ from nasalized vowels in Hindi
