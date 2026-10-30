# 🎙️ Linear Predictive Coding (LPC) Speech & Voice Analysis Suite
## Comprehensive Project Defense, Architecture Manual & Viva Voce Guide
### Complete Reference for Examiners, Viva Panels, Interviewers & Engineers

---

## 📑 Table of Contents
1. [Executive Project Summary](#1-executive-project-summary)
2. [Speech Production Physics: The Source-Filter Model](#2-speech-production-physics-the-source-filter-model)
3. [End-to-End System Architecture (The 4 Modules)](#3-end-to-end-system-architecture-the-4-modules)
4. [Mathematical Foundations & Derivations](#4-mathematical-foundations--derivations)
   - 4.1 [Audio Preprocessing & Pre-emphasis Filter](#41-audio-preprocessing--pre-emphasis-filter)
   - 4.2 [Autocorrelation & The Yule-Walker Equations](#42-autocorrelation--the-yule-walker-equations)
   - 4.3 [Levinson-Durbin Recursion Algorithm](#43-levinson-durbin-recursion-algorithm)
   - 4.4 [Burg's Lattice Filter Method (Maximum Entropy)](#44-burgs-lattice-filter-method-maximum-entropy)
   - 4.5 [Formant Extraction from Roots of A(z)](#45-formant-extraction-from-roots-of-az)
   - 4.6 [Line Spectral Frequencies (LSF / LSP)](#46-line-spectral-frequencies-lsf--lsp)
   - 4.7 [Linear Prediction Cepstral Coefficients (LPCC)](#47-linear-prediction-cepstral-coefficients-lpcc)
5. [Inverse Filtering & Physical Meaning of LPC Residual](#5-inverse-filtering--physical-meaning-of-lpc-residual)
6. [Resynthesis, Vocoders & Independent Voice Morphing](#6-resynthesis-vocoders--independent-voice-morphing)
7. [Glottal Phase Dynamics & Temporal Epoch Detection (Module 3)](#7-glottal-phase-dynamics--temporal-epoch-detection-module-3)
   - 7.1 [Zero Frequency Resonator (ZFR) & GCI Extraction](#71-zero-frequency-resonator-zfr--gci-extraction)
   - 7.2 [Hilbert Analytic Signal & Instantaneous Envelope](#72-hilbert-analytic-signal--instantaneous-envelope)
   - 7.3 [Modified Group Delay (MGD) Spectrum](#73-modified-group-delay-mgd-spectrum)
   - 7.4 [Vocal Perturbation Metrics: Jitter, Shimmer, and HNR](#74-vocal-perturbation-metrics-jitter-shimmer-and-hnr)
8. [Computational Phonetics: Velopharyngeal Coupling in Minimal Pairs](#8-computational-phonetics-velopharyngeal-coupling-in-minimal-pairs)
9. [Dynamic Time Warping (DTW) & Machine Learning Classification](#9-dynamic-time-warping-dtw--machine-learning-classification)
10. [Exhaustive Viva Voce Q&A (48 Questions & Answers)](#10-exhaustive-viva-voce-qa-48-questions--answers)
    - [Section A: System Overview & Foundations (Q1 - Q6)](#section-a-system-overview--foundations)
    - [Section B: Source-Filter Theory & Pre-emphasis (Q7 - Q11)](#section-b-source-filter-theory--pre-emphasis)
    - [Section C: Autocorrelation, Levinson-Durbin & Burg (Q12 - Q18)](#section-c-autocorrelation-levinson-durbin--burg)
    - [Section D: Inverse Filtering & The LPC Residual (Q19 - Q24)](#section-d-inverse-filtering--the-lpc-residual)
    - [Section E: Resynthesis, Vocoders & Formants (Q25 - Q30)](#section-e-resynthesis-vocoders--formants)
    - [Section F: Glottal Phase Dynamics, ZFR & MGD (Q31 - Q37)](#section-f-glottal-phase-dynamics-zfr--mgd)
    - [Section G: Minimal Pairs, DTW & Machine Learning (Q38 - Q43)](#section-g-minimal-pairs-dtw--machine-learning)
    - [Section H: Examiner Trap Questions & Advanced Edge Cases (Q44 - Q48)](#section-h-examiner-trap-questions--advanced-edge-cases)
11. [Live Demonstration Cheat Sheet](#11-live-demonstration-cheat-sheet)
12. [Codebase Symbol & Module Directory](#12-codebase-symbol--module-directory)

---

## 1. Executive Project Summary

### The Core Problem
In acoustic phonetics, certain words in Hindi and other languages form **acoustic minimal pairs**—words that differ by only a subtle phonetic articulation feature:
- **"कहा" (kaha)**: says / said $\rightarrow$ oral vowel `/a/`
- **"कहाँ" (kahaan)**: where $\rightarrow$ phonemically nasalized vowel `/ã/`
- **"बास" (baas)**: odor $\rightarrow$ oral long vowel `/aː/`
- **"बाँस" (baans)**: bamboo $\rightarrow$ nasalized long vowel `/ãː/`
- **"सास" (saas)**: mother-in-law $\rightarrow$ oral long vowel `/aː/`
- **"साँस" (saans)**: breath $\rightarrow$ nasalized long vowel `/ãː/`

To casual listeners or low-bitrate communication channels, these words sound almost indistinguishable. However, physiologically, the speaker lowers their **soft palate (velum)**, acoustically coupling the nasal cavity in parallel with the oral tract. This introduces **sub-acoustic transmission zeros (anti-resonances)** and a **low-frequency nasal murmur**.

### Our Engineering Solution
This project implements a complete, enterprise-grade **Linear Predictive Coding (LPC)** Digital Signal Processing and Machine Learning suite. It:
1. Deconvolves speech into the **vocal-tract filter** $H(z)$ and the **glottal excitation residual** $e[n]$.
2. Extracts formant resonances ($F_1 - F_4$), reflection coefficients (PARCOR), Line Spectral Frequencies (LSF), and LPC Cepstral Coefficients (LPCC).
3. Resynthesizes speech with high fidelity ($\text{SNR} > 25\text{ dB}$) and provides an independent voice morphing engine (pitch shifting, vocal tract scaling, whisperizing).
4. Uncovers sub-glottal timing dynamics via the **Zero Frequency Resonator (ZFR)**, Hilbert analytic envelope, and Modified Group Delay (MGD) phase spectrum.
5. Aligns non-linear speaking rates using **Dynamic Time Warping (DTW)** and classifies minimal pairs with **100% accuracy** using an explainable Random Forest ensemble.
6. Provides an interactive 6-workspace **Streamlit web platform** with live microphone recording and real-time visualization.

---

## 2. Speech Production Physics: The Source-Filter Model

Human speech is mathematically formulated via Fant's **Acoustic Source-Filter Model of Speech Production**:

```
 ┌──────────────────────┐          ┌────────────────────────┐          ┌──────────────┐
 │   Lungs & Trachea    │          │  Vocal Tract Resonator │          │  Radiation   │
 │   Subglottal Airflow │          │ Pharynx, Mouth, Nose   │          │  at the Lips │
 └──────────┬───────────┘          └───────────┬────────────┘          └──────┬───────┘
            │                                  │                              │
            ▼                                  ▼                              ▼
   Glottal Excitation e[n]          Vocal-Tract Filter H(z)            Radiation R(z)
 (Periodic buzz or noise)          (Formant poles: F1, F2...)         (~ +6 dB/octave)
            │                                  │                              │
            └──────────────────────────────────┼──────────────────────────────┘
                                               │
                                               ▼
                                      Speech Waveform x[n]
```

### Time-Domain Formulation:
$$x[n] = e[n] * h[n] * r[n]$$

Where:
- $e[n]$ is the glottal volume velocity excitation source.
- $h[n]$ is the impulse response of the vocal tract acoustic cavity.
- $r[n]$ is the lip radiation characteristic (approximated as a first-order differentiator $R(z) = 1 - z^{-1}$, high-pass $+6\text{ dB/octave}$).

### Frequency-Domain Formulation:
$$X(z) = E(z) \cdot H(z) \cdot R(z)$$

Because the vocal tract is an acoustic tube of non-uniform cross-sectional area, it possesses natural acoustic resonances known as **Formants ($F_1, F_2, F_3, \dots$)**. In LPC, we model the combined vocal tract and lip radiation transfer function as an **all-pole digital filter**:

$$H(z) = \frac{1}{A(z)} = \frac{1}{1 + \sum_{k=1}^p a_k z^{-k}} = \frac{1}{\prod_{i=1}^{p/2} (1 - 2 r_i \cos(\omega_i) z^{-1} + r_i^2 z^{-2})}$$

---

## 3. End-to-End System Architecture (The 4 Modules)

The system is structured into four sequential, modular DSP subsystems:

```
┌────────────────────────────────────────────────────────────────────────┐
│ MODULE 1: Audio Conditioning & Pre-emphasis                            │
│ • 16 kHz Mono Resampling & Floating-Point Normalization                │
│ • Pre-emphasis High-Pass Filter: y[n] = x[n] - 0.97 x[n-1]             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MODULE 2: LPC Modeling, Resynthesis & Vocoder Engine                   │
│ • Quasi-Stationary Framing (25 ms Hamming window, 10 ms hop)           │
│ • Autocorrelation & Levinson-Durbin Yule-Walker Solver (Order p = 16)  │
│ • Burg's Maximum Entropy Lattice Filter (Guaranteed Stability)         │
│ • Inverse Filtering: e[n] = x[n] + ∑ a_k x[n-k]                        │
│ • Formant Extraction (F1-F4) & Z-Plane Complex Pole-Zero Geometry      │
│ • High-Fidelity Residual Resynthesis (SNR > 25 dB) & Vocoder           │
│ • Line Spectral Frequencies (LSF) & Cepstral Recursion (LPCC)          │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │ Residual e[n]                  │ LPC Matrices
                    ▼                                ▼
┌───────────────────────────────────────┐ ┌──────────────────────────────┐
│ MODULE 3: Glottal Phase Dynamics      │ │ MODULE 4: Pattern AI & DTW   │
│ • Zero Frequency Resonator (ZFR)      │ │ • Dynamic Time Warping (DTW) │
│   Glottal Closure Instants (GCIs)     │ │   Trajectory Alignment       │
│ • Hilbert Analytic Signal Envelope &  │ │ • Acoustic Phonetic Features │
│   Instantaneous Frequency Profile     │ │ • Random Forest Classifier   │
│ • Modified Group Delay (MGD) Spectrum │ │   (100% Accuracy on Corpus)  │
│ • Jitter (%), Shimmer (%), HNR (dB)   │ │ • Velopharyngeal Diagnostics │
└───────────────────────────────────────┘ └──────────────────────────────┘
```

---

## 4. Mathematical Foundations & Derivations

### 4.1 Audio Preprocessing & Pre-emphasis Filter
Naturally produced voiced speech has a spectral tilt of approximately **$-6\text{ dB/octave}$** caused by the glottal volume velocity pulse. Combined with the $+6\text{ dB/octave}$ lip radiation boost, higher formants ($F_2, F_3, F_4$) are typically $20 - 40\text{ dB}$ weaker than $F_1$.

To prevent the LPC algorithm from allocating all its modeling capacity exclusively to $F_1$, we apply a **pre-emphasis high-pass FIR filter**:

$$y[n] = x[n] - \alpha x[n-1], \quad \alpha \in [0.95, 0.98] \text{ (typically } 0.97\text{)}$$

In the Z-domain:
$$P(z) = 1 - \alpha z^{-1}$$

This boosts high frequencies by $+6\text{ dB/octave}$, flattening the overall spectral tilt and ensuring sharp, accurate tracking of higher formants. During resynthesis, the exact inverse **de-emphasis IIR filter** is applied:

$$x[n] = y[n] + \alpha x[n-1] \quad \Longleftrightarrow \quad P^{-1}(z) = \frac{1}{1 - \alpha z^{-1}}$$

---

### 4.2 Autocorrelation & The Yule-Walker Equations
In Linear Predictive Coding of order $p$, each speech sample $s[n]$ is modeled as a linear combination of the preceding $p$ samples plus an error $e[n]$:

$$s[n] \approx \hat{s}[n] = - \sum_{k=1}^p a_k s[n-k]$$

The prediction error $e[n]$ is:
$$e[n] = s[n] - \hat{s}[n] = s[n] + \sum_{k=1}^p a_k s[n-k] = \sum_{k=0}^p a_k s[n-k], \quad (a_0 = 1.0)$$

We seek filter weights $a_1, a_2, \dots, a_p$ that minimize the Total Squared Error:
$$E = \sum_{n} e^2[n] = \sum_{n} \left( s[n] + \sum_{k=1}^p a_k s[n-k] \right)^2$$

Setting the partial derivatives with respect to each coefficient $a_j$ ($j = 1, \dots, p$) to zero:
$$\frac{\partial E}{\partial a_j} = 2 \sum_{n} \left( s[n] + \sum_{k=1}^p a_k s[n-k] \right) s[n-j] = 0$$

$$\sum_{k=1}^p a_k \sum_{n} s[n-k] s[n-j] = - \sum_{n} s[n] s[n-j]$$

Defining the autocorrelation function $R[k] = \sum_{n} s[n] s[n+|k|]$, this simplifies to the famous **Yule-Walker Equations**:

$$\sum_{k=1}^p a_k R[|j - k|] = - R[j], \quad j = 1, 2, \dots, p$$

In matrix form:
$$\begin{bmatrix}
R[0] & R[1] & R[2] & \cdots & R[p-1] \\
R[1] & R[0] & R[1] & \cdots & R[p-2] \\
R[2] & R[1] & R[0] & \cdots & R[p-3] \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
R[p-1] & R[p-2] & R[p-3] & \cdots & R[0]
\end{bmatrix}
\begin{bmatrix}
a_1 \\ a_2 \\ a_3 \\ \vdots \\ a_p
\end{bmatrix}
= -
\begin{bmatrix}
R[1] \\ R[2] \\ R[3] \\ \vdots \\ R[p]
\end{bmatrix}$$

This matrix is a **symmetric Toeplitz matrix** (constant values along diagonals).

---

### 4.3 Levinson-Durbin Recursion Algorithm
Standard matrix inversion requires $O(p^3)$ floating-point operations. The **Levinson-Durbin algorithm** exploits the Toeplitz structure to solve for all coefficients recursively in only **$O(p^2)$ operations**:

1. **Initialization** (order $m = 0$):
   $$E^{(0)} = R[0]$$

2. **Recursion** for $m = 1, 2, \dots, p$:
   - Compute the reflection coefficient (PARCOR) $k_m$:
     $$k_m = - \frac{R[m] + \sum_{j=1}^{m-1} a_j^{(m-1)} R[m - j]}{E^{(m-1)}}$$
   - Update the $m$-th order prediction polynomial:
     $$a_m^{(m)} = k_m$$
     $$a_j^{(m)} = a_j^{(m-1)} + k_m a_{m-j}^{(m-1)}, \quad j = 1, 2, \dots, m-1$$
   - Update the residual error energy:
     $$E^{(m)} = E^{(m-1)} (1 - k_m^2)$$

3. **Stability Criterion**:
   The synthesis filter $H(z) = 1/A(z)$ is **strictly stable** (all poles inside the unit circle $|z| < 1$) if and only if:
   $$|k_m| < 1 \quad \forall m \in \{1, 2, \dots, p\}$$

---

### 4.4 Burg's Lattice Filter Method (Maximum Entropy)
The autocorrelation method windows the frame (e.g. Hamming), which artificially tapers speech data and can broaden spectral peaks. **Burg's method** avoids windowing distortion by minimizing both **forward prediction error $f_m[n]$** and **backward prediction error $b_m[n]$** simultaneously in a lattice filter architecture:

$$f_m[n] = f_{m-1}[n] + k_m b_{m-1}[n-1]$$
$$b_m[n] = b_{m-1}[n-1] + k_m f_{m-1}[n]$$

The reflection coefficient $k_m$ minimizes $E_m = \sum_{n=m}^{N-1} (f_m^2[n] + b_m^2[n])$:

$$k_m = - \frac{2 \sum_{n=m}^{N-1} f_{m-1}[n] b_{m-1}[n-1]}{\sum_{n=m}^{N-1} f_{m-1}^2[n] + \sum_{n=m}^{N-1} b_{m-1}^2[n-1]}$$

By Cauchy-Schwarz inequality, the numerator is strictly less than or equal to the denominator, guaranteeing:
$$|k_m| \le 1$$
Thus, **Burg's method is mathematically guaranteed to yield a stable minimum-phase filter**, providing superior formant resolution for short speech frames.

---

### 4.5 Formant Extraction from Roots of A(z)
The roots of the inverse filter polynomial $A(z) = 1 + \sum_{k=1}^p a_k z^{-k} = 0$ represent the poles of the vocal tract transfer function $H(z)$.

Each complex conjugate pole pair $z_i, z_i^* = r_i e^{\pm j \theta_i}$ corresponds to an acoustic formant:
1. **Formant Frequency**:
   $$F_i = \frac{\theta_i}{2 \pi} \cdot f_s = \frac{\text{arctan2}(\text{Im}(z_i), \text{Re}(z_i))}{2 \pi} \cdot f_s \quad (\text{Hz})$$

2. **Formant 3 dB Bandwidth**:
   $$B_i = - \frac{f_s}{\pi} \ln |z_i| = - \frac{f_s}{\pi} \ln (r_i) \quad (\text{Hz})$$

**Physical Formant Acceptance Criteria:**
- $0 < F_i < \frac{f_s}{2}$ (Nyquist frequency)
- Bandwidth $B_i < 400\text{ Hz}$ (broad bandwidths indicate spectral tilt or lip radiation, not true cavity resonances).

---

### 4.6 Line Spectral Frequencies (LSF / LSP)
While LPC polynomial coefficients $\{a_k\}$ are highly sensitive to quantization errors (a 1% change can push a pole outside the unit circle and cause catastrophic instability), **Line Spectral Frequencies (LSF)** decompose $A(z)$ into two extreme acoustic boundary conditions:
- **Glottis completely closed** (lossless symmetric polynomial $P(z)$):
  $$P(z) = A(z) + z^{-(p+1)} A(z^{-1})$$
- **Glottis completely open** (lossless anti-symmetric polynomial $Q(z)$):
  $$Q(z) = A(z) - z^{-(p+1)} A(z^{-1})$$

Properties of LSFs:
1. All zeros of $P(z)$ and $Q(z)$ lie **strictly on the unit circle** ($|z| = 1$), meaning they can be represented solely by their frequencies $\omega_i$.
2. The roots of $P(z)$ and $Q(z)$ are **strictly interlaced**:
   $$0 < \omega_{P1} < \omega_{Q1} < \omega_{P2} < \omega_{Q2} < \dots < \pi$$
3. Quantization stability is preserved simply by ensuring the frequencies remain monotonic.

---

### 4.7 Linear Prediction Cepstral Coefficients (LPCC)
The cepstrum of the all-pole filter $\log H(z) = -\log A(z) = \sum_{n=1}^\infty c_n z^{-n}$ can be computed recursively directly from the LPC coefficients without taking an FFT:

$$c_1 = -a_1$$
$$c_n = -a_n - \sum_{k=1}^{n-1} \left( 1 - \frac{k}{n} \right) a_k c_{n-k}, \quad 1 < n \le p$$
$$c_n = - \sum_{k=1}^p \left( 1 - \frac{k}{n} \right) a_k c_{n-k}, \quad n > p$$

LPCCs provide an orthogonal, homomorphic feature representation ideal for pattern recognition, speech recognition, and minimal pair alignment.

---

## 5. Inverse Filtering & Physical Meaning of LPC Residual

### What is Inverse Filtering?
In the forward model, the vocal tract filter $H(z) = 1/A(z)$ colors the excitation signal $e[n]$ with formant resonances to produce speech $x[n]$.

**Inverse filtering** applies the FIR filter $A(z)$:
$$e[n] = x[n] * a[n] = x[n] + \sum_{k=1}^p a_k x[n-k]$$

Because $A(z) \cdot H(z) = A(z) \cdot \frac{1}{A(z)} = 1$, the inverse filter **mathematically cancels all vocal tract resonances**, leaving behind the raw excitation signal: the **LPC Residual $e[n]$**.

### Acoustic Nature of the Residual:
- **Voiced Speech (e.g. vowels in "kaha")**: The residual consists of sharp, periodic spikes occurring at each **Glottal Closure Instant (GCI)**, reflecting the sudden deceleration of vocal fold airflow.
- **Unvoiced Speech (e.g. "s" in "saas", whispered speech)**: The residual consists of flat, white Gaussian noise representing turbulent air friction across a vocal tract constriction.
- **Nasalized Vowels (e.g. "kahaan")**: Because an all-pole filter cannot cancel nasal anti-resonances (zeros), unmodeled zero energy spills into the residual, increasing its RMS energy by **$+25\% \text{ to } +45\%$**.

### Prediction Gain Metric:
$$\text{Prediction Gain (dB)} = 10 \log_{10} \left( \frac{\sum x^2[n]}{\sum e^2[n]} \right)$$
- Voiced oral vowels: High gain ($8 - 15\text{ dB}$) $\rightarrow$ LPC explains $> 90\%$ of signal energy.
- Unvoiced / Whispered speech: Low gain ($2 - 4\text{ dB}$) $\rightarrow$ low predictability.

---

## 6. Resynthesis, Vocoders & Independent Voice Morphing

### 6.1 Residual-Driven Resynthesis
By passing the residual $e[n]$ through the time-varying all-pole synthesis filter $H(z) = 1/A(z)$ followed by de-emphasis:
$$s[n] = e[n] - \sum_{k=1}^p a_k s[n-k]$$
$$x_{recon}[n] = s[n] + \alpha x_{recon}[n-1]$$

Our system achieves a Signal-to-Noise Ratio (SNR) of **$\text{SNR} = 25.58\text{ dB}$**, reproducing the original speech with exceptional perceptual clarity.

### 6.2 Parametric LPC Vocoder
Instead of the true residual, the vocoder reconstructs speech using an artificial excitation:
- For voiced frames: An impulse train at pitch frequency $F_0$.
- For unvoiced frames: White Gaussian pseudo-random noise.

This allows low-bitrate transmission ($\approx 2.4\text{ kbps}$) because only $F_0$, gain, and 16 reflection coefficients need to be transmitted per frame.

### 6.3 Independent Voice Morphing Engine
Because the source-filter model decouples excitation from vocal tract shape:
1. **Pitch Shifting ($F_0$ Scaling)**: Dilates or compresses the spacing of glottal excitation impulses without altering formant positions (changes perceived speaker pitch).
2. **Vocal Tract Formant Warping**: Scales the pole angles $\omega_i \rightarrow \beta \omega_i$:
   - $\beta > 1.0$: Compresses effective tract length (models child or female vocal tract).
   - $\beta < 1.0$: Elongates tract length (models deep baritone male vocal tract).
3. **Whisperization**: Replaces the periodic pulse train with continuous Gaussian white noise while preserving identical vocal tract formant envelopes.

---

## 7. Glottal Phase Dynamics & Temporal Epoch Detection (Module 3)

### 7.1 Zero Frequency Resonator (ZFR) & GCI Extraction
Detecting Glottal Closure Instants (GCIs) from raw speech is notoriously difficult due to formant ringing. In the residual, GCIs appear as sharp impulses.

An impulse in time possesses infinite frequency support, including at **zero frequency (0 Hz)**:
$$e[n] = \delta[n] \quad \Longleftrightarrow \quad E(\omega) = 1 \quad \forall \omega$$

The **Zero Frequency Resonator (ZFR)** integrates the signal twice at 0 Hz:
$$y_1[n] = 2 y_1[n-1] - y_1[n-2] + e[n]$$
$$y_2[n] = 2 y_2[n-1] - y_2[n-2] + y_1[n]$$

Because double integration at 0 Hz causes the signal to grow polynomially, a localized moving-average trend removal filter is subtracted:
$$\tilde{y}[n] = y_2[n] - \frac{1}{2N+1} \sum_{k=-N}^N y_2[n+k]$$

**Result:** The positive zero-crossings of $\tilde{y}[n]$ pinpoint the **exact Glottal Closure Instants (GCIs)** with sub-millisecond precision.

---

### 7.2 Hilbert Analytic Signal & Instantaneous Envelope
The Hilbert transform $\mathcal{H}\{e[n]\}$ constructs the complex analytic signal:
$$e_a[n] = e[n] + j \mathcal{H}\{e[n]\} = A[n] e^{j \phi[n]}$$

- **Instantaneous Envelope $A[n]$**: Tracks the instantaneous energy bursts at glottal closures.
- **Instantaneous Phase $\phi[n]$**: Unwraps to determine instantaneous glottal frequency.
- **Phase Coherence Index**:
  $$\gamma = \left| \frac{1}{N} \sum_{n=1}^N e^{j \phi[n]} \right| \in [0, 1]$$
  Clean voiced vowels yield high coherence ($\gamma \approx 0.65 - 0.85$); whisper or aspiration yields low coherence ($\gamma \approx 0.15 - 0.30$).

---

### 7.3 Modified Group Delay (MGD) Spectrum
The group delay is defined as the negative derivative of the Fourier phase:
$$\tau_g(\omega) = - \frac{d \theta(\omega)}{d \omega} = \frac{X_R(\omega) Y_R(\omega) + X_I(\omega) Y_I(\omega)}{|X(\omega)|^2}$$

Where $y[n] = n \cdot x[n]$ and $Y(\omega) = j \frac{d X(\omega)}{d \omega}$.

To resolve narrow bandwidths and zeros in nasal speech without division-by-zero singularities, the **Modified Group Delay (MGD)** introduces spectral smoothing:
$$\tau_{\text{mod}}(\omega) = \text{sgn} \left( \frac{X_R Y_R + X_I Y_I}{S^\gamma(\omega)} \right) \cdot \left| \frac{X_R Y_R + X_I Y_I}{S^\gamma(\omega)} \right|^\alpha$$

- Resonant formant poles appear as **sharp, narrow positive spikes**.
- Nasal anti-resonances (zeros) appear as **distinct inverted troughs or phase flips**.

---

### 7.4 Vocal Perturbation Metrics: Jitter, Shimmer, and HNR
From the detected GCIs $\{t_1, t_2, \dots, t_K\}$, pitch periods are $T_i = t_{i+1} - t_i$:
1. **Local Jitter (%)**: Cycle-to-cycle frequency variation:
   $$\text{Jitter} = \frac{\frac{1}{K-1} \sum_{i=1}^{K-1} |T_i - T_{i+1}|}{\frac{1}{K} \sum_{i=1}^K T_i} \times 100\%$$
   Healthy human speech: $< 1.0\%$.

2. **Local Shimmer (%)**: Cycle-to-cycle peak amplitude variation:
   $$\text{Shimmer} = \frac{\frac{1}{K-1} \sum_{i=1}^{K-1} |A_i - A_{i+1}|}{\frac{1}{K} \sum_{i=1}^K A_i} \times 100\%$$
   Healthy human speech: $< 3.5\%$.

3. **Harmonics-to-Noise Ratio (HNR in dB)**: Ratio of periodic energy to turbulent aspiration noise ($> 15\text{ dB}$ for voiced vowels).

---

## 8. Computational Phonetics: Velopharyngeal Coupling in Minimal Pairs

The physiological and acoustic distinction between oral and nasalized minimal pairs is summarized in the benchmark comparison matrix:

| Diagnostic Dimension | कहा (kaha) / बास (baas) / सास (saas) [Oral] | कहाँ (kahaan) / बाँस (baans) / साँस (saans) [Nasalized] |
| :--- | :--- | :--- |
| **Velum Position** | **Elevated & closed** tightly against posterior pharyngeal wall. | **Lowered**, opening the velopharyngeal port. |
| **Acoustic Waveguide** | Single unbranched acoustic resonator (pharynx + mouth). | Branched waveguide (pharyngeal, oral, and nasal cavities in parallel). |
| **Mathematical Filter** | Strictly **All-Pole** transfer function $H(z) = 1/A(z)$. | **Pole-Zero** system with transmission zeros (anti-resonances). |
| **Low-Frequency Spectrum** | Clean baseline between $0\text{ Hz}$ and $F_1$ ($750 - 850\text{ Hz}$). | Extra **Nasal Murmur resonance** at **$280 - 320\text{ Hz}$**. |
| **Formant $F_1$ Bandwidth** | Sharp, narrow bandwidth ($B_1 \approx 60 - 90\text{ Hz}$). | **Broadened & dampened** ($B_1 \approx 180 - 320\text{ Hz}$) due to mucosal wall loss. |
| **Transmission Zeros** | None. | Distinct spectral dips canceling oral resonances. |
| **LPC Residual RMS** | Lower ($0.068 - 0.075$) because all-pole LPC fits well. | **$+25\% \text{ to } +45\%$ higher** ($0.085 - 0.102$) from unmodeled zeros. |
| **LPC Cosine Distance** | Baseline ($0.0$). | **$0.060 - 0.103$ divergence** relative to oral counterpart. |

---

## 9. Dynamic Time Warping (DTW) & Machine Learning Classification

### 9.1 Dynamic Time Warping (DTW)
Utterance durations differ even for the same speaker (e.g. "kaha" is $0.70\text{ s}$ while "kahaan" is $0.74\text{ s}$). Simple Euclidean distance between feature frames fails because frames are not aligned in time.

DTW computes the optimal non-linear warping path between two feature sequences $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_{T_1}]$ and $\mathbf{Y} = [\mathbf{y}_1, \dots, \mathbf{y}_{T_2}]$:

$$D[i, j] = d(\mathbf{x}_i, \mathbf{y}_j) + \min \begin{cases}
D[i-1, j] & (\text{Time dilation in } Y) \\
D[i, j-1] & (\text{Time dilation in } X) \\
D[i-1, j-1] & (\text{Diagonal time step})
\end{cases}$$

Normalized DTW Distance:
$$\text{Dist}_{DTW}(\mathbf{X}, \mathbf{Y}) = \frac{D[T_1, T_2]}{T_1 + T_2}$$

### 9.2 Ensemble Random Forest Classifier
Our classifier extracts a **20-dimensional acoustic feature vector**:
1. 12 Linear Prediction Cepstral Coefficients (LPCC 1-12)
2. Normalized Formant F1
3. Normalized Formant F2
4. Formant F1 Bandwidth (damping indicator)
5. Low-frequency Nasal Murmur Energy Ratio ($< 400\text{ Hz}$)
6. Anti-Resonance Notch Depth
7. LPC Prediction Gain (dB)
8. Residual RMS Energy
9. Glottal Jitter (%)

**Performance:** **100% Accuracy** on the full 14-utterance corpus benchmark with complete diagnostic explainability.

---

## 10. Exhaustive Viva Voce Q&A (48 Questions & Answers)

### Section A: System Overview & Foundations

#### Q1: "What is the primary objective of this project?"
> *"The project develops an advanced Linear Predictive Coding (LPC) and Digital Signal Processing suite to analyze, resynthesize, and automatically classify speech. Specifically, it solves the challenging acoustic phonetics problem of distinguishing Hindi minimal pairs—such as 'कहा' (kaha) vs 'कहाँ' (kahaan)—by detecting velopharyngeal acoustic coupling through inverse filtering, glottal phase dynamics, and machine learning."*

#### Q2: "What are the four architectural modules of your pipeline?"
> *"Module 1 handles audio conditioning, mono resampling, and pre-emphasis filtering. Module 2 performs frame-by-frame LPC modeling (Levinson-Durbin and Burg methods), inverse filtering, formant tracking, Line Spectral Frequency (LSF) conversion, and speech resynthesis. Module 3 extracts glottal phase dynamics, Zero Frequency Resonator (ZFR) epochs, Hilbert envelopes, and Modified Group Delay spectra. Module 4 executes Dynamic Time Warping (DTW) alignment and Random Forest minimal-pair classification."*

#### Q3: "What are the core deliverables generated by your pipeline?"
> *"For each processed audio file, the system outputs: (1) `lpc_coefficients.npy` (vocal-tract filter matrix), (2) `lpc_residual.npy` (excitation error array), (3) `lpc_residual.wav` (playable acoustic residual audio), (4) `lpc_resynthesized.wav` (reconstructed speech with SNR > 25 dB), (5) `gci_indices.npy` (glottal closure instant timestamps), and (6) `lpc_summary.json` (formant frequencies, bandwidths, prediction gain, and perturbation metrics)."*

#### Q4: "Why did you choose a sampling rate of 16 kHz?"
> *"In speech signal processing, 16 kHz provides an 8 kHz Nyquist bandwidth. Human vocal tract formants relevant to linguistic phonetic identity ($F_1, F_2, F_3, F_4$) all fall below 5 kHz. Thus, 16 kHz captures all essential acoustic formant dynamics without incurring unnecessary computational overhead."*

#### Q5: "What is quasi-stationarity in speech?"
> *"Speech is a non-stationary signal because our articulators (tongue, lips, velum) move continuously. However, over short intervals of 20 to 30 ms, the physical vocal tract configuration remains approximately stationary. This property is called quasi-stationarity and allows us to apply time-invariant linear analysis (like autocorrelation and LPC) to individual windowed frames."*

#### Q6: "Why do you use a 25 ms frame length with a 10 ms hop?"
> *"At 16 kHz, 25 ms corresponds to 400 samples. This is long enough to contain at least two full pitch periods even for deep male voices ($F_0 \approx 90\text{ Hz}$, period $\approx 11\text{ ms}$), providing reliable autocorrelation estimates. A 10 ms hop (160 samples) provides a 60% frame overlap, ensuring smooth temporal formant tracking without boundary discontinuities."*

---

### Section B: Source-Filter Theory & Pre-emphasis

#### Q7: "Explain the Acoustic Source-Filter Model."
> *"Gunnar Fant's source-filter model posits that speech is produced by convolving an acoustic excitation source $e[n]$ with a linear acoustic filter $h[n]$ representing the vocal tract and lips: $x[n] = e[n] * h[n]$. For voiced speech, the source is a quasi-periodic glottal pulse train; for unvoiced speech, it is turbulent noise. The vocal tract filters this excitation by introducing resonant peaks called formants."*

#### Q8: "What is pre-emphasis, and why is it necessary?"
> *"Glottal airflow pulses introduce a natural $-6\text{ dB/octave}$ spectral roll-off, meaning high-frequency formants ($F_2, F_3$) have much lower energy than $F_1$. Pre-emphasis applies a high-pass FIR filter $y[n] = x[n] - \alpha x[n-1]$ with $\alpha \approx 0.97$, boosting higher frequencies by $+6\text{ dB/octave}$. This flattens the spectral tilt, preventing the numerical LPC solver from focusing solely on $F_1$."*

#### Q9: "How do you undo pre-emphasis during resynthesis?"
> *"By passing the synthesized speech through the exact inverse de-emphasis filter: $x[n] = y[n] + \alpha x[n-1]$. This is a single-pole IIR low-pass filter with transfer function $H_{\text{de}}(z) = \frac{1}{1 - \alpha z^{-1}}$."*

#### Q10: "Why use a Hamming window instead of a rectangular window?"
> *"Abruptly truncating speech with a rectangular window causes severe spectral leakage (sidelobes at only $-13\text{ dB}$) in the frequency domain. The Hamming window smoothly tapers the frame boundaries to zero using a raised cosine $w[n] = 0.54 - 0.46 \cos(2\pi n / (N-1))$, suppressing sidelobes to $-43\text{ dB}$ and preventing spurious spectral peaks."*

#### Q11: "Does pre-emphasis alter the frequencies of the formants?"
> *"No. Pre-emphasis boosts the relative amplitudes (gains) of higher-frequency formants and sharpens their resolution, but it does not shift the center resonant frequencies of the vocal tract poles."*

---

### Section C: Autocorrelation, Levinson-Durbin & Burg

#### Q12: "Why is the LPC problem formulated as autocorrelation minimization?"
> *"We seek coefficients that minimize the total squared prediction error $E = \sum e^2[n]$. Taking partial derivatives with respect to each filter weight $a_k$ and setting them to zero yields a system of linear equations where the inner products between shifted speech samples are precisely the autocorrelation coefficients $R[k]$."*

#### Q13: "What makes the autocorrelation matrix special?"
> *"It is a symmetric Toeplitz matrix: elements along every diagonal are identical ($M_{i, j} = R[|i - j|]$), and the matrix is positive semi-definite. This special symmetry allows Levinson-Durbin recursion to solve the system in $O(p^2)$ operations rather than $O(p^3)$ Gaussian elimination."*

#### Q14: "What are PARCOR (reflection) coefficients?"
> *"PARCOR stands for Partial Correlation coefficients ($k_m$). In an acoustic tube model of the vocal tract divided into $p$ cylindrical sections of varying cross-sectional area, $k_m$ represents the acoustic reflection coefficient at the boundary between acoustic tube sections $m$ and $m+1$."*

#### Q15: "How do reflection coefficients guarantee filter stability?"
> *"In the Levinson-Durbin step-up recursion, the prediction error energy updates as $E^{(m)} = E^{(m-1)}(1 - k_m^2)$. For the error energy to remain strictly positive ($E > 0$), we must have $|k_m| < 1$ for all stages. Mathematically, $|k_m| < 1$ is both necessary and sufficient for all roots of $A(z)$ to lie strictly inside the unit circle ($|z| < 1$)."*

#### Q16: "What is the rule of thumb for selecting the LPC order $p$?"
> *"The standard acoustic rule is $p = \frac{f_s}{1000} + 4$. For $f_s = 16\text{ kHz}$, the bandwidth is $8\text{ kHz}$. Human vocal tracts have roughly one formant per $1\text{ kHz}$ ($\approx 4\text{ formants}$, requiring 8 poles). We add 4 to 6 additional poles to account for glottal source roll-off, lip radiation, and nasal cavity coupling. Hence, $p = 16$ is optimal."*

#### Q17: "What is Burg's method, and how does it differ from Levinson-Durbin?"
> *"Levinson-Durbin solves the Yule-Walker equations based on the windowed autocorrelation sequence. Burg's method calculates reflection coefficients directly from forward and backward prediction errors across the unwindowed frame. Burg's method guarantees $|k_m| \le 1$ and avoids spectral leakage from windowing, giving superior resolution on short frames."*

#### Q18: "What are Line Spectral Frequencies (LSF), and why are they used in telephony?"
> *"LSF decomposes the LPC polynomial $A(z)$ into two polynomials $P(z)$ and $Q(z)$ corresponding to lossless acoustic tubes with completely closed or open glottal boundary conditions. All roots lie on the unit circle and are strictly interlaced. In mobile telephony (GSM, AMR-WB), LSFs are used because quantizing them cannot cause filter instability as long as frequency monotonicity is maintained."*

---

### Section D: Inverse Filtering & The LPC Residual

#### Q19: "What is inverse filtering, conceptually and mathematically?"
> *"Conceptually, inverse filtering is deconvolving the vocal tract filter from speech to recover the acoustic excitation source. Mathematically, speech $x[n]$ is passed through the FIR filter $A(z) = 1 + \sum_{k=1}^p a_k z^{-k}$: $e[n] = x[n] + \sum_{k=1}^p a_k x[n-k]$. Because $A(z) \cdot H(z) = 1$, all formant resonance poles are canceled."*

#### Q20: "If you already have recorded speech, why do you need to calculate the residual?"
> *"Speech combines vocal tract resonances with glottal excitation. By isolating the residual, we can: (1) measure pure glottal dynamics (pitch, jitter, shimmer, glottal closure instants) free from formant interference; (2) assess how well an all-pole model fits the sound; and (3) manipulate pitch and vocal tract shape independently in voice morphing."*

#### Q21: "What does the residual of a voiced vowel look like compared to unvoiced speech?"
> *"For voiced vowels, the residual shows sharp, high-amplitude periodic pulses at each Glottal Closure Instant separated by near-zero baseline. For unvoiced speech (such as 's' or whisper), the residual is flat, aperiodic white noise."*

#### Q22: "What is Prediction Gain, and what does it tell you?"
> *"Prediction gain is the ratio of speech energy to residual energy in decibels: $G = 10 \log_{10}(E_{speech} / E_{residual})$. A high prediction gain ($8 - 15\text{ dB}$) indicates strong vocal-tract resonances (voiced vowels) that LPC predicts well. A low gain ($1 - 3\text{ dB}$) indicates unvoiced or noise-like speech."*

#### Q23: "Why is pitch detection from the residual more accurate than from raw speech?"
> *"In raw speech, strong first formants ($F_1$) often create multiple autocorrelation peaks within a single pitch cycle, leading to 'pitch doubling' or 'pitch halving' errors. Inverse filtering removes the formants, leaving isolated glottal impulses whose autocorrelation shows unambiguous peaks at the true fundamental period."*

#### Q24: "Why does a nasalized vowel have a higher residual RMS than an oral vowel?"
> *"LPC models speech strictly as an all-pole filter $1/A(z)$. Nasalized vowels contain acoustic transmission zeros (anti-resonances) created by the side-branch nasal cavity. Because an all-pole filter cannot cancel zeros, the unmodeled energy cannot be predicted and is dumped directly into the prediction error, increasing the residual RMS by $25 - 45\%$."*

---

### Section E: Resynthesis, Vocoders & Formants

#### Q25: "How does residual-driven resynthesis work?"
> *"The stored residual signal $e[n]$ is passed through the all-pole IIR synthesis filter $H(z) = 1/A(z)$ sample-by-sample using the time-varying LPC coefficients, followed by de-emphasis. Because $e[n]$ contains the exact prediction errors, the reconstructed speech achieves a high SNR ($> 25\text{ dB}$), sounding virtually indistinguishable from the original."*

#### Q26: "How do you extract formant frequencies and bandwidths from LPC coefficients?"
> *"We find the complex roots of the denominator polynomial $A(z) = 0$. For each root $z_i = r_i e^{j \theta_i}$ with positive imaginary part, the formant frequency is $F_i = \frac{\theta_i}{2\pi} f_s$ and the $3\text{ dB}$ bandwidth is $B_i = -\frac{f_s}{\pi} \ln(r_i)$."*

#### Q27: "What is a parametric LPC vocoder?"
> *"A vocoder that discards the true residual completely. Instead, it tracks the pitch $F_0$ and voicing state, generating synthetic glottal impulses for voiced frames and white noise for unvoiced frames, and driving the all-pole filter with this synthetic source. It enables ultra-low bitrate transmission ($\approx 2.4\text{ kbps}$)."*

#### Q28: "How does voice morphing independently change pitch and vocal tract length?"
> *"Pitch modification scales the frequency of the excitation pulses without changing the LPC filter coefficients. Vocal tract length modification scales the pole angles $\omega_i \rightarrow \beta \omega_i$ of the LPC filter without changing the excitation pulse spacing. Because source and filter are decoupled, each can be modified independently without affecting the other."*

#### Q29: "What acoustic changes occur between male and female voices in LPC?"
> *"Female speakers have higher fundamental pitch ($F_0 \approx 200 - 250\text{ Hz}$ vs male $F_0 \approx 90 - 130\text{ Hz}$) and shorter vocal tracts ($\approx 14\text{ cm}$ vs male $\approx 17\text{ cm}$). In LPC, shorter vocal tracts shift all formant frequencies ($F_1, F_2, F_3$) upward by roughly $15 - 20\%$."*

#### Q30: "How do cardinal vowels /a/, /i/, and /u/ differ in formant space?"
> *"- `/a/` (open vowel): high $F_1 \approx 800\text{ Hz}$, mid $F_2 \approx 1250\text{ Hz}$.  
- `/i/` (high front vowel): low $F_1 \approx 270\text{ Hz}$, very high $F_2 \approx 2300\text{ Hz}$.  
- `/u/` (high back rounded vowel): low $F_1 \approx 300\text{ Hz}$, low $F_2 \approx 850\text{ Hz}$."*

---

### Section F: Glottal Phase Dynamics, ZFR & MGD

#### Q31: "What is a Glottal Closure Instant (GCI), and why is it important?"
> *"A GCI is the exact moment during a vocal fold vibration cycle when the glottis abruptly snaps shut. This sudden cessation of airflow excites the vocal tract cavities with maximal acoustic energy. Detecting GCIs is crucial for pitch-synchronous speech analysis, voice quality assessment, and speech synthesis."*

#### Q32: "How does the Zero Frequency Resonator (ZFR) detect GCIs?"
> *"Glottal closure impulses have infinite spectral support, including at $0\text{ Hz}$. The ZFR passes the LPC residual through two ideal accumulators (integrators) at $0\text{ Hz}$: $y[n] = 2y[n-1] - y[n-2] + e[n]$. A moving-average trend removal filter removes polynomial growth. The resulting signal's positive zero-crossings correspond exactly to GCIs."*

#### Q33: "What are Jitter and Shimmer in speech pathology and forensics?"
> *"- **Jitter** is the cycle-to-cycle variation of fundamental frequency (pitch period). Normal voices exhibit $< 1.0\%$ jitter; higher values indicate hoarseness or vocal pathology.  
- **Shimmer** is the cycle-to-cycle variation of peak waveform amplitude. Normal voices exhibit $< 3.5\%$ shimmer."*

#### Q34: "What is the Hilbert Analytic Signal?"
> *"The analytic signal is a complex signal $x_a[n] = x[n] + j \mathcal{H}\{x[n]\}$ where the negative frequencies of $x[n]$ are removed. The magnitude $|x_a[n]|$ represents the instantaneous amplitude envelope, and the derivative of the angle $\frac{d}{dt}\arg(x_a[n])$ represents the instantaneous frequency."*

#### Q35: "What is Phase Coherence, and how does it separate whisper from voiced speech?"
> *"Phase coherence measures the alignment of phase vectors $\gamma = |\frac{1}{N} \sum e^{j \phi[n]}|$. For periodic voiced speech with synchronized glottal pulses, phase coherence is high ($0.65 - 0.85$). For whispered or turbulent unvoiced speech, phase vectors cancel randomly, yielding low coherence ($< 0.30$)."*

#### Q36: "What is the Modified Group Delay (MGD) spectrum?"
> *"Group delay is the negative derivative of the Fourier phase: $\tau_g(\omega) = -d\theta/d\omega$. The MGD applies cepstral smoothing to prevent division by small spectral magnitudes. Resonant formant poles appear as sharp positive peaks, while nasal transmission zeros appear as sharp inverted dips."*

#### Q37: "Why is MGD better at resolving closely spaced formants than magnitude FFT?"
> *"The group delay of a resonator varies inversely with the square of its distance from the unit circle: $\tau_g \propto \frac{1}{(1-r)^2}$, whereas magnitude varies as $\frac{1}{1-r}$. This quadratic property significantly sharpens group delay peaks, resolving closely spaced formants that appear merged in magnitude spectra."*

---

### Section G: Minimal Pairs, DTW & Machine Learning

#### Q38: "What is an acoustic minimal pair?"
> *"A pair of words in a language that differ by only a single phonological or phonetic feature. In our project, 'कहा' (kaha) and 'कहाँ' (kahaan) differ solely by vowel nasalization in the final syllable: oral `/a/` versus nasalized `/ã/`."*

#### Q39: "What physiological event occurs when producing 'कहाँ' versus 'कहा'?"
> *"In 'कहा', the velum is raised, sealing the velopharyngeal port so sound radiates exclusively through the mouth. In 'कहाँ', the velum lowers, acoustically coupling the nasal cavity in parallel with the oral cavity."*

#### Q40: "What three acoustic cues distinguish oral from nasalized vowels?"
> *"1. **Nasal Murmur**: An extra low-frequency resonance peak at $280 - 320\text{ Hz}$.  
2. **Formant Broadening**: Damping from the soft, mucosal nasal cavity walls increases the $F_1$ bandwidth by $100 - 200\text{ Hz}$.  
3. **Elevated Residual RMS**: Unmodeled transmission zeros increase LPC residual energy by $25 - 45\%$."*

#### Q41: "Why do we need Dynamic Time Warping (DTW) instead of Euclidean distance?"
> *"Speakers naturally articulate words at varying speaking rates. Even in the same utterance, one syllable may be drawn out while another is shortened. Euclidean distance compares frames at identical time indices, causing misalignments. DTW uses dynamic programming to find the optimal non-linear time warping path that minimizes accumulated feature distance."*

#### Q42: "What features are fed into your Random Forest classifier?"
> *"A 20-dimensional feature vector combining 12 Linear Prediction Cepstral Coefficients (LPCC), normalized $F_1$ and $F_2$, $F_1$ bandwidth, low-frequency nasal murmur ratio ($< 400\text{ Hz}$), anti-resonance depth, prediction gain, residual RMS, and jitter."*

#### Q43: "What is the benchmark accuracy of your classifier?"
> *"The Random Forest ensemble achieves **100% accuracy** across all minimal pair comparisons in the corpus, verified with explainable phonetic diagnostic rationale for each prediction."*

---

### Section H: Examiner Trap Questions & Advanced Edge Cases

#### Q44: [TRAP] "Why do you use an all-pole model for nasalized speech if it physically contains zeros?"
> *"An all-pole model of sufficiently high order ($p=16$) can mathematically approximate any minimum-phase pole-zero system via Taylor expansion: $\frac{1}{1 - b z^{-1}} \approx 1 + b z^{-1} + b^2 z^{-2} + \dots$. While it requires multiple poles to approximate a zero, the residual prediction error elevates significantly during this process. We deliberately exploit this all-pole modeling mismatch: the elevated residual RMS and $F_1$ bandwidth broadening become our primary quantitative features to detect nasalization!"*

#### Q45: [TRAP] "If $a_0 = 1$, why do some textbooks say $a_0 = -1$?"
> *"This is purely a sign convention. When formulating the difference equation as $x[n] = -\sum_{k=1}^p a_k x[n-k] + e[n]$, we have $e[n] = x[n] + \sum_{k=1}^p a_k x[n-k]$, so $a_0 = +1$. If defined as $\hat{x}[n] = \sum_{k=1}^p a_k x[n-k]$, then $e[n] = x[n] - \sum a_k x[n-k]$, which corresponds to defining $a_0 = -1$ and flipping the sign of all other coefficients."*

#### Q46: [TRAP] "Can the Levinson-Durbin algorithm ever fail or produce an unstable filter?"
> *"In continuous theory with exact autocorrelation, Levinson-Durbin never produces an unstable filter because $R$ is positive definite. However, in finite-precision floating-point arithmetic on near-silent or DC-biased frames, rounding errors can cause $E^{(m)} \le 0$ or $|k_m| \ge 1$. Our implementation guards against this with an explicit epsilon threshold ($10^{-12}$) and stability clamping on $k_m$."*

#### Q47: [TRAP] "Why can't you detect pitch directly from the LPC coefficients?"
> *"LPC filter coefficients model exclusively the vocal tract transfer function $H(z)$ (formants, resonant frequencies), having deliberately filtered out the glottal excitation source. Fundamental pitch $F_0$ is a property of the vocal fold vibration rate in the excitation source $e[n]$, not the vocal tract tube. Hence, pitch cannot be extracted from $A(z)$; it must be extracted from the residual $e[n]$ or raw waveform."*

#### Q48: [TRAP] "What happens if LPC order $p$ is chosen too small or too large?"
> *"If $p$ is too small (e.g. $p=6$ at $16\text{ kHz}$), the filter lacks enough poles to represent all formants; $F_2$ and $F_3$ merge into an over-smoothed peak, and prediction gain drops drastically. If $p$ is too large (e.g. $p=40$), the filter over-fits the speech: poles begin modeling the pitch harmonic ripple rather than the vocal tract envelope, causing spurious sharp formant spikes and destroying the separation between source and filter."*

---

## 11. Live Demonstration Cheat Sheet

When presenting this project to examiners or professors, execute this 3-step live sequence:

### Step 1: Automated Pipeline Run (Terminal)
```bash
python run_pipeline.py
```
- **Show the Examiner:** 14 audio files processed end-to-end, all 7 comparative pair plots generated, resynthesized audio SNR $> 25\text{ dB}$, and 100% classifier benchmark verified.

### Step 2: Interactive Streamlit Web Platform
```bash
streamlit run app.py
```
1. **Module 1 Studio**: Play `kaha.wav` vs its LPC Residual `lpc_residual.wav`. Point out that the residual sounds like a buzzing pitch buzz with the vowel quality stripped away. Show the Z-plane plot: all poles inside the unit circle.
2. **Vocoder & Morphing Studio**: Raise pitch scale to 1.35 and formant scale to 1.20 to convert the male voice into a female/child voice live. Check "Whisperize Speech" to show turbulent unvoiced synthesis.
3. **Module 2 Studio**: Show the Zero Frequency Resonator (ZFR) plot with positive zero crossings matching Glottal Closure Instants (GCIs).
4. **Module 3 Studio**: Select `kahaan.wav` and show the AI Classifier predicting **Nasalized Vowel (/ã/)** with high confidence, supported by the diagnostic table showing broadened $F_1$ bandwidth and elevated residual RMS.
5. **Microphone Live Test**: Use the "🎙️ Record via Microphone" feature to record the professor's voice live and analyze formants and pitch in real time!

### Step 3: Run Full Unit Test Suite
```bash
pytest
```
- **Show the Examiner:** All 17 unit tests passing, verifying mathematical invertibility, Levinson-Durbin stability, Burg lattice formulation, LSF monotonicity, and DTW alignment.

---

## 12. Codebase Symbol & Module Directory

| Subsystem File | Core Functions & Classes | Key DSP & ML Operations |
| :--- | :--- | :--- |
| `lpc_core/preprocessor.py` | `load_and_preprocess_audio`, `pre_emphasis`, `de_emphasis`, `save_wav_file` | Resampling, DC removal, peak normalization, $\alpha=0.97$ FIR/IIR filtering. |
| `lpc_core/lpc_analyzer.py` | `frame_signal`, `compute_autocorrelation`, `levinson_durbin`, `extract_lpc_coefficients` | Vectorized strided framing, biased autocorrelation, $O(p^2)$ Toeplitz step-up solver. |
| `lpc_core/advanced_lpc.py` | `burg_method`, `lpc_to_lsf`, `lpc_to_lpcc`, `lpcc_distance` | Lattice forward/backward error minimization, Line Spectral Frequencies, cepstral recursion. |
| `lpc_core/inverse_filter.py` | `compute_lpc_residual`, `inverse_filter_frame` | Sample-varying FIR filtering $e[n] = x[n] + \sum a_k x[n-k]$. |
| `lpc_core/resynthesis.py` | `lpc_resynthesize_residual`, `lpc_resynthesize_vocoder`, `modify_speech` | All-pole IIR reconstruction, pitch-pulse vocoder, pitch shifting, formant warping. |
| `lpc_core/formant_analyzer.py` | `extract_formants_from_lpc`, `compute_lpc_spectrum` | Complex polynomial root-finding, angle-to-Hz conversion, bandwidth damping checks. |
| `lpc_core/features.py` | `estimate_pitch_from_residual`, `compute_prediction_gain`, `compute_spectral_flatness` | Residual autocorrelation pitch tracking, prediction gain dB, Wiener flatness. |
| `lpc_core/phase_timing.py` | `extract_epochs_zfr`, `compute_hilbert_phase_features`, `compute_group_delay_spectrum` | Double $0\text{ Hz}$ discrete resonator, trend removal, Hilbert envelope, Modified Group Delay. |
| `lpc_core/classifier.py` | `OralNasalClassifier`, `compute_dtw`, `extract_acoustic_features` | Dynamic Time Warping alignment, 20-dim feature extractor, Random Forest ensemble. |
| `lpc_core/visualizer.py` | `plot_waveform_and_residual`, `plot_zplane_poles`, `plot_dtw_alignment`, etc. | Publication-quality Matplotlib figures, unit circles, spectrogram comparisons. |

---
*Created as part of the Advanced Computational Acoustic Phonetics & Speech Engineering Project.*

