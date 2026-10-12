# 🎙️ LPC Project: Complete Explanation & Viva Guide
### Member 2: Linear Predictive Coding (LPC) and Inverse Filtering

---

## Table of Contents
1. [The Big Picture: What is our Team Doing?](#1-the-big-picture)
2. [What is Member 2's Exact Responsibility?](#2-what-is-member-2s-responsibility)
3. [How Does Human Voice Work? (Source-Filter Model)](#3-how-human-voice-works)
4. [What is LPC? (Explained in Plain English)](#4-what-is-lpc)
5. [Why Predict Samples? Are We Trying to Predict the Future?](#5-why-predict-samples)
6. [Why Calculate the Difference (Error)? Why is it Useful?](#6-why-calculate-the-error)
7. [What is Inverse Filtering?](#7-what-is-inverse-filtering)
8. [Why Do We Divide Audio into Short Frames (25 ms)?](#8-why-short-frames)
9. [What Does "LPC Order 16" Mean?](#9-what-does-lpc-order-16-mean)
10. [Step-by-Step Mathematical Flow with Real Numbers](#10-step-by-step-mathematical-flow)
11. [What Files Do We Output and Pass to Member 3?](#11-what-files-do-we-output)
12. [How Does This Differentiate Words like 'कहा' (kaha) and 'कहाँ' (kahaan)?](#12-differentiating-kaha-vs-kahaan)
13. [Ready-Made Answers to Questions Your Professor Will Ask](#13-viva-questions-and-answers)

---

## 1. The Big Picture: What is our Team Doing?

Imagine two Hindi words that sound almost identical to the human ear:
- **कहा (kaha)**: says / said (oral vowel in the 2nd syllable)
- **कहाँ (kahaan)**: where (nasalized vowel in the 2nd syllable)

Your 4-person team is asking:  
> *"Can a computer mathematically detect the acoustic differences between these words, even when human ears might confuse them?"*

Here is how the 4 team members divide the work:
```
[Member 1: Recording]
  Records audio -> Saves hello.wav, kaha.wav, kahaan.wav
         │
         ▼
[Member 2: YOU - LPC Analysis]  <-- YOUR PART
  1. Estimates the vocal-tract filter shape (LPC Coefficients)
  2. Strips away the vocal-tract effect using Inverse Filtering
  3. Extracts the LPC Residual (the raw voice excitation signal)
         │
         ▼
[Member 3: Phase & Timing]
  Analyzes temporal timing, phase coherence, and glottal pulse shapes on the residual
         │
         ▼
[Member 4: Comparison & Classification]
  Compares features using algorithms/machine learning to tell the words apart!
```

---

## 2. What is Member 2's Responsibility?

Your job is the mathematical bridge:
1. **Take the speech audio** from Member 1.
2. **Estimate the vocal-tract resonances (formants)** using Linear Predictive Coding (LPC).
3. **Perform inverse filtering** to calculate the **LPC residual**.
4. **Save and deliver three files**:
   - `lpc_coefficients.npy` (the filter coefficients describing the vocal tract)
   - `lpc_residual.npy` (the numerical prediction-error array)
   - `lpc_residual.wav` (the audio of the residual so anyone can listen to it)

---

## 3. How Human Voice Works: The Source-Filter Model

When you speak any word (like *"kaha"*), your body produces sound in two distinct stages:

```
[ Lungs + Vocal Cords ]       ──▶      [ Throat + Mouth + Nose ]     ──▶   [ Microphone ]
     (SOURCE)                                   (FILTER)                      (RECORDING)
Produces buzz / periodic pulse          Shapes the sound (resonances)        Combined sound
or unvoiced turbulent noise             amplifies specific frequencies       x[n]
e[n]                                    H(z) = 1 / A(z)
```

In signal processing, this is written as:
$$\text{Speech Signal } x[n] = \text{Glottal Source } e[n] * \text{Vocal-Tract Filter } h[n]$$

- **The Source ($e[n]$)**: The periodic buzz of your vocal folds (or aspiration/noise).
- **The Filter ($h[n]$)**: The tube formed by your throat, tongue, mouth, and nasal cavity. Depending on your mouth shape, it amplifies certain frequencies (called **Formants**: F1, F2, F3).
- **The Microphone**: Records both mixed together into a single waveform $x[n]$.

**The Problem:** You cannot stick a microphone inside your throat to record the clean source, and another inside your mouth to record the clean filter. You only have the combined microphone recording.  
**The Solution:** LPC mathematically separates them!

---

## 4. What is LPC? (Explained in Plain English)

**LPC** stands for **Linear Predictive Coding**:
- **Linear**: We only use multiplication and addition (no powers like $x^2$ or square roots).
- **Predictive**: It uses previous sound samples ($x[n-1], x[n-2], \dots$) to predict the current sample ($x[n]$).
- **Coding**: It represents complex speech using a compact set of numbers (coefficients).

### A Simple Numerical Example:
Suppose a tiny piece of speech has these numerical values:
- Sample at time $n-1$: $0.80$
- Sample at time $n-2$: $0.60$

LPC uses weights ($a_1, a_2$) to predict what the current sample $\hat{x}[n]$ should be:
$$\hat{x}[n] = - (a_1 \cdot x[n-1] + a_2 \cdot x[n-2])$$

Suppose LPC calculates the prediction to be $\mathbf{0.70}$.  
Then we look at the real recording: the actual sample $x[n]$ was $\mathbf{0.85}$.

The difference is:
$$\text{Difference } e[n] = x[n] - \hat{x}[n] = 0.85 - 0.70 = \mathbf{0.15}$$

This difference is called the **Prediction Error** or the **LPC Residual**.

---

## 5. Why Predict Samples? Are We Trying to Predict the Future?

**NO! Absolutely not.**  
You are not doing weather forecasting or stock market prediction. You are analyzing an audio file that has *already been recorded*.

### Why use prediction then?
Because nearby sound samples in speech are correlated *specifically because of the resonance of the vocal tract*.  
- When your mouth holds the shape of an vowel like "/a/", that vocal tract acts like a resonance chamber (like blowing across the top of a bottle).
- That resonance causes past samples to echo and influence upcoming samples in a very specific mathematical pattern.
- By finding the weights ($a_k$) that best predict the signal from past samples, **we are actually finding the exact frequency response of your vocal tract!**

Prediction is just the *mathematical tool* used to model the vocal-tract filter.

---

## 6. Why Calculate the Difference (Error)? Why is it Useful?

This was your biggest question:  
> *"what we will get if we cal diff , why it was usefull,,, why r we predecting error,,,,"*

Here is the key insight:

```
[ Actual Recording x[n] ]   =   [ Vocal-Tract Resonances ]   +   [ Vocal-Cord Excitation ]
                                   (Predicted by LPC)              (NOT predicted by LPC)
```

1. **The LPC model predicts the vocal tract.** (The smooth resonance curve of the mouth and throat).
2. **What the LPC model CANNOT predict is the sudden impulse when the vocal cords snap shut.** That sudden pulse is new energy entering the system from the lungs!
3. Therefore, when you subtract the predicted speech from the actual speech:
   $$\text{Difference } e[n] = \text{Actual Speech } x[n] - \text{Predicted Speech } \hat{x}[n]$$
   **The vocal-tract shaping is cancelled out, and what remains is the vocal-cord excitation pulse train (the residual)!**

### Why is this useful for distinguishing words?
- In ordinary speech waveforms, the vocal-tract resonances obscure the fine timing and shape of the vocal-cord pulses.
- In the **residual**, the formant resonance "rings" are gone!
- This exposes:
  * Clean glottal pulse timing (pitch period $T_0$)
  * Breathy aspiration noise
  * Nasal coupling and extra acoustic energy leakage (vital for telling **कहा** apart from **कहाँ**!)

Member 3 and Member 4 can now inspect this excitation signal directly!

---

## 7. What is Inverse Filtering?

You know that the vocal tract acts as a filter:
$$H(z) = \frac{1}{A(z)} = \frac{1}{1 + a_1 z^{-1} + a_2 z^{-2} + \dots + a_p z^{-p}}$$
This filter amplifies certain frequencies (the vowel formants).

**Inverse Filtering** simply flips that filter upside down:
$$A(z) = 1 + a_1 z^{-1} + a_2 z^{-2} + \dots + a_p z^{-p}$$

When you pass the recorded speech $x[n]$ through $A(z)$:
$$e[n] = x[n] + a_1 x[n-1] + a_2 x[n-2] + \dots + a_p x[n-p]$$

In Python:
```python
residual = scipy.signal.lfilter(lpc_coefficients, [1.0], speech_frame)
```
- The numerator is the LPC coefficients array `[1.0, a1, a2, ..., ap]`.
- The denominator is `[1.0]`.
- This cancels out the vocal tract poles and outputs the prediction error $e[n]$.

---

## 8. Why Do We Divide Audio into Short Frames (25 ms)?

When you speak, your tongue, jaw, and vocal cords move continuously:
- You make a consonant sound like "/k/" (takes ~40 ms)
- Then you open your mouth for vowel "/a/" (takes ~200 ms)
- Then you breathe for "/h/" (takes ~80 ms)

If you calculated one single LPC filter for the entire 1-second recording, it would be an average of your tongue in 5 different places at once, which is meaningless!

Therefore, we use **Quasi-Stationarity**:
- For 20 to 30 milliseconds (0.025 seconds), your mouth doesn't have time to move significantly.
- During that tiny window (400 samples at 16 kHz), your vocal tract shape is almost constant.
- We slide this window forward by 10 milliseconds (160 samples) at a time (**Hop Length**).
- This produces a separate set of LPC coefficients for every frame, tracking your vocal tract dynamically!

---

## 9. What Does "LPC Order 16" Mean?

The order $p = 16$ means the computer uses the **16 previous samples** to predict the current sample:
$$x[n] \approx - \sum_{k=1}^{16} a_k x[n-k]$$

### Why 16? (The Engineering Rule of Thumb)
At a sampling rate of $f_s = 16,000 \text{ Hz}$, the audio contains frequencies up to $f_s / 2 = 8,000 \text{ Hz}$ (Nyquist frequency).

In human acoustics:
- The human vocal tract averages 17 cm in length.
- It produces roughly **1 formant resonance per 1,000 Hz**.
- In 8,000 Hz bandwidth, there are approximately 4 to 5 formants.
- Each formant resonance requires a complex conjugate pole pair (2 filter poles).
- 4 to 5 formants $\times 2 = 8$ to $10$ poles.
- We add 4 to 6 extra poles to model glottal source spectral tilt, lip radiation, and nasal resonances:
$$p = \frac{f_s}{1000} + 4 = 16 + 4 \approx 16 \text{ to } 20$$
Order 16 is the standard setting for 16 kHz speech processing.

---

## 10. Step-by-Step Mathematical Flow with Real Numbers

Let's trace one frame through our code:

```
[Step 1: Input Audio]
  Sample rate fs = 16,000 Hz. Audio normalized between -1.0 and +1.0.

[Step 2: Pre-Emphasis]
  y[n] = x[n] - 0.97 * x[n-1]
  (Flattens natural -6dB/octave glottal drop, boosting high frequencies)

[Step 3: Framing & Windowing]
  Frame length = 400 samples (25 ms)
  Hop length = 160 samples (10 ms)
  Multiplied by Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / 399)

[Step 4: Autocorrelation]
  R[k] = sum( w_frame[n] * w_frame[n+k] ) for k = 0, 1, ..., 16
  R[0] is the total energy of the frame.

[Step 5: Levinson-Durbin Recursion]
  Solves Toeplitz system R * a = -r in O(p^2) steps.
  Output: a = [1.0, a1, a2, ..., a16] (17 numbers)

[Step 6: Inverse Filtering]
  e[n] = x[n] + a1*x[n-1] + a2*x[n-2] + ... + a16*x[n-16]
  Output: e[n] (the residual array)
```

---

## 11. What Files Do We Output and Pass to Member 3?

| File Name | Data Type | Dimensions | What it Contains |
| :--- | :--- | :--- | :--- |
| `lpc_coefficients.npy` | Float64 Array | `(num_frames, 17)` | The estimated vocal-tract filter weights for every 25ms frame. |
| `lpc_residual.npy` | Float64 Array | `(num_samples,)` | The exact numerical prediction-error signal. Matches audio length 1-to-1. |
| `lpc_residual.wav` | 16-bit PCM Audio | 16,000 Hz | The residual saved as a playable WAV file so you and your team can listen to it. |
| `lpc_summary.json` | JSON Object | — | Formant frequencies (F1, F2, F3), prediction gain in dB, residual RMS energy. |

---

## 12. Differentiating 'कहा' (kaha) vs 'कहाँ' (kahaan)

Here is why your LPC analysis is critical for distinguishing these two Hindi words:

| Acoustic Feature | कहा (kaha) - Oral Vowel | कहाँ (kahaan) - Nasalized Vowel |
| :--- | :--- | :--- |
| **Vocal Tract Configuration** | Velum (soft palate) is **closed**. Sound exits only through mouth. | Velum is **lowered**. Sound exits through both mouth and nasal cavity. |
| **LPC Spectral Envelope** | Sharp, clear oral formants (F1 ~ 800 Hz, F2 ~ 1250 Hz). | Extra **nasal resonance** (~ 280-320 Hz and ~ 1050 Hz). F1 is broadened and dampened. |
| **LPC Coefficients** | Different polynomial roots reflect oral cavity geometry. | Shifted roots reflect nasal-oral acoustic coupling. |
| **LPC Residual Energy** | Lower residual energy because the all-pole model fits oral vowel well. | Higher residual energy because all-pole LPC cannot perfectly cancel nasal zeros (anti-resonances). |
| **Residual Excitation Waveform** | Periodic sharp glottal spikes. | Modified glottal pulses due to nasal acoustic impedance load. |

When Member 4 runs classification, the **LPC Cosine Distance** and **Residual RMS Ratio** provide quantitative proof that the computer can detect the acoustic difference!

---

## 13. Ready-Made Answers to Questions Your Professor Will Ask

### Q1: "What was your specific role in this group project?"
> *"I was Member 2, responsible for Linear Predictive Coding (LPC) and Inverse Filtering. I took the speech recordings from Member 1, estimated the all-pole vocal-tract filter coefficients frame-by-frame using the Levinson-Durbin algorithm, and applied inverse filtering to extract the LPC residual signal. I then saved `lpc_coefficients.npy`, `lpc_residual.npy`, and `lpc_residual.wav` for Member 3 and Member 4."*

### Q2: "Why are you predicting samples if you already have the recorded audio?"
> *"We do not predict samples to forecast future speech. Prediction is simply the mathematical formulation used in LPC to model the resonance of the vocal tract. Because speech samples are correlated due to vocal-tract resonance, fitting a linear predictor allows us to calculate the transfer function of the vocal tract. The difference between actual speech and predicted speech is the prediction error, which gives us the excitation source."*

### Q3: "What is the physical meaning of the LPC Residual?"
> *"According to the source-filter model, speech is the convolution of the glottal excitation source and the vocal-tract filter. The LPC filter models the vocal tract. When we subtract the model's prediction from the speech signal (inverse filtering), we cancel out the vocal-tract resonances. The residual represents the acoustic source excitation—namely, the glottal pulses for voiced sounds or aspiration noise for unvoiced sounds."*

### Q4: "Why do we divide the speech into 25 ms frames with 10 ms hop?"
> *"Because speech is non-stationary; our articulators constantly move as we speak. Over 25 ms, however, the vocal tract geometry is approximately stationary (quasi-stationarity). A 10 ms hop provides smooth overlap, preventing boundary discontinuities when tracking changing formants over time."*

### Q5: "Why did you use LPC order 16?"
> *"At our 16 kHz sampling rate, the bandwidth is 8 kHz. With human vocal tracts averaging ~1 formant per 1 kHz, we expect roughly 4 to 5 formants, requiring 8 to 10 poles. We add 6 extra poles to account for glottal source roll-off, lip radiation, and nasal coupling. Order 16 is the standard rule of thumb: p = fs/1000 + 4."*

### Q6: "How does the Levinson-Durbin algorithm work?"
> *"Directly solving the Yule-Walker equations by matrix inversion takes O(p^3) operations. Because the autocorrelation matrix is Toeplitz (symmetric with constant diagonals), the Levinson-Durbin algorithm solves it recursively in O(p^2) operations, calculating both the LPC filter coefficients and the PARCOR reflection coefficients."*
