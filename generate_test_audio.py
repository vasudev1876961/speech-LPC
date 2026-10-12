"""
generate_test_audio.py
----------------------
Generates realistic acoustic speech audio samples for testing the LPC pipeline:
1. 'kaha.wav'   (Hindi minimal pair 1: oral vowel /a/ in 2nd syllable)
2. 'kahaan.wav' (Hindi minimal pair 2: nasalized vowel /ã/ in 2nd syllable)
3. 'hello.wav'  (Standard greeting with voiced vowels /ɛ/ and /oʊ/)

Uses the classic source-filter acoustic model:
- Glottal pulse excitation train (F0 ~ 130 Hz with natural intonation contour)
- Vocal tract resonant bandpass/all-pole filtering (formants F1, F2, F3)
- Nasalization pole-zero pairs for 'kahaan'
- Aspiration noise for /h/
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import lfilter, butter, iirnotch

def create_glottal_pulse_train(duration, fs=16000, f0=130.0, jitter=0.01):
    """
    Synthesize Rosenberg glottal pulse train for natural voiced speech source.
    """
    num_samples = int(duration * fs)
    source = np.zeros(num_samples)
    
    # Intonation pitch curve (gentle declination)
    t = np.linspace(0, duration, num_samples)
    pitch_contour = f0 * (1.0 - 0.15 * t / duration)
    
    phase = 0.0
    for i in range(num_samples):
        current_f0 = pitch_contour[i] * (1.0 + jitter * (np.random.rand() - 0.5))
        period_samples = fs / current_f0
        
        # Rosenberg glottal pulse shape
        n_mod = phase % period_samples
        tn = n_mod / period_samples
        if tn < 0.4:
            # Opening phase: smooth quadratic rise
            pulse = 0.5 * (1 - np.cos(np.pi * tn / 0.4))
        elif tn < 0.6:
            # Closing phase: steep cosine fall
            pulse = np.cos(np.pi * (tn - 0.4) / (2 * 0.2))
        else:
            # Closed phase
            pulse = 0.0
            
        source[i] = pulse
        phase += 1.0
        
    # Glottal derivative to simulate radiation at the lips (+6 dB/octave)
    source_diff = np.diff(source, prepend=source[0])
    return source_diff

def formant_filter(signal, fs, freq, bandwidth, gain=1.0):
    """
    2nd-order digital resonator representing a vocal tract formant resonance.
    """
    omega = 2.0 * np.pi * freq / fs
    r = np.exp(-np.pi * bandwidth / fs)
    b = [gain * (1.0 - r * np.cos(omega))]
    a = [1.0, -2.0 * r * np.cos(omega), r * r]
    return lfilter(b, a, signal)

def synthesize_kaha(fs=16000):
    """
    Synthesizes 'kaha' (कहा):
    - [k]: 50ms silence + 25ms unvoiced burst
    - [a]: 220ms vowel (F1=750Hz, F2=1250Hz, F3=2550Hz)
    - [h]: 80ms breathy aspiration (noise excited tract)
    - [a]: 300ms stressed oral vowel (F1=800Hz, F2=1200Hz, F3=2600Hz)
    """
    total_dur = 0.70
    n = int(total_dur * fs)
    
    # Times (samples)
    n_k_stop = int(0.04 * fs)
    n_k_burst = int(0.07 * fs)
    n_a1_end = int(0.28 * fs)
    n_h_end = int(0.36 * fs)
    n_a2_end = n
    
    voiced_source = create_glottal_pulse_train(total_dur, fs=fs, f0=135.0)
    noise_source = np.random.randn(n) * 0.25
    
    audio = np.zeros(n)
    
    # 1. /k/ burst
    audio[n_k_stop:n_k_burst] = noise_source[n_k_stop:n_k_burst] * np.hamming(n_k_burst - n_k_stop) * 0.6
    
    # 2. First vowel /a/
    len_a1 = n_a1_end - n_k_burst
    env_a1 = np.hanning(len_a1 * 2)[:len_a1]
    src_a1 = voiced_source[n_k_burst:n_a1_end] * env_a1
    v1 = formant_filter(src_a1, fs, 750, 100, gain=1.2)
    v1 += formant_filter(src_a1, fs, 1250, 120, gain=0.8)
    v1 += formant_filter(src_a1, fs, 2550, 150, gain=0.3)
    audio[n_k_burst:n_a1_end] += v1
    
    # 3. Intervocalic /h/
    len_h = n_h_end - n_a1_end
    env_h = np.sin(np.pi * np.linspace(0, 1, len_h))
    src_h = (noise_source[n_a1_end:n_h_end] * 0.7 + voiced_source[n_a1_end:n_h_end] * 0.3) * env_h
    vh = formant_filter(src_h, fs, 800, 200, gain=0.6)
    vh += formant_filter(src_h, fs, 1300, 250, gain=0.4)
    audio[n_a1_end:n_h_end] += vh
    
    # 4. Second vowel /a/ (oral)
    len_a2 = n_a2_end - n_h_end
    env_a2 = np.ones(len_a2)
    fade_len = int(0.08 * fs)
    env_a2[-fade_len:] = np.cos(np.linspace(0, np.pi/2, fade_len))**2
    env_a2[:int(0.04*fs)] = np.sin(np.linspace(0, np.pi/2, int(0.04*fs)))**2
    
    src_a2 = voiced_source[n_h_end:n_a2_end] * env_a2
    v2 = formant_filter(src_a2, fs, 820, 90, gain=1.4)
    v2 += formant_filter(src_a2, fs, 1220, 110, gain=0.9)
    v2 += formant_filter(src_a2, fs, 2600, 140, gain=0.35)
    audio[n_h_end:n_a2_end] += v2
    
    # Normalize to -1 to +1 safe amplitude
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.85
    return audio

def synthesize_kahaan(fs=16000):
    """
    Synthesizes 'kahaan' (कहाँ):
    - [k], [a], [h] same as kaha
    - [ã]: nasalized vowel!
      Key acoustic differences in Hindi nasalized vowels:
      1. Low-frequency nasal murmur formant (Fn1 ~ 280-320 Hz)
      2. Nasal zero / anti-resonance (~ 750 Hz) that dampens oral F1
      3. Extra nasal resonance (Fn2 ~ 1000-1100 Hz)
      4. Wider formant bandwidths due to acoustic loss in nasal cavity
      5. Glottal-nasal pulse coupling in the residual signal
    """
    total_dur = 0.74
    n = int(total_dur * fs)
    
    n_k_stop = int(0.04 * fs)
    n_k_burst = int(0.07 * fs)
    n_a1_end = int(0.28 * fs)
    n_h_end = int(0.36 * fs)
    n_an_end = n
    
    voiced_source = create_glottal_pulse_train(total_dur, fs=fs, f0=135.0)
    noise_source = np.random.randn(n) * 0.25
    
    audio = np.zeros(n)
    
    # 1. /k/ burst
    audio[n_k_stop:n_k_burst] = noise_source[n_k_stop:n_k_burst] * np.hamming(n_k_burst - n_k_stop) * 0.6
    
    # 2. First vowel /a/
    len_a1 = n_a1_end - n_k_burst
    env_a1 = np.hanning(len_a1 * 2)[:len_a1]
    src_a1 = voiced_source[n_k_burst:n_a1_end] * env_a1
    v1 = formant_filter(src_a1, fs, 750, 100, gain=1.2)
    v1 += formant_filter(src_a1, fs, 1250, 120, gain=0.8)
    v1 += formant_filter(src_a1, fs, 2550, 150, gain=0.3)
    audio[n_k_burst:n_a1_end] += v1
    
    # 3. Intervocalic /h/
    len_h = n_h_end - n_a1_end
    env_h = np.sin(np.pi * np.linspace(0, 1, len_h))
    src_h = (noise_source[n_a1_end:n_h_end] * 0.7 + voiced_source[n_a1_end:n_h_end] * 0.3) * env_h
    vh = formant_filter(src_h, fs, 800, 200, gain=0.6)
    vh += formant_filter(src_h, fs, 1300, 250, gain=0.4)
    audio[n_a1_end:n_h_end] += vh
    
    # 4. Nasalized vowel /ã/
    len_an = n_an_end - n_h_end
    env_an = np.ones(len_an)
    fade_len = int(0.09 * fs)
    env_an[-fade_len:] = np.cos(np.linspace(0, np.pi/2, fade_len))**2
    env_an[:int(0.04*fs)] = np.sin(np.linspace(0, np.pi/2, int(0.04*fs)))**2
    
    src_an = voiced_source[n_h_end:n_an_end] * env_an
    # Nasal Formants & modified Oral Formants
    # Fn1 (nasal murmur) around 280 Hz
    vn = formant_filter(src_an, fs, 280, 70, gain=1.1)
    # Reduced F1 (dampened by nasal zero) with broader bandwidth
    vn += formant_filter(src_an, fs, 780, 220, gain=0.7)
    # Nasal resonance Fn2 around 1050 Hz
    vn += formant_filter(src_an, fs, 1050, 140, gain=0.95)
    # F2 around 1280 Hz
    vn += formant_filter(src_an, fs, 1280, 150, gain=0.6)
    # F3 around 2650 Hz
    vn += formant_filter(src_an, fs, 2650, 200, gain=0.25)
    
    audio[n_h_end:n_an_end] += vn
    
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.85
    return audio

def synthesize_hello(fs=16000):
    """
    Synthesizes 'Hello':
    - [h]: 80ms aspiration
    - [ɛ]: 220ms vowel (F1=600Hz, F2=1800Hz, F3=2600Hz)
    - [l]: 100ms lateral (F1=400Hz, F2=1000Hz, F3=2800Hz)
    - [oʊ]: 280ms diphthong (F1=500->400Hz, F2=1000->850Hz)
    """
    total_dur = 0.70
    n = int(total_dur * fs)
    
    voiced_source = create_glottal_pulse_train(total_dur, fs=fs, f0=125.0)
    noise_source = np.random.randn(n) * 0.25
    
    audio = np.zeros(n)
    
    # 1. [h]
    n_h = int(0.09 * fs)
    h_env = np.hanning(n_h * 2)[:n_h]
    audio[:n_h] = formant_filter(noise_source[:n_h] * h_env, fs, 1200, 300, gain=0.5)
    
    # 2. [e]
    n_e_end = int(0.32 * fs)
    len_e = n_e_end - n_h
    env_e = np.ones(len_e)
    env_e[:int(0.03*fs)] = np.linspace(0, 1, int(0.03*fs))
    src_e = voiced_source[n_h:n_e_end] * env_e
    ve = formant_filter(src_e, fs, 580, 90, gain=1.2)
    ve += formant_filter(src_e, fs, 1820, 110, gain=0.85)
    ve += formant_filter(src_e, fs, 2600, 140, gain=0.4)
    audio[n_h:n_e_end] += ve
    
    # 3. [l]
    n_l_end = int(0.42 * fs)
    len_l = n_l_end - n_e_end
    src_l = voiced_source[n_e_end:n_l_end] * 0.7
    vl = formant_filter(src_l, fs, 400, 100, gain=0.9)
    vl += formant_filter(src_l, fs, 1050, 120, gain=0.5)
    audio[n_e_end:n_l_end] += vl
    
    # 4. [o]
    len_o = n - n_l_end
    env_o = np.ones(len_o)
    env_o[-int(0.08*fs):] = np.cos(np.linspace(0, np.pi/2, int(0.08*fs)))**2
    src_o = voiced_source[n_l_end:] * env_o
    vo = formant_filter(src_o, fs, 480, 80, gain=1.3)
    vo += formant_filter(src_o, fs, 950, 100, gain=0.7)
    vo += formant_filter(src_o, fs, 2400, 150, gain=0.3)
    audio[n_l_end:] += vo
    
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.85
    return audio

def save_wav(filename, audio, fs=16000):
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wavfile.write(filename, fs, audio_int16)
    print(f"Saved: {filename} ({len(audio)} samples, {len(audio)/fs:.2f}s)")

def main():
    os.makedirs("data", exist_ok=True)
    fs = 16000
    
    print("Generating speech audio samples for LPC experimentation...")
    kaha_audio = synthesize_kaha(fs)
    save_wav("data/kaha.wav", kaha_audio, fs)
    
    kahaan_audio = synthesize_kahaan(fs)
    save_wav("data/kahaan.wav", kahaan_audio, fs)
    
    hello_audio = synthesize_hello(fs)
    save_wav("data/hello.wav", hello_audio, fs)
    
    print("All sample audio files successfully generated in ./data/ directory!")

if __name__ == "__main__":
    main()
