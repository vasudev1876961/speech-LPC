"""
generate_test_audio.py
----------------------
Generates realistic acoustic speech audio samples for testing and comparing
the LPC analysis pipeline across multiple dimensions:

1. Minimal Pairs (Oral vs Nasalized Vowels in Hindi):
   - 'kaha.wav'   vs 'kahaan.wav' (कहा vs कहाँ)
   - 'baas.wav'   vs 'baans.wav'  (बास [odor] vs बाँस [bamboo])
   - 'saas.wav'   vs 'saans.wav'  (सास [mother-in-law] vs साँस [breath])

2. Voice / Speaker Gender & Pitch Comparisons:
   - 'kaha.wav'          (Male voice, F0 ~ 130 Hz)
   - 'kaha_female.wav'   (Female voice, F0 ~ 225 Hz, shorter vocal tract)
   - 'kaha_deep_male.wav'(Deep male voice, F0 ~ 95 Hz, longer vocal tract)
   - 'hello.wav'         (Male greeting, F0 ~ 125 Hz)
   - 'hello_female.wav'  (Female greeting, F0 ~ 230 Hz)

3. Cardinal Acoustic Vowels (Vocal Tract Resonances):
   - 'vowel_a.wav' (/a/ as in 'father', open vowel: high F1, mid F2)
   - 'vowel_i.wav' (/i/ as in 'see', high front vowel: low F1, very high F2)
   - 'vowel_u.wav' (/u/ as in 'boot', high back rounded vowel: low F1, low F2)

4. Excitation Source Contrast:
   - 'whisper_kaha.wav' (Whispered speech: unvoiced turbulent noise source)

Uses the source-filter acoustic speech model:
- Rosenberg glottal pulse train with pitch contours and natural jitter
- 2nd-order resonator digital filter cascade for formants
- Nasal anti-resonances and extra nasal poles for nasalized vowels
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import lfilter, butter

def create_glottal_pulse_train(duration, fs=16000, f0=130.0, jitter=0.01):
    """
    Synthesize Rosenberg glottal pulse train for natural voiced speech source.
    """
    num_samples = int(duration * fs)
    source = np.zeros(num_samples)
    
    t = np.linspace(0, duration, num_samples)
    pitch_contour = f0 * (1.0 - 0.12 * t / max(duration, 0.001))
    
    phase = 0.0
    for i in range(num_samples):
        current_f0 = pitch_contour[i] * (1.0 + jitter * (np.random.rand() - 0.5))
        phase += max(current_f0, 40.0) / fs
        if phase >= 1.0:
            phase -= 1.0
        tn = phase
        if tn < 0.4:
            pulse = 0.5 * (1.0 - np.cos(np.pi * tn / 0.4))
        elif tn < 0.6:
            pulse = np.cos(np.pi * (tn - 0.4) / (2.0 * 0.2))
        else:
            pulse = 0.0
            
        source[i] = pulse
        
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

def normalize_audio(audio, peak=0.85):
    max_val = np.max(np.abs(audio))
    if max_val > 1e-9:
        return (audio / max_val) * peak
    return audio

# ==========================================
# 1. 'kaha' & 'kahaan'
# ==========================================

def synthesize_kaha(fs=16000, f0=135.0, f_scale=1.0):
    total_dur = 0.70
    n = int(total_dur * fs)
    
    n_k_stop = int(0.04 * fs)
    n_k_burst = int(0.07 * fs)
    n_a1_end = int(0.28 * fs)
    n_h_end = int(0.36 * fs)
    n_a2_end = n
    
    voiced = create_glottal_pulse_train(total_dur, fs=fs, f0=f0)
    noise = np.random.randn(n) * 0.25
    audio = np.zeros(n)
    
    # 1. /k/ burst
    audio[n_k_stop:n_k_burst] = noise[n_k_stop:n_k_burst] * np.hamming(n_k_burst - n_k_stop) * 0.6
    
    # 2. First vowel /a/
    len_a1 = n_a1_end - n_k_burst
    env_a1 = np.hanning(len_a1 * 2)[:len_a1]
    src_a1 = voiced[n_k_burst:n_a1_end] * env_a1
    v1 = formant_filter(src_a1, fs, 750 * f_scale, 100 * f_scale, gain=1.2)
    v1 += formant_filter(src_a1, fs, 1250 * f_scale, 120 * f_scale, gain=0.8)
    v1 += formant_filter(src_a1, fs, 2550 * f_scale, 150 * f_scale, gain=0.3)
    audio[n_k_burst:n_a1_end] += v1
    
    # 3. /h/
    len_h = n_h_end - n_a1_end
    env_h = np.sin(np.pi * np.linspace(0, 1, len_h))
    src_h = (noise[n_a1_end:n_h_end] * 0.7 + voiced[n_a1_end:n_h_end] * 0.3) * env_h
    vh = formant_filter(src_h, fs, 800 * f_scale, 200, gain=0.6)
    vh += formant_filter(src_h, fs, 1300 * f_scale, 250, gain=0.4)
    audio[n_a1_end:n_h_end] += vh
    
    # 4. Second vowel /a/ (oral)
    len_a2 = n_a2_end - n_h_end
    env_a2 = np.ones(len_a2)
    fade_len = int(0.08 * fs)
    env_a2[-fade_len:] = np.cos(np.linspace(0, np.pi/2, fade_len))**2
    env_a2[:int(0.04*fs)] = np.sin(np.linspace(0, np.pi/2, int(0.04*fs)))**2
    
    src_a2 = voiced[n_h_end:n_a2_end] * env_a2
    v2 = formant_filter(src_a2, fs, 820 * f_scale, 90 * f_scale, gain=1.4)
    v2 += formant_filter(src_a2, fs, 1220 * f_scale, 110 * f_scale, gain=0.9)
    v2 += formant_filter(src_a2, fs, 2600 * f_scale, 140 * f_scale, gain=0.35)
    audio[n_h_end:n_a2_end] += v2
    
    return normalize_audio(audio)

def synthesize_kahaan(fs=16000, f0=135.0, f_scale=1.0):
    total_dur = 0.74
    n = int(total_dur * fs)
    
    n_k_stop = int(0.04 * fs)
    n_k_burst = int(0.07 * fs)
    n_a1_end = int(0.28 * fs)
    n_h_end = int(0.36 * fs)
    n_an_end = n
    
    voiced = create_glottal_pulse_train(total_dur, fs=fs, f0=f0)
    noise = np.random.randn(n) * 0.25
    audio = np.zeros(n)
    
    audio[n_k_stop:n_k_burst] = noise[n_k_stop:n_k_burst] * np.hamming(n_k_burst - n_k_stop) * 0.6
    
    len_a1 = n_a1_end - n_k_burst
    env_a1 = np.hanning(len_a1 * 2)[:len_a1]
    src_a1 = voiced[n_k_burst:n_a1_end] * env_a1
    v1 = formant_filter(src_a1, fs, 750 * f_scale, 100 * f_scale, gain=1.2)
    v1 += formant_filter(src_a1, fs, 1250 * f_scale, 120 * f_scale, gain=0.8)
    v1 += formant_filter(src_a1, fs, 2550 * f_scale, 150 * f_scale, gain=0.3)
    audio[n_k_burst:n_a1_end] += v1
    
    len_h = n_h_end - n_a1_end
    env_h = np.sin(np.pi * np.linspace(0, 1, len_h))
    src_h = (noise[n_a1_end:n_h_end] * 0.7 + voiced[n_a1_end:n_h_end] * 0.3) * env_h
    vh = formant_filter(src_h, fs, 800 * f_scale, 200, gain=0.6)
    vh += formant_filter(src_h, fs, 1300 * f_scale, 250, gain=0.4)
    audio[n_a1_end:n_h_end] += vh
    
    # Nasalized vowel /ã/
    len_an = n_an_end - n_h_end
    env_an = np.ones(len_an)
    fade_len = int(0.09 * fs)
    env_an[-fade_len:] = np.cos(np.linspace(0, np.pi/2, fade_len))**2
    env_an[:int(0.04*fs)] = np.sin(np.linspace(0, np.pi/2, int(0.04*fs)))**2
    
    src_an = voiced[n_h_end:n_an_end] * env_an
    vn = formant_filter(src_an, fs, 280, 70, gain=1.1)
    vn += formant_filter(src_an, fs, 780 * f_scale, 220, gain=0.7)
    vn += formant_filter(src_an, fs, 1050 * f_scale, 140, gain=0.95)
    vn += formant_filter(src_an, fs, 1280 * f_scale, 150, gain=0.6)
    vn += formant_filter(src_an, fs, 2650 * f_scale, 200, gain=0.25)
    audio[n_h_end:n_an_end] += vn
    
    return normalize_audio(audio)

# ==========================================
# 2. 'baas' & 'baans' Minimal Pair
# ==========================================

def synthesize_baas(fs=16000, f0=130.0, nasal=False):
    """
    Synthesizes 'baas' (बास) or 'baans' (बाँस):
    - [b]: Voiced bilabial stop (closure voice bar + burst)
    - [aː] or [ãː]: Long vowel (oral vs nasalized)
    - [s]: Unvoiced alveolar fricative high-frequency turbulence
    """
    total_dur = 0.75
    n = int(total_dur * fs)
    
    n_b_close = int(0.05 * fs)
    n_b_burst = int(0.08 * fs)
    n_v_end = int(0.52 * fs)
    
    voiced = create_glottal_pulse_train(total_dur, fs=fs, f0=f0)
    noise = np.random.randn(n)
    audio = np.zeros(n)
    
    # 1. [b] Voice bar during closure
    t_close = np.arange(n_b_close) / fs
    voice_bar = 0.25 * np.sin(2 * np.pi * f0 * t_close)
    audio[:n_b_close] = voice_bar
    
    # Release burst
    b_filter, a_filter = butter(2, 600 / (fs/2), btype='low')
    audio[n_b_close:n_b_burst] = lfilter(b_filter, a_filter, noise[n_b_close:n_b_burst]) * 0.4
    
    # 2. Vowel /aː/ or /ãː/
    len_v = n_v_end - n_b_burst
    env_v = np.ones(len_v)
    env_v[:int(0.03*fs)] = np.linspace(0, 1, int(0.03*fs))
    env_v[-int(0.05*fs):] = np.linspace(1, 0.4, int(0.05*fs))
    src_v = voiced[n_b_burst:n_v_end] * env_v
    
    if not nasal:
        # Oral vowel
        vv = formant_filter(src_v, fs, 780, 95, gain=1.3)
        vv += formant_filter(src_v, fs, 1260, 110, gain=0.85)
        vv += formant_filter(src_v, fs, 2580, 140, gain=0.35)
    else:
        # Nasalized vowel
        vv = formant_filter(src_v, fs, 290, 75, gain=1.1)     # Fn1 nasal murmur
        vv += formant_filter(src_v, fs, 770, 210, gain=0.65)   # Damped F1
        vv += formant_filter(src_v, fs, 1080, 130, gain=0.90)  # Fn2 nasal pole
        vv += formant_filter(src_v, fs, 1290, 150, gain=0.55)  # F2
        vv += formant_filter(src_v, fs, 2620, 190, gain=0.25)
    audio[n_b_burst:n_v_end] += vv
    
    # 3. [s] Alveolar fricative
    len_s = n - n_v_end
    b_s, a_s = butter(4, [4200 / (fs/2), min(7600 / (fs/2), 0.95)], btype='band')
    s_noise = lfilter(b_s, a_s, noise[n_v_end:])
    env_s = np.sin(np.pi * np.linspace(0, 1, len_s))**1.2
    audio[n_v_end:] += s_noise * env_s * 0.35
    
    return normalize_audio(audio)

# ==========================================
# 3. 'saas' & 'saans' Minimal Pair
# ==========================================

def synthesize_saas(fs=16000, f0=130.0, nasal=False):
    """
    Synthesizes 'saas' (सास) or 'saans' (साँस):
    - [s]: Initial unvoiced alveolar fricative
    - [aː] or [ãː]: Medial vowel
    - [s]: Final unvoiced alveolar fricative
    """
    total_dur = 0.78
    n = int(total_dur * fs)
    
    n_s1_end = int(0.18 * fs)
    n_v_end = int(0.58 * fs)
    
    voiced = create_glottal_pulse_train(total_dur, fs=fs, f0=f0)
    noise = np.random.randn(n)
    audio = np.zeros(n)
    
    b_s, a_s = butter(4, [4300 / (fs/2), min(7600 / (fs/2), 0.95)], btype='band')
    
    # 1. Initial [s]
    s1_noise = lfilter(b_s, a_s, noise[:n_s1_end])
    env_s1 = np.sin(np.pi * np.linspace(0, 1, n_s1_end))**1.3
    audio[:n_s1_end] = s1_noise * env_s1 * 0.32
    
    # 2. Vowel
    len_v = n_v_end - n_s1_end
    env_v = np.ones(len_v)
    env_v[:int(0.04*fs)] = np.sin(np.linspace(0, np.pi/2, int(0.04*fs)))**2
    env_v[-int(0.04*fs):] = np.cos(np.linspace(0, np.pi/2, int(0.04*fs)))**2
    src_v = voiced[n_s1_end:n_v_end] * env_v
    
    if not nasal:
        vv = formant_filter(src_v, fs, 770, 90, gain=1.35)
        vv += formant_filter(src_v, fs, 1240, 110, gain=0.85)
        vv += formant_filter(src_v, fs, 2550, 140, gain=0.35)
    else:
        vv = formant_filter(src_v, fs, 290, 70, gain=1.15)
        vv += formant_filter(src_v, fs, 760, 210, gain=0.65)
        vv += formant_filter(src_v, fs, 1070, 130, gain=0.92)
        vv += formant_filter(src_v, fs, 1270, 150, gain=0.55)
        vv += formant_filter(src_v, fs, 2600, 180, gain=0.25)
    audio[n_s1_end:n_v_end] += vv
    
    # 3. Final [s]
    len_s2 = n - n_v_end
    s2_noise = lfilter(b_s, a_s, noise[n_v_end:])
    env_s2 = np.sin(np.pi * np.linspace(0, 1, len_s2))**1.2
    audio[n_v_end:] += s2_noise * env_s2 * 0.35
    
    return normalize_audio(audio)

# ==========================================
# 4. 'hello' Variants
# ==========================================

def synthesize_hello(fs=16000, f0=125.0, f_scale=1.0):
    total_dur = 0.70
    n = int(total_dur * fs)
    
    voiced = create_glottal_pulse_train(total_dur, fs=fs, f0=f0)
    noise = np.random.randn(n) * 0.25
    audio = np.zeros(n)
    
    # 1. [h]
    n_h = int(0.09 * fs)
    h_env = np.hanning(n_h * 2)[:n_h]
    audio[:n_h] = formant_filter(noise[:n_h] * h_env, fs, 1200 * f_scale, 300, gain=0.5)
    
    # 2. [e]
    n_e_end = int(0.32 * fs)
    len_e = n_e_end - n_h
    env_e = np.ones(len_e)
    env_e[:int(0.03*fs)] = np.linspace(0, 1, int(0.03*fs))
    src_e = voiced[n_h:n_e_end] * env_e
    ve = formant_filter(src_e, fs, 580 * f_scale, 90 * f_scale, gain=1.2)
    ve += formant_filter(src_e, fs, 1820 * f_scale, 110 * f_scale, gain=0.85)
    ve += formant_filter(src_e, fs, 2600 * f_scale, 140 * f_scale, gain=0.4)
    audio[n_h:n_e_end] += ve
    
    # 3. [l]
    n_l_end = int(0.42 * fs)
    len_l = n_l_end - n_e_end
    src_l = voiced[n_e_end:n_l_end] * 0.7
    vl = formant_filter(src_l, fs, 400 * f_scale, 100 * f_scale, gain=0.9)
    vl += formant_filter(src_l, fs, 1050 * f_scale, 120 * f_scale, gain=0.5)
    audio[n_e_end:n_l_end] += vl
    
    # 4. [o]
    len_o = n - n_l_end
    env_o = np.ones(len_o)
    env_o[-int(0.08*fs):] = np.cos(np.linspace(0, np.pi/2, int(0.08*fs)))**2
    src_o = voiced[n_l_end:] * env_o
    vo = formant_filter(src_o, fs, 480 * f_scale, 80 * f_scale, gain=1.3)
    vo += formant_filter(src_o, fs, 950 * f_scale, 100 * f_scale, gain=0.7)
    vo += formant_filter(src_o, fs, 2400 * f_scale, 150 * f_scale, gain=0.3)
    audio[n_l_end:] += vo
    
    return normalize_audio(audio)

# ==========================================
# 5. Cardinal Vowels (/a/, /i/, /u/)
# ==========================================

def synthesize_vowel(vowel_type="a", fs=16000, f0=130.0, duration=0.65):
    """
    Synthesize sustained cardinal vowels illustrating extreme vocal tract shapes:
    - /a/: Open vowel (high F1 ~ 800Hz, mid F2 ~ 1250Hz)
    - /i/: High front vowel (low F1 ~ 280Hz, very high F2 ~ 2350Hz)
    - /u/: High back rounded vowel (low F1 ~ 320Hz, low F2 ~ 850Hz)
    """
    n = int(duration * fs)
    voiced = create_glottal_pulse_train(duration, fs=fs, f0=f0)
    
    env = np.ones(n)
    attack = int(0.06 * fs)
    decay = int(0.08 * fs)
    env[:attack] = np.sin(np.linspace(0, np.pi/2, attack))**2
    env[-decay:] = np.cos(np.linspace(0, np.pi/2, decay))**2
    src = voiced * env
    
    if vowel_type == "a":
        v = formant_filter(src, fs, 800, 90, gain=1.4)
        v += formant_filter(src, fs, 1250, 110, gain=0.85)
        v += formant_filter(src, fs, 2600, 140, gain=0.35)
    elif vowel_type == "i":
        v = formant_filter(src, fs, 280, 70, gain=1.4)
        v += formant_filter(src, fs, 2350, 100, gain=0.95)
        v += formant_filter(src, fs, 2900, 130, gain=0.4)
    elif vowel_type == "u":
        v = formant_filter(src, fs, 320, 75, gain=1.4)
        v += formant_filter(src, fs, 850, 90, gain=0.9)
        v += formant_filter(src, fs, 2250, 130, gain=0.3)
    else:
        v = formant_filter(src, fs, 500, 100, gain=1.0)
        
    return normalize_audio(v)

# ==========================================
# 6. Whispered Speech (Unvoiced Excitation)
# ==========================================

def synthesize_whisper_kaha(fs=16000):
    """
    Whispered version of 'kaha':
    The vocal tract resonances are identical, but the glottal pulse train
    is completely replaced by unvoiced turbulent airflow (white noise).
    LPC inverse filter eliminates resonances and leaves flat noise residual!
    """
    total_dur = 0.70
    n = int(total_dur * fs)
    noise = np.random.randn(n) * 0.4
    
    n_k_stop = int(0.04 * fs)
    n_k_burst = int(0.07 * fs)
    n_a1_end = int(0.28 * fs)
    n_h_end = int(0.36 * fs)
    n_a2_end = n
    
    audio = np.zeros(n)
    audio[n_k_stop:n_k_burst] = noise[n_k_stop:n_k_burst] * np.hamming(n_k_burst - n_k_stop) * 0.6
    
    len_a1 = n_a1_end - n_k_burst
    env_a1 = np.hanning(len_a1 * 2)[:len_a1]
    src_a1 = noise[n_k_burst:n_a1_end] * env_a1
    v1 = formant_filter(src_a1, fs, 750, 120, gain=1.1)
    v1 += formant_filter(src_a1, fs, 1250, 140, gain=0.8)
    v1 += formant_filter(src_a1, fs, 2550, 180, gain=0.3)
    audio[n_k_burst:n_a1_end] += v1
    
    len_h = n_h_end - n_a1_end
    env_h = np.sin(np.pi * np.linspace(0, 1, len_h))
    src_h = noise[n_a1_end:n_h_end] * env_h
    vh = formant_filter(src_h, fs, 800, 220, gain=0.6)
    vh += formant_filter(src_h, fs, 1300, 270, gain=0.4)
    audio[n_a1_end:n_h_end] += vh
    
    len_a2 = n_a2_end - n_h_end
    env_a2 = np.ones(len_a2)
    fade_len = int(0.08 * fs)
    env_a2[-fade_len:] = np.cos(np.linspace(0, np.pi/2, fade_len))**2
    env_a2[:int(0.04*fs)] = np.sin(np.linspace(0, np.pi/2, int(0.04*fs)))**2
    src_a2 = noise[n_h_end:n_a2_end] * env_a2
    v2 = formant_filter(src_a2, fs, 820, 110, gain=1.3)
    v2 += formant_filter(src_a2, fs, 1220, 130, gain=0.85)
    v2 += formant_filter(src_a2, fs, 2600, 170, gain=0.35)
    audio[n_h_end:n_a2_end] += v2
    
    return normalize_audio(audio)

def save_wav(filename, audio, fs=16000):
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wavfile.write(filename, fs, audio_int16)
    print(f"  [+] Generated: {filename} ({len(audio)} samples, {len(audio)/fs:.2f}s)")

def main():
    os.makedirs("data", exist_ok=True)
    fs = 16000
    
    print("\n=======================================================")
    print("  GENERATING COMPREHENSIVE SPEECH DATASET FOR LPC SUITE")
    print("=======================================================")
    
    # 1. Primary Minimal Pair (kaha vs kahaan)
    print("\n[Group 1] Hindi Minimal Pair: 'kaha' vs 'kahaan'")
    save_wav("data/kaha.wav", synthesize_kaha(fs, f0=135.0, f_scale=1.0), fs)
    save_wav("data/kahaan.wav", synthesize_kahaan(fs, f0=135.0, f_scale=1.0), fs)
    
    # 2. Minimal Pair: 'baas' vs 'baans' (बास vs बाँस)
    print("\n[Group 2] Hindi Minimal Pair: 'baas' vs 'baans'")
    save_wav("data/baas.wav", synthesize_baas(fs, f0=130.0, nasal=False), fs)
    save_wav("data/baans.wav", synthesize_baas(fs, f0=130.0, nasal=True), fs)
    
    # 3. Minimal Pair: 'saas' vs 'saans' (सास vs साँस)
    print("\n[Group 3] Hindi Minimal Pair: 'saas' vs 'saans'")
    save_wav("data/saas.wav", synthesize_saas(fs, f0=130.0, nasal=False), fs)
    save_wav("data/saans.wav", synthesize_saas(fs, f0=130.0, nasal=True), fs)
    
    # 4. Gender / Speaker Voice Comparisons
    print("\n[Group 4] Speaker Pitch & Vocal Tract Comparisons")
    save_wav("data/kaha_female.wav", synthesize_kaha(fs, f0=225.0, f_scale=1.17), fs)
    save_wav("data/kaha_deep_male.wav", synthesize_kaha(fs, f0=95.0, f_scale=0.92), fs)
    save_wav("data/hello.wav", synthesize_hello(fs, f0=125.0, f_scale=1.0), fs)
    save_wav("data/hello_female.wav", synthesize_hello(fs, f0=230.0, f_scale=1.16), fs)
    
    # 5. Cardinal Acoustic Vowels
    print("\n[Group 5] Cardinal Vowels (/a/, /i/, /u/)")
    save_wav("data/vowel_a.wav", synthesize_vowel("a", fs, f0=130.0), fs)
    save_wav("data/vowel_i.wav", synthesize_vowel("i", fs, f0=130.0), fs)
    save_wav("data/vowel_u.wav", synthesize_vowel("u", fs, f0=130.0), fs)
    
    # 6. Excitation Contrast (Whisper)
    print("\n[Group 6] Excitation Contrast (Voiced vs Whispered)")
    save_wav("data/whisper_kaha.wav", synthesize_whisper_kaha(fs), fs)
    
    print("\n=======================================================")
    print("  SUCCESS: All speech files generated in ./data/!")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
