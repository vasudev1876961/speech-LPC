"""
app.py
------
Interactive Streamlit Dashboard for Linear Predictive Coding (LPC) Speech Analysis.
Provides:
  - Audio playback (Original Speech vs LPC Residual Excitation)
  - Interactive parameter controls (LPC Order, Frame length, Hop length, Pre-emphasis)
  - Rich visual analysis: Waveforms, Spectrograms, LPC Spectral Envelopes
  - Multi-Voice & Minimal Pair Comparison Studio:
      * Hindi Minimal Pairs ('kaha' vs 'kahaan', 'baas' vs 'baans', 'saas' vs 'saans')
      * Speaker Gender & Pitch Contrasts (Male vs Female voice)
      * Cardinal Vowels (/i/ vs /u/, vocal tract geometry)
      * Voiced vs Whispered excitation source
      * Custom 2-voice pairwise comparator
  - One-click export for Member 3 & Member 4 deliverables (.npy, .wav, .json)
"""

import os
import io
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from lpc_core.preprocessor import load_and_preprocess_audio, save_wav_file
from lpc_core.lpc_analyzer import extract_lpc_coefficients
from lpc_core.inverse_filter import compute_lpc_residual
from lpc_core.formant_analyzer import extract_formants_from_lpc, compute_lpc_spectrum
from lpc_core.features import extract_lpc_features
from lpc_core.visualizer import (
    plot_waveform_and_residual,
    plot_spectrogram_comparison,
    plot_lpc_spectral_envelope
)
from compare_words import compare_pair

# Page configuration
st.set_page_config(
    page_title="LPC Speech & Voice Analysis Suite",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .stAudio {
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .comparison-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background-color: #e0e7ff;
        color: #3730a3;
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎙️ Linear Predictive Coding (LPC) Speech & Voice Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Acoustic Feature Extraction, Inverse Filtering, & Cross-Voice Comparative Analysis</div>', unsafe_allow_html=True)

# Ensure essential audio samples exist
if not os.path.exists("data/kaha.wav") or not os.path.exists("data/baas.wav"):
    os.system("python generate_test_audio.py")

# Available presets dictionary
PRESET_AUDIOS = {
    "कहा (kaha) - Oral /a/ [Male Standard]": "data/kaha.wav",
    "कहाँ (kahaan) - Nasalized /ã/ [Male]": "data/kahaan.wav",
    "बास (baas) - Oral /aː/ [Male]": "data/baas.wav",
    "बाँस (baans) - Nasalized /ãː/ [Male]": "data/baans.wav",
    "सास (saas) - Oral /aː/ [Male]": "data/saas.wav",
    "साँस (saans) - Nasalized /ãː/ [Male]": "data/saans.wav",
    "कहा (kaha) - Female Voice (F0 ~ 225 Hz)": "data/kaha_female.wav",
    "कहा (kaha) - Deep Baritone Voice (F0 ~ 95 Hz)": "data/kaha_deep_male.wav",
    "Hello - Male Voice (F0 ~ 125 Hz)": "data/hello.wav",
    "Hello - Female Voice (F0 ~ 230 Hz)": "data/hello_female.wav",
    "Vowel /a/ ('father') - Open Vowel": "data/vowel_a.wav",
    "Vowel /i/ ('see') - High Front Vowel": "data/vowel_i.wav",
    "Vowel /u/ ('boot') - High Back Rounded Vowel": "data/vowel_u.wav",
    "कहा (kaha) - Whispered (Unvoiced Noise Source)": "data/whisper_kaha.wav"
}

# Sidebar settings
st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.radio("Analysis Mode", ["Single Audio Analysis", "Compare Voices & Words"])

st.sidebar.subheader("LPC DSP Parameters")
order = st.sidebar.slider("LPC Order (p)", min_value=4, max_value=24, value=16, step=2,
                         help="Number of predictor coefficients. Rule of thumb: fs/1000 + 4.")
frame_ms = st.sidebar.slider("Frame Length (ms)", min_value=10.0, max_value=40.0, value=25.0, step=2.5,
                            help="Window length for quasi-stationary analysis.")
hop_ms = st.sidebar.slider("Hop Length (ms)", min_value=5.0, max_value=20.0, value=10.0, step=2.5,
                          help="Frame advancement step.")
pre_emphasis = st.sidebar.checkbox("Apply Pre-emphasis (alpha=0.97)", value=True,
                                  help="Boosts higher formant frequencies by flattening glottal roll-off.")

# =========================================================================
# MODE 1: SINGLE AUDIO ANALYSIS
# =========================================================================
if mode == "Single Audio Analysis":
    st.subheader("1. Audio Input Selection")
    input_source = st.radio("Choose audio source:", ["Preset Voice Sample", "Upload Custom WAV"], horizontal=True)
    
    audio_path = None
    if input_source == "Preset Voice Sample":
        sample_label = st.selectbox("Select preset utterance:", list(PRESET_AUDIOS.keys()))
        audio_path = PRESET_AUDIOS[sample_label]
    else:
        uploaded_file = st.file_uploader("Upload 16kHz WAV file", type=["wav"])
        if uploaded_file is not None:
            os.makedirs("scratch", exist_ok=True)
            audio_path = os.path.join("scratch", uploaded_file.name)
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
    if audio_path and os.path.exists(audio_path):
        # Process audio
        proc_sig, raw_sig, fs = load_and_preprocess_audio(
            audio_path,
            target_fs=16000,
            apply_preemph=pre_emphasis,
            preemph_coeff=0.97
        )
        lpc_res = extract_lpc_coefficients(
            proc_sig,
            fs=fs,
            order=order,
            frame_ms=frame_ms,
            hop_ms=hop_ms
        )
        coeffs = lpc_res['lpc_coeffs']
        residual = compute_lpc_residual(proc_sig, lpc_res, method="sample_varying")
        features = extract_lpc_features(proc_sig, residual, lpc_res)
        
        # Audio Players Side by Side
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("##### 🔊 Original Speech Recording")
            st.audio(audio_path)
            st.caption(f"File: `{os.path.basename(audio_path)}` | Samples: {len(raw_sig):,} ({len(raw_sig)/fs:.2f}s)")
        with col_a2:
            st.markdown("##### 🔊 LPC Residual (Excitation Source)")
            res_norm = residual / (np.max(np.abs(residual)) + 1e-9) * 0.9
            res_int16 = np.int16(np.clip(res_norm, -1.0, 1.0) * 32767)
            wav_bytes = io.BytesIO()
            wavfile.write(wav_bytes, fs, res_int16)
            st.audio(wav_bytes.getvalue(), format="audio/wav")
            st.caption("Acoustic Inverse Filter: Vocal tract resonances removed, leaving glottal pulses.")

        # Metric summary cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Frames", f"{len(coeffs)}")
        with m2:
            st.metric("LPC Matrix Shape", f"{coeffs.shape[0]} × {coeffs.shape[1]}")
        with m3:
            st.metric("Prediction Gain", f"{features['mean_prediction_gain_db']:.2f} dB")
        with m4:
            st.metric("Residual RMS", f"{features['mean_residual_rms']:.4f}")

        st.markdown("---")
        
        # Visual Graphs Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Waveform & Residual", "🌈 Spectrogram Comparison", "🔬 LPC Formants & Spectral Envelope"])
        
        with tab1:
            fig1 = plot_waveform_and_residual(raw_sig, residual, fs, title=os.path.basename(audio_path))
            st.pyplot(fig1)
            plt.close(fig1)
            
        with tab2:
            fig2 = plot_spectrogram_comparison(raw_sig, residual, fs, title=f"Spectrogram: {os.path.basename(audio_path)}")
            st.pyplot(fig2)
            plt.close(fig2)
            
        with tab3:
            mid_frame = len(coeffs) // 2
            frame_slider = st.slider("Select Frame Index to Inspect Spectrum", 0, len(coeffs) - 1, mid_frame)
            frame_sample_start = lpc_res['indices'][frame_slider]
            frame_data = proc_sig[frame_sample_start : frame_sample_start + lpc_res['frame_len']]
            
            fig3 = plot_lpc_spectral_envelope(
                frame_data,
                coeffs[frame_slider],
                fs,
                frame_time=lpc_res['frame_times'][frame_slider]
            )
            st.pyplot(fig3)
            plt.close(fig3)
            
            f_vals, _ = extract_formants_from_lpc(coeffs[frame_slider], fs=fs)
            st.write("**Extracted Formants for this frame:** " + ", ".join([f"**F{i+1}:** {int(f)} Hz" for i, f in enumerate(f_vals[:4])]))

        st.markdown("---")
        # Download Section for Member 3
        st.subheader("💾 Export Deliverables for Member 3 & Member 4")
        dcol1, dcol2, dcol3 = st.columns(3)
        
        with dcol1:
            coeff_bytes = io.BytesIO()
            np.save(coeff_bytes, coeffs)
            st.download_button(
                "📥 Download lpc_coefficients.npy",
                data=coeff_bytes.getvalue(),
                file_name="lpc_coefficients.npy",
                mime="application/octet-stream"
            )
            
        with dcol2:
            res_bytes = io.BytesIO()
            np.save(res_bytes, residual)
            st.download_button(
                "📥 Download lpc_residual.npy",
                data=res_bytes.getvalue(),
                file_name="lpc_residual.npy",
                mime="application/octet-stream"
            )
            
        with dcol3:
            st.download_button(
                "📥 Download lpc_residual.wav",
                data=wav_bytes.getvalue(),
                file_name="lpc_residual.wav",
                mime="audio/wav"
            )

# =========================================================================
# MODE 2: COMPARE VOICES & WORDS
# =========================================================================
else:
    st.subheader("👥 Comparative Voice & Minimal Pair Studio")
    st.markdown("Compare vocal tract filters, formant shifts, and glottal residual signals across minimal pairs, speaker genders, vowels, and excitation sources.")
    
    COMPARISON_PRESETS = {
        "1. Minimal Pair: 'kaha' vs 'kahaan' (कहा vs कहाँ — Oral vs Nasalized)": ("data/kaha.wav", "data/kahaan.wav", "kaha", "kahaan"),
        "2. Minimal Pair: 'baas' vs 'baans' (बास vs बाँस — Oral vs Nasalized)": ("data/baas.wav", "data/baans.wav", "baas", "baans"),
        "3. Minimal Pair: 'saas' vs 'saans' (सास vs साँस — Oral vs Nasalized)": ("data/saas.wav", "data/saans.wav", "saas", "saans"),
        "4. Speaker Voice: 'kaha' Male vs 'kaha' Female (Pitch F0 & Vocal Tract Length)": ("data/kaha.wav", "data/kaha_female.wav", "kaha_male", "kaha_female"),
        "5. Speaker Voice: 'hello' Male vs 'hello' Female": ("data/hello.wav", "data/hello_female.wav", "hello_male", "hello_female"),
        "6. Vocal Tract Shape: Vowel /i/ ('see') vs Vowel /u/ ('boot')": ("data/vowel_i.wav", "data/vowel_u.wav", "vowel_i", "vowel_u"),
        "7. Excitation Contrast: Voiced 'kaha' vs Whispered 'whisper_kaha'": ("data/kaha.wav", "data/whisper_kaha.wav", "kaha_voiced", "kaha_whisper"),
        "8. ⚙️ Custom Pair Selection": (None, None, "voice_1", "voice_2")
    }
    
    preset_selection = st.selectbox("Select Voice / Minimal Pair Comparison:", list(COMPARISON_PRESETS.keys()))
    f1_default, f2_default, n1_default, n2_default = COMPARISON_PRESETS[preset_selection]
    
    if preset_selection.startswith("8."):
        # Custom selection
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            all_files = list(PRESET_AUDIOS.keys())
            f1_choice = st.selectbox("Select Voice 1:", all_files, index=0)
            file1 = PRESET_AUDIOS[f1_choice]
            name1 = os.path.splitext(os.path.basename(file1))[0]
        with c_col2:
            f2_choice = st.selectbox("Select Voice 2:", all_files, index=1)
            file2 = PRESET_AUDIOS[f2_choice]
            name2 = os.path.splitext(os.path.basename(file2))[0]
    else:
        file1 = f1_default
        file2 = f2_default
        name1 = n1_default
        name2 = n2_default
        
    # Audio comparison players
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown(f"##### 🔊 Voice 1: `{name1}`")
        if os.path.exists(file1):
            st.audio(file1)
    with pcol2:
        st.markdown(f"##### 🔊 Voice 2: `{name2}`")
        if os.path.exists(file2):
            st.audio(file2)
            
    st.markdown("---")
    
    # Run Comparison
    if st.button("🚀 Run LPC Comparative Analysis", type="primary"):
        with st.spinner(f"Analyzing and comparing '{name1}' vs '{name2}' with LPC..."):
            report = compare_pair(file1, file2, name1=name1, name2=name2)
            st.success("Analysis complete!")
            
            # Key comparative metrics
            cm1, cm2, cm3, cm4 = st.columns(4)
            with cm1:
                st.metric("LPC Cosine Distance", f"{report['mean_lpc_cosine_distance']:.4f}", help="0 = Identical vocal tract filter, 1 = Orthogonal")
            with cm2:
                st.metric(f"{name1} Residual RMS", f"{report['word1_mean_residual_rms']:.4f}")
            with cm3:
                st.metric(f"{name2} Residual RMS", f"{report['word2_mean_residual_rms']:.4f}")
            with cm4:
                st.metric("Residual Energy Ratio", f"{report['residual_rms_ratio']:.2f}")
                
            # Acoustic explanation alert
            st.info(f"💡 **Acoustic Phonetics Interpretation:** {report['acoustic_interpretation']}")
            
            # Formant comparison table
            st.write("##### 🔬 Formant Resonances (Mid-Vowel Region)")
            formant_data = {
                "Formant": ["F1", "F2", "F3", "F4"],
                f"{name1} (Hz)": [f"{f} Hz" for f in (report['word1_vowel_formants_hz'] + [0,0,0,0])[:4]],
                f"{name2} (Hz)": [f"{f} Hz" for f in (report['word2_vowel_formants_hz'] + [0,0,0,0])[:4]]
            }
            st.table(formant_data)
            
            # Show Visuals
            st.write("##### 📈 Comparative Acoustic Visualizations")
            
            comp_plot = report.get('comparison_plot')
            envelope_plot = report.get('spectral_envelope_plot')
            
            if comp_plot and os.path.exists(comp_plot):
                st.image(comp_plot, caption=f"Speech Waveforms, LPC Residuals & RMS Energy Contours ({name1} vs {name2})", use_container_width=True)
                
            if envelope_plot and os.path.exists(envelope_plot):
                st.image(envelope_plot, caption=f"LPC Spectral Envelope Overlay ({name1} vs {name2})", use_container_width=True)
                
            # Download JSON report
            st.download_button(
                "📥 Download Comparison Report (.json)",
                data=json.dumps(report, indent=2),
                file_name=f"lpc_comparison_{name1}_vs_{name2}.json",
                mime="application/json"
            )
