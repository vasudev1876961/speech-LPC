"""
app.py
------
Interactive Streamlit Dashboard for Linear Predictive Coding (LPC) Speech Analysis.
Provides:
  - Audio playback (Original Speech vs LPC Residual)
  - Interactive parameter controls (LPC Order, Frame length, Hop length)
  - Rich visual analysis: Waveforms, Spectrograms, LPC Filter Spectral Envelopes
  - Word comparison mode ('kaha' vs 'kahaan')
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

# Page configuration
st.set_page_config(
    page_title="LPC Speech Analyzer | Member 2",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for a clean, modern aesthetic
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
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎙️ Linear Predictive Coding (LPC) Speech Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Acoustic Feature & Residual Signal Extraction Module (Member 2 Deliverable)</div>', unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.radio("Analysis Mode", ["Single Audio Analysis", "Compare Words ('kaha' vs 'kahaan')"])

st.sidebar.subheader("LPC Parameters")
order = st.sidebar.slider("LPC Order (p)", min_value=4, max_value=24, value=16, step=2,
                         help="Number of predictor coefficients. Rule of thumb: fs/1000 + 4.")
frame_ms = st.sidebar.slider("Frame Length (ms)", min_value=10.0, max_value=40.0, value=25.0, step=2.5,
                            help="Window length for quasi-stationary analysis.")
hop_ms = st.sidebar.slider("Hop Length (ms)", min_value=5.0, max_value=20.0, value=10.0, step=2.5,
                          help="Frame advancement step.")
pre_emphasis = st.sidebar.checkbox("Apply Pre-emphasis (alpha=0.97)", value=True,
                                  help="Boosts higher formant frequencies by flattening glottal roll-off.")

# Ensure sample files exist
for sample in ["data/kaha.wav", "data/kahaan.wav", "data/hello.wav"]:
    if not os.path.exists(sample):
        os.system("python generate_test_audio.py")
        break

if mode == "Single Audio Analysis":
    st.subheader("1. Audio Input")
    input_source = st.radio("Choose audio source:", ["Preset Sample Audio", "Upload Custom WAV"], horizontal=True)
    
    audio_path = None
    if input_source == "Preset Sample Audio":
        sample_choice = st.selectbox("Select preset word:", ["data/kaha.wav (कहा)", "data/kahaan.wav (कहाँ)", "data/hello.wav (Hello)"])
        audio_path = sample_choice.split()[0]
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
        with col_a2:
            st.markdown("##### 🔊 LPC Residual (Excitation Source)")
            # Export residual as temp buffer for streamlit audio
            res_norm = residual / (np.max(np.abs(residual)) + 1e-9) * 0.9
            res_int16 = np.int16(np.clip(res_norm, -1.0, 1.0) * 32767)
            wav_bytes = io.BytesIO()
            wavfile.write(wav_bytes, fs, res_int16)
            st.audio(wav_bytes.getvalue(), format="audio/wav")
            st.caption("Notice: Vocal tract resonances are removed, leaving glottal pulses/noise.")

        # Metric summary cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Frames", f"{len(coeffs)}")
        with m2:
            st.metric("LPC Matrix Shape", f"{coeffs.shape[0]} × {coeffs.shape[1]}")
        with m3:
            st.metric("Mean Prediction Gain", f"{features['mean_prediction_gain_db']:.2f} dB")
        with m4:
            st.metric("Residual RMS Energy", f"{features['mean_residual_rms']:.4f}")

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
            
            # Formant values
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

else:
    # Comparison Mode
    st.subheader("Word Comparison: 'kaha' (कहा) vs 'kahaan' (कहाँ)")
    st.info("Member 2 provides the mathematical extraction (LPC coefficients & residuals). This tool demonstrates how the residual and LPC filter capture differences between oral and nasalized vowels.")
    
    if os.path.exists("data/kaha.wav") and os.path.exists("data/kahaan.wav"):
        from compare_words import compare_pair
        
        if st.button("🚀 Run Comparative Analysis"):
            with st.spinner("Analyzing minimal pair with LPC..."):
                report = compare_pair("data/kaha.wav", "data/kahaan.wav", name1="kaha", name2="kahaan")
                st.success("Comparison completed!")
                
                c_m1, c_m2, c_m3 = st.columns(3)
                with c_m1:
                    st.metric("LPC Cosine Distance", f"{report['mean_lpc_cosine_distance']:.4f}")
                with c_m2:
                    st.metric("kaha Residual RMS", f"{report['word1_mean_residual_rms']:.4f}")
                with c_m3:
                    st.metric("kahaan Residual RMS", f"{report['word2_mean_residual_rms']:.4f}")
                    
                st.write("**Formant Comparison (Syllable 2):**")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.write("Oral 'kaha' Formants:", [f"{f} Hz" for f in report['word1_vowel_formants_hz']])
                with col_f2:
                    st.write("Nasalized 'kahaan' Formants:", [f"{f} Hz" for f in report['word2_vowel_formants_hz']])
                    
                if os.path.exists("outputs/comparison/comparison_kaha_vs_kahaan.png"):
                    st.image("outputs/comparison/comparison_kaha_vs_kahaan.png", caption="Side-by-side Speech & Residual Comparison")
                if os.path.exists("outputs/comparison/spectral_envelope_kaha_vs_kahaan.png"):
                    st.image("outputs/comparison/spectral_envelope_kaha_vs_kahaan.png", caption="LPC Spectral Envelopes: Oral /a/ vs Nasalized /ã/")
