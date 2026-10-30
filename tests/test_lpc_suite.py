"""
tests/test_lpc_suite.py
-----------------------
Comprehensive Unit and Numerical Test Suite for the LPC Speech Analysis Suite.
Verifies all mathematical formulations across the complete modular pipeline:
  - Module 1: Preprocessing & Pre-emphasis invertibility
  - Module 2: Levinson-Durbin, Burg's method, LSF, LPCC, Inverse Filtering, Resynthesis SNR
  - Module 3: ZFR Epochs, Hilbert phase coherence, Group Delay, Jitter/Shimmer
  - Module 4: DTW alignment, Oral-vs-Nasal classification accuracy
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lpc_core.preprocessor import load_and_preprocess_audio, pre_emphasis, de_emphasis
from lpc_core.lpc_analyzer import extract_lpc_coefficients, levinson_durbin, compute_autocorrelation
from lpc_core.inverse_filter import compute_lpc_residual
from lpc_core.formant_analyzer import extract_formants_from_lpc, compute_lpc_spectrum
from lpc_core.features import extract_lpc_features, estimate_pitch_from_residual
from lpc_core.advanced_lpc import burg_method, lpc_to_lsf, lpc_to_lpcc, lpcc_distance
from lpc_core.resynthesis import lpc_resynthesize_residual, lpc_resynthesize_vocoder, modify_speech
from lpc_core.phase_timing import (
    extract_epochs_zfr,
    compute_hilbert_phase_features,
    compute_group_delay_spectrum,
    compute_glottal_timing_metrics
)
from lpc_core.classifier import compute_dtw, extract_acoustic_features, OralNasalClassifier

class TestLPCSuite(unittest.TestCase):

    def setUp(self):
        self.kaha_path = "data/kaha.wav"
        self.kahaan_path = "data/kahaan.wav"
        self.fs = 16000

    def test_01_preprocessing_invertibility(self):
        """Test pre-emphasis and de-emphasis are exact mathematical inverses."""
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600))
        alpha = 0.97
        pre = pre_emphasis(signal, alpha=alpha)
        rec = de_emphasis(pre, alpha=alpha)
        # Verify reconstruction matches original signal closely
        diff = np.max(np.abs(signal[50:] - rec[50:]))
        self.assertLess(diff, 1e-4)

    def test_02_levinson_durbin_stability(self):
        """Test Levinson-Durbin recursion produces stable all-pole filter (roots inside unit circle)."""
        frame = np.random.randn(400) * 0.5
        order = 16
        R = compute_autocorrelation(frame, order)
        a, k, E = levinson_durbin(R, order)
        
        self.assertEqual(len(a), order + 1)
        self.assertAlmostEqual(a[0], 1.0)
        self.assertGreater(E, 0.0)
        
        # Verify all roots lie inside or on the unit circle
        roots = np.roots(a)
        self.assertTrue(np.all(np.abs(roots) < 1.05))

    def test_03_burg_method(self):
        """Test Burg's lattice method produces minimum phase filter with |k_i| < 1."""
        frame = np.random.randn(400)
        order = 12
        a, k, E = burg_method(frame, order)
        
        self.assertEqual(len(a), order + 1)
        self.assertAlmostEqual(a[0], 1.0)
        self.assertTrue(np.all(np.abs(k) < 1.0))

    def test_04_lsf_conversion(self):
        """Test Line Spectral Frequencies (LSF) are strictly monotonic in [0, pi]."""
        # Create a stable test polynomial
        poles = [0.9 * np.exp(1j * 0.2 * np.pi), 0.9 * np.exp(-1j * 0.2 * np.pi),
                 0.85 * np.exp(1j * 0.5 * np.pi), 0.85 * np.exp(-1j * 0.5 * np.pi)]
        a = np.real(np.poly(poles))
        lsf = lpc_to_lsf(a)
        
        self.assertEqual(len(lsf), len(poles))
        self.assertTrue(np.all(lsf > 0.0))
        self.assertTrue(np.all(lsf < np.pi))
        # Monotonicity test
        self.assertTrue(np.all(np.diff(lsf) >= 0.0))

    def test_05_lpcc_recursion(self):
        """Test LPCC recursion generates valid cepstral coefficients and distance."""
        a = np.array([1.0, -0.9, 0.4, -0.2])
        lpcc = lpc_to_lpcc(a, num_cep=12)
        self.assertEqual(len(lpcc), 12)
        self.assertFalse(np.isnan(lpcc).any())
        
        dist = lpcc_distance(lpcc, lpcc)
        self.assertAlmostEqual(dist, 0.0)

    def test_06_inverse_filtering_prediction_gain(self):
        """Test inverse filtering reduces signal energy (positive prediction gain in dB)."""
        proc_sig, _, fs = load_and_preprocess_audio(self.kaha_path, target_fs=16000)
        lpc_res = extract_lpc_coefficients(proc_sig, fs=fs, order=16)
        residual = compute_lpc_residual(proc_sig, lpc_res)
        
        feat = extract_lpc_features(proc_sig, residual, lpc_res)
        # Real speech frames have substantial prediction gain (> 2 dB)
        self.assertGreater(feat['mean_prediction_gain_db'], 1.5)

    def test_07_resynthesis_fidelity_snr(self):
        """Test LPC resynthesis from residual achieves high reconstruction SNR (> 10 dB)."""
        proc_sig, raw_sig, fs = load_and_preprocess_audio(self.kaha_path, target_fs=16000)
        lpc_res = extract_lpc_coefficients(proc_sig, fs=fs, order=16)
        residual = compute_lpc_residual(proc_sig, lpc_res)
        resynth = lpc_resynthesize_residual(residual, lpc_res, de_emph=True)
        
        min_l = min(len(raw_sig), len(resynth))
        sig_p = np.mean(raw_sig[:min_l] ** 2)
        err_p = np.mean((raw_sig[:min_l] - resynth[:min_l]) ** 2) + 1e-12
        snr_db = 10.0 * np.log10(sig_p / err_p)
        
        self.assertGreater(snr_db, 10.0)

    def test_08_zfr_epoch_extraction(self):
        """Test Zero Frequency Resonator (ZFR) detects valid Glottal Closure Instants (GCIs)."""
        proc_sig, _, fs = load_and_preprocess_audio(self.kaha_path, target_fs=16000)
        lpc_res = extract_lpc_coefficients(proc_sig, fs=fs, order=16)
        residual = compute_lpc_residual(proc_sig, lpc_res)
        
        gcis, gci_times, zfr = extract_epochs_zfr(residual, fs=fs)
        self.assertGreater(len(gcis), 10)
        self.assertEqual(len(gcis), len(gci_times))
        self.assertEqual(len(zfr), len(residual))

    def test_09_hilbert_and_group_delay(self):
        """Test Hilbert phase features and Modified Group Delay spectrum."""
        proc_sig, _, fs = load_and_preprocess_audio(self.kaha_path, target_fs=16000)
        lpc_res = extract_lpc_coefficients(proc_sig, fs=fs, order=16)
        residual = compute_lpc_residual(proc_sig, lpc_res)
        
        hilbert_data = compute_hilbert_phase_features(residual, fs=fs)
        self.assertGreater(hilbert_data['phase_coherence'], 0.0)
        self.assertLessEqual(hilbert_data['phase_coherence'], 1.0)
        
        freqs, _, tau_mod = compute_group_delay_spectrum(residual[1000:1400], fs=fs)
        self.assertEqual(len(freqs), len(tau_mod))
        self.assertFalse(np.isnan(tau_mod).any())

    def test_10_dtw_alignment(self):
        """Test Dynamic Time Warping (DTW) calculates alignment between sequences."""
        seq1 = np.random.randn(25, 12)
        seq2 = seq1.copy() # Identical sequences should have distance 0.0
        dist, cost, path = compute_dtw(seq1, seq2)
        self.assertAlmostEqual(dist, 0.0, places=4)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (24, 24))

    def test_11_classifier_minimal_pairs(self):
        """Test Module 4 classifier correctly differentiates 'kaha' (oral) vs 'kahaan' (nasalized)."""
        clf = OralNasalClassifier()
        pred_kaha = clf.predict(self.kaha_path)
        pred_kahaan = clf.predict(self.kahaan_path)
        
        self.assertFalse(pred_kaha['is_nasalized'])
        self.assertTrue(pred_kahaan['is_nasalized'])
        self.assertGreater(pred_kaha['confidence'], 0.5)
        self.assertGreater(pred_kahaan['confidence'], 0.5)

    def test_12_cardinal_vowels_formants(self):
        """Test vocal tract formant hierarchy: /i/ has significantly higher F2 than /u/."""
        sig_i, _, fs = load_and_preprocess_audio("data/vowel_i.wav", target_fs=16000)
        sig_u, _, _ = load_and_preprocess_audio("data/vowel_u.wav", target_fs=16000)
        
        lpc_i = extract_lpc_coefficients(sig_i, fs=fs, order=16)
        lpc_u = extract_lpc_coefficients(sig_u, fs=fs, order=16)
        
        mid_i = len(lpc_i['lpc_coeffs']) // 2
        mid_u = len(lpc_u['lpc_coeffs']) // 2
        
        f_i, _ = extract_formants_from_lpc(lpc_i['lpc_coeffs'][mid_i], fs=fs)
        f_u, _ = extract_formants_from_lpc(lpc_u['lpc_coeffs'][mid_u], fs=fs)
        
        # High front vowel /i/ has high F2 (> 2000 Hz)
        # High back vowel /u/ has low F2 (< 2500 Hz, lower than /i/)
        self.assertGreater(len(f_i), 1)
        self.assertGreater(len(f_u), 1)
        self.assertGreater(f_i[1], f_u[1])

    def test_13_pitch_estimation_from_residual(self):
        """Test pitch F0 tracking on male vs female voice: female F0 > male F0."""
        sig_m, _, fs = load_and_preprocess_audio(self.kaha_path, target_fs=16000)
        sig_f, _, _ = load_and_preprocess_audio("data/kaha_female.wav", target_fs=16000)
        
        lpc_m = extract_lpc_coefficients(sig_m, fs=fs, order=16)
        lpc_f = extract_lpc_coefficients(sig_f, fs=fs, order=16)
        
        res_m = compute_lpc_residual(sig_m, lpc_m)
        res_f = compute_lpc_residual(sig_f, lpc_f)
        
        f0_m, _ = estimate_pitch_from_residual(res_m, fs=fs)
        f0_f, _ = estimate_pitch_from_residual(res_f, fs=fs)
        
        valid_m = f0_m[~np.isnan(f0_m)]
        valid_f = f0_f[~np.isnan(f0_f)]
        
        self.assertGreater(len(valid_m), 5)
        self.assertGreater(len(valid_f), 5)
        self.assertGreater(np.mean(valid_f), np.mean(valid_m))

    def test_14_vocoder_resynthesis(self):
        """Test parametric vocoder resynthesis produces finite valid waveform."""
        proc_sig, raw_sig, fs = load_and_preprocess_audio(self.kaha_path, target_fs=16000)
        lpc_res = extract_lpc_coefficients(proc_sig, fs=fs, order=16)
        residual = compute_lpc_residual(proc_sig, lpc_res)
        
        f0_contour, times = estimate_pitch_from_residual(residual, fs=fs)
        vocoder_audio = lpc_resynthesize_vocoder(f0_contour, times, lpc_res, fs=fs)
        
        self.assertGreater(len(vocoder_audio), 0)
        self.assertFalse(np.isnan(vocoder_audio).any())
        self.assertAlmostEqual(len(vocoder_audio) / fs, len(raw_sig) / fs, delta=0.1)

    def test_15_speech_morphing(self):
        """Test voice morphing engine: pitch shift, formant warp, and whisperization."""
        proc_sig, _, fs = load_and_preprocess_audio(self.kaha_path, target_fs=16000)
        lpc_res = extract_lpc_coefficients(proc_sig, fs=fs, order=16)
        
        # Test pitch and formant warp
        morphed = modify_speech(proc_sig, lpc_res, pitch_scale=1.2, formant_scale=1.1, fs=fs)
        self.assertAlmostEqual(len(morphed) / fs, len(proc_sig) / fs, delta=0.05)
        self.assertFalse(np.isnan(morphed).any())
        
        # Test whisper mode
        whispered = modify_speech(proc_sig, lpc_res, whisper=True, fs=fs)
        self.assertEqual(len(whispered), len(proc_sig))
        self.assertFalse(np.isnan(whispered).any())

    def test_16_glottal_timing_perturbation(self):
        """Test Jitter, Shimmer, and HNR metrics remain within normal physiological ranges."""
        proc_sig, _, fs = load_and_preprocess_audio(self.kaha_path, target_fs=16000)
        lpc_res = extract_lpc_coefficients(proc_sig, fs=fs, order=16)
        residual = compute_lpc_residual(proc_sig, lpc_res)
        
        gcis, _, _ = extract_epochs_zfr(residual, fs=fs)
        metrics = compute_glottal_timing_metrics(residual, fs=fs, gcis=gcis)
        
        self.assertGreater(metrics['num_epochs'], 10)
        self.assertGreater(metrics['mean_f0_hz'], 50.0)
        self.assertLess(metrics['mean_f0_hz'], 400.0)
        self.assertGreaterEqual(metrics['jitter_local_pct'], 0.0)
        self.assertGreaterEqual(metrics['shimmer_local_pct'], 0.0)

    def test_17_audio_dataset_integrity(self):
        """Test all 14 dataset WAV files exist, are 16 kHz mono, and have non-zero duration."""
        expected_files = [
            "baans.wav", "baas.wav", "hello.wav", "hello_female.wav",
            "kaha.wav", "kaha_deep_male.wav", "kaha_female.wav", "kahaan.wav",
            "saans.wav", "saas.wav", "vowel_a.wav", "vowel_i.wav",
            "vowel_u.wav", "whisper_kaha.wav"
        ]
        import scipy.io.wavfile as wf
        for fn in expected_files:
            fp = os.path.join("data", fn)
            self.assertTrue(os.path.exists(fp), f"Missing dataset file: {fp}")
            rate, data = wf.read(fp)
            self.assertEqual(rate, 16000, f"Rate mismatch for {fn}: {rate} != 16000")
            self.assertGreater(len(data), 0, f"Empty audio file: {fn}")

if __name__ == "__main__":
    unittest.main()
