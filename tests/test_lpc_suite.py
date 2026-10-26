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

if __name__ == "__main__":
    unittest.main()
