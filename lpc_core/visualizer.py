"""
lpc_core/visualizer.py
----------------------
Matplotlib publication-quality plotting utilities for LPC analysis,
residual waveforms, spectrograms, formant envelopes, and acoustic comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from .formant_analyzer import compute_lpc_spectrum, extract_formants_from_lpc

# Styling defaults for crisp, professional graphs
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_waveform_and_residual(speech, residual, fs, title="Speech Waveform vs LPC Residual", save_path=None):
    """
    Plots the original speech waveform and the corresponding LPC residual.
    """
    time_speech = np.arange(len(speech)) / fs
    time_res = np.arange(len(residual)) / fs
    
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    
    # 1. Original Speech
    axes[0].plot(time_speech, speech, color='#1f77b4', lw=0.9, alpha=0.9)
    axes[0].set_title(f"{title} - Original Speech Waveform x[n]", fontsize=12, fontweight='bold', pad=8)
    axes[0].set_ylabel("Amplitude", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylim(-1.05, 1.05)
    
    # 2. LPC Residual
    axes[1].plot(time_res, residual, color='#d62728', lw=0.8, alpha=0.85)
    axes[1].set_title("LPC Residual Signal e[n] (Prediction Error / Source Excitation)", fontsize=12, fontweight='bold', pad=8)
    axes[1].set_xlabel("Time (seconds)", fontsize=10)
    axes[1].set_ylabel("Amplitude", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

def plot_spectrogram_comparison(speech, residual, fs, title="Spectrogram: Original vs LPC Residual", save_path=None):
    """
    Plots side-by-side spectrograms demonstrating the flattening of vocal-tract formants.
    """
    nperseg = int(0.025 * fs)
    noverlap = int(0.015 * fs)
    
    f_sp, t_sp, Sxx_sp = spectrogram(speech, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_res, t_res, Sxx_res = spectrogram(residual, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    
    im1 = ax1.pcolormesh(t_sp, f_sp, 10 * np.log10(Sxx_sp + 1e-10), shading='gouraud', cmap='magma')
    ax1.set_title("Original Speech (Resonant Formants Visible)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_ylim(0, fs / 2)
    fig.colorbar(im1, ax=ax1, format='%+2.0f dB', pad=0.02)
    
    im2 = ax2.pcolormesh(t_res, f_res, 10 * np.log10(Sxx_res + 1e-10), shading='gouraud', cmap='magma')
    ax2.set_title("LPC Residual (Formants Stripped, Flattened Excitation)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time (s)")
    fig.colorbar(im2, ax=ax2, format='%+2.0f dB', pad=0.02)
    
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

def plot_lpc_spectral_envelope(frame, a, fs, frame_time=0.0, save_path=None):
    """
    Plots the discrete Fourier transform of a speech frame against the LPC all-pole model envelope.
    """
    N = len(frame)
    win_frame = frame * np.hamming(N)
    
    # FFT spectrum
    n_fft = 1024
    fft_spec = np.abs(np.fft.rfft(win_frame, n=n_fft))
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
    fft_db = 20 * np.log10(np.maximum(fft_spec, 1e-8))
    
    # LPC Model spectrum
    lpc_freqs, lpc_db = compute_lpc_spectrum(a, fs=fs, n_fft=512)
    
    # Normalize scales for visual alignment
    fft_db -= np.max(fft_db)
    lpc_db -= np.max(lpc_db)
    
    # Formants
    formants, _ = extract_formants_from_lpc(a, fs=fs)
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(fft_freqs, fft_db, color='#888888', alpha=0.6, label='FFT Spectrum (Harmonics)', lw=1.0)
    ax.plot(lpc_freqs, lpc_db, color='#d62728', lw=2.0, label='LPC All-Pole Filter 1/|A(f)| (Vocal Tract Envelope)')
    
    # Annotate formants
    for idx, f in enumerate(formants[:4]):
        ax.axvline(f, color='#2ca02c', linestyle=':', alpha=0.8, lw=1.2)
        ax.text(f + 40, -10 - idx * 6, f"F{idx+1}={int(f)}Hz", color='#1b611b', fontsize=9, fontweight='bold')
        
    ax.set_title(f"Speech Spectrum vs LPC Formant Envelope (t = {frame_time:.3f}s)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Normalized Magnitude (dB)", fontsize=10)
    ax.set_xlim(0, fs / 2)
    ax.set_ylim(-50, 5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

def plot_word_comparison(res1, res2, name1="Word A (kaha)", name2="Word B (kahaan)", save_path=None):
    """
    Comprehensive multi-panel comparison between two words.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex='col')
    
    # Word 1 Speech & Residual
    t1 = np.arange(len(res1['speech'])) / res1['fs']
    axes[0, 0].plot(t1, res1['speech'], color='#1f77b4', lw=0.8)
    axes[0, 0].set_title(f"{name1} - Original Speech", fontweight='bold')
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.4)
    
    axes[1, 0].plot(t1, res1['residual'], color='#d62728', lw=0.8)
    axes[1, 0].set_title(f"{name1} - LPC Residual", fontweight='bold')
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.4)
    
    # Residual RMS Energy
    t_rms1 = res1['feat'].get('frame_times', np.linspace(0, t1[-1], len(res1['feat']['residual_rms'])))
    axes[2, 0].plot(t_rms1, res1['feat']['residual_rms'], color='#2ca02c', lw=1.5)
    axes[2, 0].set_title(f"{name1} - Residual RMS Energy Contour", fontweight='bold')
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("RMS Energy")
    axes[2, 0].grid(True, alpha=0.4)
    
    # Word 2 Speech & Residual
    t2 = np.arange(len(res2['speech'])) / res2['fs']
    axes[0, 1].plot(t2, res2['speech'], color='#1f77b4', lw=0.8)
    axes[0, 1].set_title(f"{name2} - Original Speech", fontweight='bold')
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.4)
    
    axes[1, 1].plot(t2, res2['residual'], color='#d62728', lw=0.8)
    axes[1, 1].set_title(f"{name2} - LPC Residual", fontweight='bold')
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(True, alpha=0.4)
    
    t_rms2 = res2['feat'].get('frame_times', np.linspace(0, t2[-1], len(res2['feat']['residual_rms'])))
    axes[2, 1].plot(t_rms2, res2['feat']['residual_rms'], color='#2ca02c', lw=1.5)
    axes[2, 1].set_title(f"{name2} - Residual RMS Energy Contour", fontweight='bold')
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("RMS Energy")
    axes[2, 1].grid(True, alpha=0.4)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

def plot_zplane_poles(a, fs=16000, title="LPC All-Pole Filter in Z-Plane", save_path=None):
    """
    Plots the roots of the LPC polynomial A(z) in the complex z-plane.
    Shows the unit circle, pole positions (x), and formant angles.
    """
    roots = np.roots(a)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#555555', linestyle='--', lw=1.2, label='Unit Circle (|z|=1)')
    
    # Axes
    ax.axhline(0, color='#aaaaaa', lw=0.8)
    ax.axvline(0, color='#aaaaaa', lw=0.8)
    
    # Poles
    ax.scatter(np.real(roots), np.imag(roots), color='#d62728', marker='x', s=70, lw=2.0, label='LPC Poles (Resonances)')
    
    # Annotate formants in upper half
    formants, _ = extract_formants_from_lpc(a, fs=fs)
    for i, f in enumerate(formants[:3]):
        angle = 2.0 * np.pi * f / fs
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color='#2ca02c', linestyle=':', alpha=0.7)
        ax.text(1.08 * np.cos(angle), 1.08 * np.sin(angle), f"F{i+1}\n({int(f)}Hz)", color='#1b611b', fontsize=8, fontweight='bold', ha='center')
        
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("Real Axis", fontsize=10)
    ax.set_ylabel("Imaginary Axis", fontsize=10)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

def plot_dtw_alignment(cost_matrix, path, name1="Word 1", name2="Word 2", dist=0.0, save_path=None):
    """
    Visualizes Dynamic Time Warping (DTW) accumulated cost matrix and optimal warping path.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cost_matrix.T, origin='lower', cmap='viridis', aspect='auto')
    
    # Warping path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, color='#ff7f0e', lw=2.5, label=f'Optimal Path (Dist: {dist:.3f})')
    
    ax.set_title(f"Dynamic Time Warping (DTW): {name1} vs {name2}", fontsize=11, fontweight='bold')
    ax.set_xlabel(f"{name1} Frame Index", fontsize=10)
    ax.set_ylabel(f"{name2} Frame Index", fontsize=10)
    fig.colorbar(im, ax=ax, label="Cumulative Distance")
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

def plot_phase_timing(residual, zfr, gci_indices, fs=16000, title="Epoch (GCI) & Phase Analysis", save_path=None):
    """
    Plots the LPC residual along with the Zero Frequency Resonator (ZFR) signal
    and detected Glottal Closure Instants (GCIs / Epochs).
    """
    # Zoom in to a representative 60 ms voiced segment
    dur_ms = 60.0
    num_samples = int(dur_ms * 1e-3 * fs)
    start_sample = max(0, int(len(residual) * 0.45))
    end_sample = min(len(residual), start_sample + num_samples)
    
    t_axis = (np.arange(start_sample, end_sample) - start_sample) / fs * 1000.0 # in ms
    sub_res = residual[start_sample:end_sample]
    sub_zfr = zfr[start_sample:end_sample]
    
    # GCIs within segment
    mask = (gci_indices >= start_sample) & (gci_indices < end_sample)
    sub_gcis = gci_indices[mask]
    sub_gci_t = (sub_gcis - start_sample) / fs * 1000.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    
    # Residual with Epoch markers
    ax1.plot(t_axis, sub_res, color='#1f77b4', lw=1.0, label='LPC Residual e[n]')
    for gt in sub_gci_t:
        ax1.axvline(gt, color='#d62728', linestyle='--', alpha=0.8, lw=1.2)
    ax1.scatter(sub_gci_t, [0]*len(sub_gci_t), color='#d62728', s=40, zorder=5, label='Epochs (GCIs)')
    ax1.set_title(f"{title} - LPC Residual with Glottal Closure Instants (GCIs)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9)
    
    # ZFR signal with zero crossings
    ax2.plot(t_axis, sub_zfr, color='#2ca02c', lw=1.2, label='ZFR Signal (0-Hz Trend Removed)')
    ax2.axhline(0, color='#666666', linestyle='-', lw=0.8)
    for gt in sub_gci_t:
        ax2.axvline(gt, color='#d62728', linestyle='--', alpha=0.8, lw=1.2)
    ax2.set_title("Zero Frequency Resonator (ZFR) Trend-Removed Signal (Positive Zero Crossings = GCIs)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Normalized ZFR")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

def plot_group_delay(freqs, tau_mod, fs=16000, title="Modified Group Delay Spectrum", save_path=None):
    """
    Plots the Modified Group Delay (MGD) spectrum highlighting formant peaks
    and anti-resonance zeros.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, tau_mod, color='#9467bd', lw=1.5, label='Modified Group Delay')
    ax.axhline(0, color='#888888', linestyle='--', lw=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Group Delay (Arbitrary Units)", fontsize=10)
    ax.set_xlim(0, fs / 2)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

