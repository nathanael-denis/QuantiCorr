"""
compute_snr_harmonic_radar_v2.py

Improved version:
- Higher PSD resolution (nperseg=8192)
- Wider signal region (signal_bw_hz=300 kHz)
- More stable SNR computation (handles low-power signals gracefully)
- Processes only '5 cm' and '25 cm' directories
"""

import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch


def read_iq_file(path, fmt='float32', dtype_scale=1.0):
    if fmt == 'float32':
        dtype = np.float32
    elif fmt == 'int16':
        dtype = np.int16
    elif fmt == 'uint8':
        dtype = np.uint8
    else:
        raise ValueError('Unsupported format')

    raw = np.fromfile(path, dtype=dtype)
    if len(raw) % 2 != 0:
        raw = raw[:-1]

    i = raw[0::2].astype(np.float64)
    q = raw[1::2].astype(np.float64)
    return (i + 1j * q) * dtype_scale


def psd_snr(iq, fs, signal_bw_hz=3e5):
    x = iq - np.mean(iq)
    if len(x) < 256:
        return np.nan, 0.0, 0.0  # too short to process reliably

    # Use large nperseg for better frequency resolution
    nperseg = 8192 if len(x) >= 8192 else 2 ** int(np.floor(np.log2(len(x))))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, nfft=nperseg,
                   return_onesided=False, scaling='density')
    Pxx = np.fft.fftshift(Pxx)
    f = np.fft.fftshift(f) - fs / 2

    idx_peak = np.argmax(Pxx)
    bin_hz = fs / len(Pxx)
    half_bins = max(1, int(np.round((signal_bw_hz / 2.0) / bin_hz)))

    sig_idx = np.arange(max(0, idx_peak - half_bins),
                        min(len(Pxx), idx_peak + half_bins + 1))

    signal_power = np.sum(Pxx[sig_idx])

    noise_bins = np.setdiff1d(np.arange(len(Pxx)), sig_idx)
    if len(noise_bins) == 0:
        return np.nan, signal_power, 0.0

    noise_power = np.mean(Pxx[noise_bins]) * len(noise_bins)
    eps = 1e-20

    snr = 10 * np.log10((signal_power + eps) / (noise_power + eps))
    if not np.isfinite(snr):
        snr = np.nan

    return float(snr), float(signal_power), float(noise_power)


def process_all(root_dir, sample_rate, fmt, dtype_scale, signal_bw_hz):
    root = Path(root_dir)
    valid_distances = ['5 cm', '25 cm']
    rows = []

    for distance_dir in sorted(root.iterdir()):
        if not distance_dir.is_dir() or distance_dir.name not in valid_distances:
            continue
        distance = distance_dir.name
        for geometry_dir in sorted(distance_dir.iterdir()):
            if not geometry_dir.is_dir():
                continue
            geometry = geometry_dir.name
            for quantity_dir in sorted(geometry_dir.iterdir()):
                if not quantity_dir.is_dir():
                    continue
                quantity = quantity_dir.name
                files = [p for p in quantity_dir.iterdir() if p.is_file()]
                if not files:
                    continue
                iq_path = files[0]
                try:
                    iq = read_iq_file(iq_path, fmt=fmt, dtype_scale=dtype_scale)
                    snr_db, spow, npow = psd_snr(iq, fs=sample_rate, signal_bw_hz=signal_bw_hz)
                    rows.append({
                        'distance': distance,
                        'geometry': geometry,
                        'quantity': quantity,
                        'filepath': str(iq_path),
                        'snr_db': snr_db,
                        'signal_power': spow,
                        'noise_power': npow
                    })
                    print(f"[OK] {distance}/{geometry}/{quantity} -> SNR={snr_db:.2f} dB")
                except Exception as e:
                    print(f"[ERROR] {iq_path}: {e}")

    return pd.DataFrame(rows)


def compute_aggregates(df):
    summary = {}
    for dist in ['5 cm', '25 cm']:
        m = df['distance'].astype(str).str.contains(dist)
        summary[f'avg_snr_{dist}'] = float(df.loc[m, 'snr_db'].mean()) if m.sum() > 0 else None

    for dist in ['5 cm', '25 cm']:
        m = df['geometry'].astype(str).str.contains('spread') & df['distance'].astype(str).str.contains(dist)
        summary[f'avg_snr_spread_{dist}'] = float(df.loc[m, 'snr_db'].mean()) if m.sum() > 0 else None

    quantities = sorted(df['quantity'].astype(str).unique(),
                        key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else x)
    summary['snr_by_quantity'] = {}
    for q in quantities:
        summary['snr_by_quantity'][q] = {}
        for dist in ['5 cm', '25 cm']:
            m = df['quantity'].astype(str).str.fullmatch(q) & df['distance'].astype(str).str.contains(dist)
            summary['snr_by_quantity'][q][dist] = float(df.loc[m, 'snr_db'].mean()) if m.sum() > 0 else None

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=r'C:\\GNURadio\\Quantify_Corrosion',
                   help='Root directory containing 5 cm and 25 cm subdirectories')
    p.add_argument('--sample_rate', type=float, default=500000, help='Sample rate in Hz (e.g., 8e6)')
    p.add_argument('--fmt', default='float32', choices=['float32', 'int16', 'uint8'])
    p.add_argument('--dtype_scale', type=float, default=1.0)
    p.add_argument('--signal_bw_hz', type=float, default=3e5)
    p.add_argument('--out_csv', default='snr_results.csv')
    p.add_argument('--out_summary', default='snr_summary.json')
    args = p.parse_args()

    df = process_all(args.root, args.sample_rate, args.fmt, args.dtype_scale, args.signal_bw_hz)
    if df.empty:
        print('No IQ files processed.')
        sys.exit(1)

    df.to_csv(args.out_csv, index=False)
    summary = compute_aggregates(df)
    with open(args.out_summary, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
