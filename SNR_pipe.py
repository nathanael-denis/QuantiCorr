"""
compute_snr_steelpipe_safe.py
Robust version: prevents NaN SNR, handles flat or empty IQ signals gracefully
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
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
    if len(raw) < 4:
        return np.array([], dtype=np.complex64)
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    i = raw[0::2].astype(np.float64)
    q = raw[1::2].astype(np.float64)
    return (i + 1j * q) * dtype_scale

def psd_snr(iq, fs, signal_bw_hz=5e5):
    if len(iq) < 256:
        return np.nan, 0.0, 0.0, "too_short"

    x = iq - np.mean(iq)
    if np.allclose(x, 0, atol=1e-12):
        return np.nan, 0.0, 0.0, "flat_signal"

    nperseg = max(256, min(len(x), 8192))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, nfft=nperseg,
                   return_onesided=False, scaling='density')
    Pxx = np.fft.fftshift(Pxx)

    if np.all(Pxx == 0):
        return np.nan, 0.0, 0.0, "flat_psd"

    idx_peak = np.argmax(Pxx)
    bin_hz = fs / len(Pxx)
    half_bins = max(1, int(np.round((signal_bw_hz / 2.0) / bin_hz)))

    sig_idx = np.arange(max(0, idx_peak - half_bins),
                        min(len(Pxx), idx_peak + half_bins + 1))
    signal_power = np.sum(Pxx[sig_idx])

    noise_bins = np.setdiff1d(np.arange(len(Pxx)), sig_idx)
    noise_floor = np.median(Pxx[noise_bins]) if len(noise_bins) > 0 else 1e-12
    noise_power = noise_floor * len(sig_idx) + 1e-12

    if noise_power <= 0 or signal_power <= 0:
        return np.nan, signal_power, noise_power, "invalid_powers"

    snr = 10 * np.log10(signal_power / noise_power)
    if np.isnan(snr) or np.isinf(snr):
        return np.nan, signal_power, noise_power, "bad_ratio"

    return float(snr), float(signal_power), float(noise_power), "ok"

def process_steelpipe(root_dir, sample_rate, fmt, dtype_scale, signal_bw_hz):
    root = Path(root_dir)
    rows = []

    for iq_path in root.rglob('*'):
        if iq_path.is_file():
            try:
                iq = read_iq_file(iq_path, fmt=fmt, dtype_scale=dtype_scale)
                snr_db, spow, npow, status = psd_snr(iq, fs=sample_rate, signal_bw_hz=signal_bw_hz)
                if np.isnan(snr_db):
                    print(f"[WARN] {iq_path} -> SNR=NaN ({status})")
                else:
                    print(f"[OK] Processed {iq_path} -> SNR={snr_db:.2f} dB")
                rows.append({
                    'filepath': str(iq_path),
                    'snr_db': snr_db,
                    'signal_power': spow,
                    'noise_power': npow,
                    'status': status
                })
            except Exception as e:
                print(f"[ERROR] {iq_path}: {e}")

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=r'C:\\GNURadio\\Quantify_Corrosion\\SteelPipe')
    parser.add_argument('--sample_rate', required=True, type=float)
    parser.add_argument('--fmt', default='float32', choices=['float32', 'int16', 'uint8'])
    parser.add_argument('--dtype_scale', type=float, default=1.0)
    parser.add_argument('--signal_bw_hz', type=float, default=5e5)
    parser.add_argument('--out_csv', default='snr_steelpipe.csv')
    parser.add_argument('--out_summary', default='snr_steelpipe_summary.json')
    args = parser.parse_args()

    df = process_steelpipe(args.root, args.sample_rate, args.fmt, args.dtype_scale, args.signal_bw_hz)
    if df.empty:
        print('No IQ files processed.')
        sys.exit(1)

    df.to_csv(args.out_csv, index=False)

    summary = {
        'mean_snr': float(df['snr_db'].mean(skipna=True)),
        'valid_count': int(df['snr_db'].notna().sum()),
        'nan_count': int(df['snr_db'].isna().sum()),
        'statuses': dict(df['status'].value_counts())
    }

    with open(args.out_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
