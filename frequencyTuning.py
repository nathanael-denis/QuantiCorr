import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from scipy.signal import welch
from scipy.interpolate import make_interp_spline

# === Parameters ===
folder = r"C:\GNURadio\Quantify_Corrosion\Frequency Calibration Files - Fundamental"   # <-- Change this to your actual path
sample_rate = 2e6                    # Hz (set to your actual sampling rate)
nperseg = 262144                     # FFT segment length
file_pattern = "*GHZ"                # Match all your IQ files

# === Helper Functions ===
def read_iq(filename):
    """Read interleaved float32 IQ samples."""
    raw = np.fromfile(filename, dtype=np.float32)
    iq = raw[::2] + 1j * raw[1::2]
    return iq

def get_fundamental_from_name(name):
    """Extract fundamental frequency (GHz) from filename."""
    match = re.search(r"(\d+(?:\.\d+)?)GHZ", name.upper())
    if match:
        return float(match.group(1))
    return None

def compute_peak_dbfs(iq, fs):
    """Compute peak PSD level in dBFS using Welch method."""
    f, Pxx = welch(iq, fs=fs, nperseg=nperseg, return_onesided=False)
    Pxx_dB = 10 * np.log10(np.abs(Pxx) + 1e-20)
    return np.max(Pxx_dB)

# === Main Processing ===
fundamentals = []
harmonic_magnitudes = []

for file in sorted(glob.glob(f"{folder}\\{file_pattern}")):
    f0 = get_fundamental_from_name(file)
    if f0 is None:
        continue

    iq = read_iq(file)
    peak_dbfs = compute_peak_dbfs(iq, sample_rate)

    fundamentals.append(f0)
    harmonic_magnitudes.append(peak_dbfs)

    print(f"Fundamental: {f0:.2f} GHz → Harmonic peak: {peak_dbfs:.2f} dBFS")

# === Sort the data ===
fundamentals = np.array(fundamentals)
harmonic_magnitudes = np.array(harmonic_magnitudes)
sort_idx = np.argsort(fundamentals)
fundamentals = fundamentals[sort_idx]
harmonic_magnitudes = harmonic_magnitudes[sort_idx]

# === Interpolate for smooth curve ===
x_smooth = np.linspace(fundamentals.min(), fundamentals.max(), 300)
y_smooth = make_interp_spline(fundamentals, harmonic_magnitudes)(x_smooth)

# === Plot ===
plt.figure(figsize=(8,5))
plt.plot(x_smooth, y_smooth, '-', linewidth=2, color='royalblue', label='Smoothed Curve')
plt.scatter(fundamentals, harmonic_magnitudes, color='black', s=40, zorder=3, label='Measured Points')
#plt.title("Second Harmonic Magnitude vs Fundamental Frequency")
plt.xlabel("Fundamental Frequency (GHz)")
plt.ylabel("Magnitude at 2f₀ (dBFS)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
