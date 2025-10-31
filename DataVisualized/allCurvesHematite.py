import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from scipy.signal import welch
from scipy.interpolate import make_interp_spline

# === Parameters ===
sample_rate = 2e6        # Hz
nperseg = 262144         # FFT segment length
file_pattern = "*GHZ"    # Match pattern for IQ files

# === Global font scaling (increase by ~33%) ===
plt.rcParams.update({
    'font.size': 14,            # base font size (was ~10–11)
    'axes.titlesize': 16,       # title font
    'axes.labelsize': 15,       # axis labels
    'xtick.labelsize': 13,      # x-tick labels
    'ytick.labelsize': 13,      # y-tick labels
    'legend.fontsize': 13,      # legend text
})

# === Folder paths and desired colors ===
folders = {
    "Fundamental": {
        "path": r"C:\GNURadio\Quantify_Corrosion\Frequency Calibration Files - Fundamental",
        "color": "royalblue"
    },
    "Harmonic Stack": {
        "path": r"C:\GNURadio\Quantify_Corrosion\Frequency Calibration Files - Harmonic Stack",
        "color": "saddlebrown"
    },
    "Harmonic Spread": {
        "path": r"C:\GNURadio\Quantify_Corrosion\Frequency Calibration Files - Harmonic Spread",
        "color": "orange"
    }
}

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

# === Processing and plotting ===
plt.figure(figsize=(8,5))

for label, info in folders.items():
    folder = info["path"]
    color = info["color"]
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

        print(f"[{label}] {f0:.2f} GHz → Peak: {peak_dbfs:.2f} dBFS")

    if not fundamentals:
        continue

    fundamentals = np.array(fundamentals)
    harmonic_magnitudes = np.array(harmonic_magnitudes)
    sort_idx = np.argsort(fundamentals)
    fundamentals = fundamentals[sort_idx]
    harmonic_magnitudes = harmonic_magnitudes[sort_idx]

    x_smooth = np.linspace(fundamentals.min(), fundamentals.max(), 300)
    y_smooth = make_interp_spline(fundamentals, harmonic_magnitudes)(x_smooth)

    plt.plot(x_smooth, y_smooth, linewidth=2.2, color=color, label=label)
    plt.scatter(fundamentals, harmonic_magnitudes, s=45, color=color, edgecolor='black', zorder=3)

# === Final formatting ===
plt.xlabel("Fundamental Frequency (GHz)")
plt.ylabel("Magnitude (dBFS)")
#plt.title("Comparison of Harmonic Responses")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
