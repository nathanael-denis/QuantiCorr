import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from scipy.signal import welch
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D

# === Parameters ===
sample_rate = 2e6
nperseg = 262144
file_pattern = "*GHZ"

# === Global font scaling ===
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
})

# === Folder paths and colors ===
folders = {
    "Goethite": {"path": r"C:\GNURadio\Quantify_Corrosion\Goethite Calibration", "color": "goldenrod"},
    "Magnetite": {"path": r"C:\GNURadio\Quantify_Corrosion\Magnetite Calibration", "color": "black"},
    "Rust compound": {"path": r"C:\GNURadio\Quantify_Corrosion\Calibration - Rust compound", "color": "#8B2500"}  # reddish-brown
}

def read_iq(filename):
    raw = np.fromfile(filename, dtype=np.float32)
    return raw[::2] + 1j * raw[1::2]

def get_fundamental_from_name(name):
    match = re.search(r"(\d+(?:\.\d+)?)GHZ", name.upper())
    return float(match.group(1)) if match else None

def compute_peak_dbfs(iq, fs):
    f, Pxx = welch(iq, fs=fs, nperseg=nperseg, return_onesided=False)
    return np.max(10 * np.log10(np.abs(Pxx) + 1e-20))

plt.figure(figsize=(8,5))
legend_handles = []

for label, info in folders.items():
    folder = info["path"]
    fundamentals, magnitudes = [], []

    for file in sorted(glob.glob(f"{folder}\\{file_pattern}")):
        f0 = get_fundamental_from_name(file)
        if f0 is None:
            continue
        iq = read_iq(file)
        fundamentals.append(f0)
        magnitudes.append(compute_peak_dbfs(iq, sample_rate))
        print(f"[{label}] {f0:.2f} GHz → Peak: {magnitudes[-1]:.2f} dBFS")

    if not fundamentals:
        continue

    fundamentals = np.array(fundamentals)
    magnitudes = np.array(magnitudes)
    idx = np.argsort(fundamentals)
    fundamentals, magnitudes = fundamentals[idx], magnitudes[idx]

    x_smooth = np.linspace(fundamentals.min(), fundamentals.max(), 300)
    y_smooth = make_interp_spline(fundamentals, magnitudes)(x_smooth)

    plt.plot(x_smooth, y_smooth, color=info["color"], linewidth=2.2, label=label)
    plt.scatter(fundamentals, magnitudes, s=45, color=info["color"], edgecolor='black', zorder=3)
    legend_handles.append(Line2D([], [], color=info["color"], lw=2.2, label=label))

# Final formatting
plt.xlabel("Fundamental Frequency (GHz)")
plt.ylabel("Magnitude (dBFS)")
plt.legend(handles=legend_handles)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
