import os
import json
import numpy as np
import random
from scipy.fft import fftshift, fft
from PIL import Image

# ===========================
# Parameters
# ===========================
IMAGE_SIZE = (224, 224)
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
SEED = 42

AUGMENT_TRAIN = True
AUGMENT_OOD = True

# ---------------------------
# Temperature configuration
# ---------------------------
TEMP_TRAIN_RANGE = (10, 35)  # °C range for training
TEMP_OOD_RANGE = (45, 55)     # °C range for OOD (simulate colder/unstable cases)
T_REF = 25                   # Reference temperature (°C)

# ---------------------------
# Physical coefficients
# ---------------------------
GAIN_COEFF_DB_PER_C = -0.02   # amplitude drift per °C (≈ -0.02 dB/°C)
PHASE_COEFF_DEG_PER_C = 0.1   # phase drift per °C (≈ 0.1°/°C)
NOISE_COEFF_DB_PER_C = 0.05   # thermal noise floor increase per °C

BASE_SNR_DB = 30              # reference SNR at T_REF

# FFT parameters
FFT_SIZE = 4096
NUM_STACKS = 128
SAMPLES_PER_SEGMENT = FFT_SIZE * NUM_STACKS

random.seed(SEED)
np.random.seed(SEED)


# ===========================
# Helper functions
# ===========================
def temperature_to_augmentations(temp_c):
    """Convert temperature to amplitude scale, phase drift, and SNR."""
    delta_t = temp_c - T_REF

    # Gain drift (convert dB to linear)
    gain_db = GAIN_COEFF_DB_PER_C * delta_t
    amp_scale = 10 ** (gain_db / 20)

    # Phase drift (degrees)
    phase_drift_deg = PHASE_COEFF_DEG_PER_C * delta_t

    # Noise degradation (lower SNR with temperature)
    snr_db = BASE_SNR_DB - NOISE_COEFF_DB_PER_C * delta_t

    return amp_scale, phase_drift_deg, snr_db


def apply_amplitude_scaling(iq_samples, scale):
    return iq_samples * scale


def apply_phase_drift(iq_samples, phase_drift_deg):
    drift_rad = np.deg2rad(phase_drift_deg)
    phase_ramp = np.exp(1j * np.linspace(0, drift_rad, len(iq_samples)))
    return iq_samples * phase_ramp


def add_acgn(iq_samples, snr_db):
    """Add Additive Complex Gaussian Noise (thermal noise simulation)."""
    signal_power = np.mean(np.abs(iq_samples) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(iq_samples)) + 1j * np.random.randn(len(iq_samples))
    )
    return iq_samples + noise


# ===========================
# FFT Stack Generation
# ===========================
def compute_spectrum(iq_samples, fft_size=1024):
    spectrum = np.abs(fftshift(fft(iq_samples, fft_size)))
    spectrum_db = 20 * np.log10(spectrum + 1e-6)
    spectrum_db -= np.min(spectrum_db)
    spectrum_db /= np.max(spectrum_db)
    spectrum_db *= 255
    return spectrum_db


def generate_fft_stack_image(iq_samples, fft_size=FFT_SIZE, num_stacks=NUM_STACKS, image_size=IMAGE_SIZE):
    num_samples = len(iq_samples)
    stacked_spectrum = np.zeros((num_stacks, fft_size))

    for j in range(num_stacks):
        start_idx = j * fft_size
        chunk = iq_samples[start_idx:start_idx + fft_size]
        if len(chunk) < fft_size:
            break
        stacked_spectrum[j, :] = compute_spectrum(chunk, fft_size)

    stacked_image = stacked_spectrum.flatten()
    stacked_image -= np.min(stacked_image)
    stacked_image /= np.max(stacked_image)
    stacked_image *= 255
    stacked_image = stacked_image.reshape((num_stacks, fft_size))

    image = Image.fromarray(stacked_image.astype(np.uint8))
    image = image.resize(image_size, Image.BILINEAR)
    return image


# ===========================
# Image Writing
# ===========================
def write_fft_image(iq_segment, split_name, out_folder):
    # Choose temperature for augmentation
    if split_name == 'train' and AUGMENT_TRAIN:
        temp_c = np.random.uniform(*TEMP_TRAIN_RANGE)
    elif split_name == 'ood' and AUGMENT_OOD:
        temp_c = np.random.uniform(*TEMP_OOD_RANGE)
    else:
        temp_c = T_REF  # nominal condition

    # Convert temperature to augmentation parameters
    amp_scale, phase_drift_deg, snr_db = temperature_to_augmentations(temp_c)

    # Apply augmentations
    iq_segment = apply_amplitude_scaling(iq_segment, amp_scale)
    iq_segment = apply_phase_drift(iq_segment, phase_drift_deg)
    iq_segment = add_acgn(iq_segment, snr_db)

    # Generate and save image
    image = generate_fft_stack_image(iq_segment)
    os.makedirs(out_folder, exist_ok=True)
    fname = f"fft_T{int(temp_c)}C_{np.random.randint(1e8):08d}.png"
    image.save(os.path.join(out_folder, fname))

    return temp_c, amp_scale, phase_drift_deg, snr_db


# ===========================
# Main Dataset Processing
# ===========================
def process_iq_dataset(iq_dir, output_dir):
    classes = sorted([d for d in os.listdir(iq_dir) if os.path.isdir(os.path.join(iq_dir, d))])

    # Save mapping for training
    class_to_idx = {c: i for i, c in enumerate(classes)}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    for class_name in classes:
        class_path = os.path.join(iq_dir, class_name)
        iq_files = sorted([
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ])

        for iq_path in iq_files:
            print(f"Processing {iq_path}")
            iq = np.fromfile(iq_path, dtype=np.complex64)

            total_segments = len(iq) // SAMPLES_PER_SEGMENT
            segments = [
                (i * SAMPLES_PER_SEGMENT, (i + 1) * SAMPLES_PER_SEGMENT)
                for i in range(total_segments)
            ]

            random.Random(SEED).shuffle(segments)

            n = len(segments)
            n_train = int(n * TRAIN_SPLIT)
            n_val = int(n * VAL_SPLIT)

            idx_train = segments[:n_train]
            idx_val = segments[n_train:n_train + n_val]
            idx_test = segments[n_train + n_val:]

            # Write each split
            for (s, e) in idx_train:
                out = os.path.join(output_dir, "train", class_name)
                write_fft_image(iq[s:e], "train", out)

            for (s, e) in idx_val:
                out = os.path.join(output_dir, "val", class_name)
                write_fft_image(iq[s:e], "val", out)

            for (s, e) in idx_test:
                out = os.path.join(output_dir, "test", class_name)
                write_fft_image(iq[s:e], "test", out)

            # OOD generation
            if AUGMENT_OOD:
                for (s, e) in idx_test:
                    out = os.path.join(output_dir, "ood", class_name)
                    write_fft_image(iq[s:e], "ood", out)


if __name__ == "__main__":
    IQ_DIR = r"C:\GNURadio\Quantify_Corrosion\25 cm\SPREAD"
    OUTPUT_DIR = r"output"
    process_iq_dataset(IQ_DIR, OUTPUT_DIR)
