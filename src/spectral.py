import numpy as np

def hann_window_2d(h, w):
    # Separable 2D window: outer product of two 1D Hann windows
    return np.outer(np.hanning(h), np.hanning(w))

def compute_fft(image):
    # Step 1: Convert RGB image to grayscale
    image = image.astype(np.float32)

    gray = (
        0.299 * image[:, :, 0] +
        0.587 * image[:, :, 1] +
        0.114 * image[:, :, 2]
    )

    gray = gray - gray.mean()
    gray = gray * hann_window_2d(*gray.shape)
    # Step 2: Apply the 2D Fourier transform
    fft = np.fft.fft2(gray)
    fft=np.fft.fftshift(fft)  # Shift the zero frequency component to the center
    return fft

def compute_power_spectrum(fft):
    return np.abs(fft) ** 2



def high_pass_filter(fft_centered, sigma=30):
    h, w = fft_centered.shape

    # Coordinates relative to the spectrum center
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2

    # Squared distance from the center
    D2 = (x - cx)**2 + (y - cy)**2

    # Gaussian high-pass filter
    H = 1 - np.exp(-D2 / (2 * sigma**2))

    # Apply the filter to the FFT
    return fft_centered * H


def radial_profile(power):
    h, w = power.shape
    y, x = np.indices(power.shape)
    r = np.hypot(x - w // 2, y - h // 2).astype(int)
    sums = np.bincount(r.ravel(), weights=power.ravel())
    counts = np.bincount(r.ravel())
    return sums / np.maximum(counts, 1)


def radial_log_profile(image, r_max=256):
    """log10 of the radial power profile of an RGB image (Hann window applied in compute_fft).

    We stop at r_max=256: beyond that, the rings are cut off by the spectrum borders.
    """
    power = compute_power_spectrum(compute_fft(image))
    power = power / power.sum()
    profile = radial_profile(power)[:r_max]
    return np.log10(profile + 1e-12)


def radial_slope(log_profile, r_min=5, r_max=250):
    """Slope of log10(P) versus log10(r), fitted on [r_min, r_max)."""
    r = np.arange(r_min, r_max)
    slope, intercept = np.polyfit(np.log10(r), log_profile[r_min:r_max], 1)
    return slope, intercept


def mean_log_spectrum(images, sigma=10):
    """Mean of log10(power) over a list of RGB images (log BEFORE averaging).

    Averaging the log makes every image weigh about the same, so a few
    high-contrast images do not dominate the class mean.
    """
    acc = 0
    for image in images:
        fft = high_pass_filter(compute_fft(image), sigma)
        acc = acc + np.log10(compute_power_spectrum(fft) + 1e-12)
    return acc / len(images)


# ---------------------------------------------------------------------------
# Spectral feature vector (Part A)
# ---------------------------------------------------------------------------

BANDS = [(1, 16), (16, 64), (64, 128), (128, 192), (192, 256)]   # radius ranges

SPECTRAL_FEATURE_NAMES = (
    [f"band_E_{a}_{b}" for a, b in BANDS]                          # 5
    + ["hf_lf_ratio", "radial_slope", "radial_fit_std", "nyquist_dev"]   # 4
    + ["nyquist_axis_peak"]                                              # 1
)                                                                        # = 10


def extract_spectral_features(image):
    """Extract 10 spectral features from a 512x512 RGB image (uint8).

    All features are computed on the power spectrum normalized by its total
    energy, so they describe the SHAPE of the spectrum, not the contrast.

    - band_E_a_b: log10 of the energy fraction in the ring a <= r < b.
    - hf_lf_ratio: log10 of (energy r >= 128) / (energy r < 64).
    - radial_slope: slope of log10 P(r) versus log10 r (r in [5, 250)).
    - radial_fit_std: std of the residual of that line (roughness of the curve).
    - nyquist_dev: mean residual of the line for r in [200, 250) (HF bend).
    - nyquist_axis_peak: log10 of the power at the Nyquist points of the axes
      divided by the median power of the ring 200 <= r < 256.
    """
    if image.shape != (512, 512, 3):
        raise ValueError("Input image must have shape (512, 512, 3).")

    eps = 1e-12
    power = compute_power_spectrum(compute_fft(image))
    power = power / power.sum()
    h, w = power.shape
    y, x = np.indices(power.shape)
    r = np.hypot(x - w // 2, y - h // 2).astype(int)

    features = []

    # (a) energy per frequency band, then high/low frequency ratio
    energies = [power[(r >= a) & (r < b)].sum() for a, b in BANDS]
    features += [np.log10(e + eps) for e in energies]
    features.append(np.log10((energies[3] + energies[4] + eps) / (energies[0] + energies[1] + eps)))

    # (b) shape of the radial profile
    log_profile = np.log10(radial_profile(power)[:256] + eps)
    slope, intercept = radial_slope(log_profile)
    radii = np.arange(5, 250)
    residual = log_profile[5:250] - (slope * np.log10(radii) + intercept)
    features += [slope, residual.std(), residual[radii >= 200].mean()]

    # (c) Nyquist points of the axes (period-2 patterns)
    nyquist_points = [power[0, w // 2], power[h // 2, 0], power[0, 0]]
    reference = np.median(power[(r >= 200) & (r < 256)])
    features.append(np.log10(np.mean(nyquist_points) / (reference + eps) + eps))

    return np.asarray(features, dtype=np.float64)
