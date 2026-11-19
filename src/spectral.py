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
    profile = radial_profile(power)[:r_max]
    return np.log10(profile + 1e-12)


def radial_slope(log_profile, r_min=5, r_max=250):
    """Slope of log10(P) versus log10(r), fitted on [r_min, r_max)."""
    r = np.arange(r_min, r_max)
    slope, intercept = np.polyfit(np.log10(r), log_profile[r_min:r_max], 1)
    return slope, intercept
