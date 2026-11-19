"""Method D — RGB & colour analysis.

Idea: a real camera records ONE colour per pixel (Bayer filter) and fills in
the two missing colours by interpolation (demosaicing). This creates strong,
2-periodic links between R, G and B. A generator outputs R, G, B directly,
so these links should look different
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import imageio.v3 as iio
import numpy as np
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis

from src.preprocessing import center_crop

CHANNELS = ["R", "G", "B"]
PAIRS = [(0, 1), (0, 2), (1, 2)]          # (R,G), (R,B), (G,B)
SIGMA_RESIDUAL = 1.0                       # Gaussian scale of the high-pass

# ---------------------------------------------------------------------------
# Helper functions from practicals
# ---------------------------------------------------------------------------


def read_rgb_image(path: str | Path) -> np.ndarray:
    """Read an 8-bit RGB image with imageio. Returns (H, W, 3) uint8."""
    image = iio.imread(path)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]                        # drop alpha if present
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected 8-bit RGB at {path}, got {image.shape}/{image.dtype}")
    return image


def get_channel(image: np.ndarray, channel: int) -> np.ndarray:
    """Extract one channel (0=R, 1=G, 2=B). Returns (H, W)."""
    return image[:, :, channel]


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float64 [0,1]."""
    return image.astype(np.float64) / 255


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Decode sRGB values in [0,1] to linear light (undo the gamma)."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.04045,
                    values / 12.92,
                    ((values + 0.055) / 1.055) ** 2.4)


def compute_luminance(rgb_linear: np.ndarray) -> np.ndarray:
    """Linear luminance Y from linear RGB (Rec. 709 weights)."""
    return (0.2126 * rgb_linear[:, :, 0]
            + 0.7152 * rgb_linear[:, :, 1]
            + 0.0722 * rgb_linear[:, :, 2])


def fft2_centered(image: np.ndarray) -> np.ndarray:
    """Centered 2D DFT (zero frequency in the middle)."""
    return np.fft.fftshift(np.fft.fft2(image))


def spectrum_components(spectrum: np.ndarray):
    """Magnitude, log(1 + magnitude), phase."""
    magnitude = np.abs(spectrum)
    return magnitude, np.log(1 + magnitude), np.angle(spectrum)


def percentile_display(image: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    """Map any real array to [0,1] for display (robust stretch)."""
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float64)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def show_image(ax: Axes, image: np.ndarray, title: str = "", *, signed: bool = False) -> None:
    """Display a grayscale array; signed=True uses a zero-centred colormap."""
    if signed:
        lim = max(1e-12, float(np.max(np.abs(image))))
        ax.imshow(image, cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="nearest")
    else:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Building blocks of method D
# ---------------------------------------------------------------------------

def load_crop(path: str | Path, crop_size: int = 512) -> np.ndarray:
    """Read an image, take the central 512x512 crop, normalise to [0,1]."""
    return normalize_rgb(center_crop(read_rgb_image(path), crop_size))


def high_pass_residual(channel: np.ndarray, sigma: float = SIGMA_RESIDUAL) -> np.ndarray:
    """Residual = image - Gaussian-blurred image.

    The blur keeps the scene (low frequencies); subtracting it keeps only the
    fine high-frequency traces left by the acquisition / generation process.
    """
    return channel - gaussian_filter(channel, sigma=sigma, mode="reflect")


def rgb_residuals(rgb: np.ndarray) -> np.ndarray:
    """High-pass residual of each channel. (H, W, 3) -> (H, W, 3)."""
    return np.stack([high_pass_residual(get_channel(rgb, c)) for c in range(3)], axis=2)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two arrays (0 if one is constant)."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else 0.0


def lattice_spread(residual: np.ndarray) -> float:
    """Demosaicing trace in the image domain.

    Split pixels into the 4 positions of the 2x2 Bayer grid (even/odd row x
    even/odd column) and compare the residual energy of the 4 groups.
    Camera: the groups behave differently (some pixels were measured,
    others interpolated) -> large spread. Generator: no grid -> small spread.
    Returns (max - min) / mean of the 4 variances.
    """
    variances = np.array([np.var(residual[i::2, j::2]) for i in (0, 1) for j in (0, 1)])
    mean = variances.mean()
    return float((variances.max() - variances.min()) / mean) if mean > 0 else 0.0


def nyquist_peak_ratio(residual: np.ndarray) -> float:
    """Demosaicing trace in the Fourier domain.

    A pattern that repeats every 2 pixels creates peaks at the highest
    frequencies (the 'Nyquist' frequencies: corners and edge midpoints of the
    centred spectrum). Returns the energy at those points divided by the mean
    energy of the spectrum.
    """
    magnitude, _, _ = spectrum_components(fft2_centered(residual))
    power = magnitude ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    # With fftshift and even size, frequency 1/2 sits at index 0.
    nyquist = power[0, cx] + power[cy, 0] + power[0, 0]
    mean_power = power.mean()
    return float(nyquist / (3 * mean_power)) if mean_power > 0 else 0.0


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    [f"corr_raw_{CHANNELS[a]}{CHANNELS[b]}" for a, b in PAIRS]            # 3
    + [f"corr_res_{CHANNELS[a]}{CHANNELS[b]}" for a, b in PAIRS]          # 3
    + [f"res_std_{c}" for c in CHANNELS]                                  # 3
    + [f"res_kurt_{c}" for c in CHANNELS]                                 # 3
    + ["res_var_ratio_R/G", "res_var_ratio_B/G"]                          # 2
    + ["chroma_res_std_R-G", "chroma_res_std_B-G"]                        # 2
    + [f"lattice_spread_{c}" for c in CHANNELS]                           # 3
    + [f"nyquist_peak_{c}" for c in CHANNELS]                             # 3
)                                                                          # = 22


def extract_rgb_features(image: np.ndarray) -> np.ndarray:
    """Extract the 22 RGB/colour features from a 512x512 RGB image.

    Accepts uint8 in [0,255] (like npr.py) or float in [0,1].
    Returns a float64 vector of shape (22,), ordered as FEATURE_NAMES.
    """
    if image.shape != (512, 512, 3):
        raise ValueError("Input image must have shape (512, 512, 3).")
    rgb = normalize_rgb(image) if image.dtype == np.uint8 else np.asarray(image, np.float64)
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Float RGB values must be in [0, 1].")

    res = rgb_residuals(rgb)
    features = []

    # (a) correlation between channels on raw pixels: mostly scene content
    features += [correlation(rgb[..., a], rgb[..., b]) for a, b in PAIRS]

    # (b) same on residuals: scene removed, acquisition traces remain
    features += [correlation(res[..., a], res[..., b]) for a, b in PAIRS]

    # (c) residual statistics per channel
    stds = [float(np.std(res[..., c])) for c in range(3)]
    features += stds
    # kurtosis is undefined for a constant channel (e.g. fully saturated sky)
    features += [float(kurtosis(res[..., c].ravel(), fisher=True)) if stds[c] > 0 else 0.0
                 for c in range(3)]

    # (d) relative noise level of R and B compared with G
    eps = 1e-12
    features += [stds[0] ** 2 / (stds[1] ** 2 + eps), stds[2] ** 2 / (stds[1] ** 2 + eps)]

    # (e) chroma differences: high-frequency energy of R-G and B-G
    features += [float(np.std(high_pass_residual(rgb[..., 0] - rgb[..., 1]))),
                 float(np.std(high_pass_residual(rgb[..., 2] - rgb[..., 1])))]

    # (f) demosaicing traces (image domain, then Fourier domain)
    features += [lattice_spread(res[..., c]) for c in range(3)]
    features += [nyquist_peak_ratio(res[..., c]) for c in range(3)]

    return np.asarray(features, dtype=np.float64)


def extract_features_from_paths(image_paths) -> np.ndarray:
    """Stack features of many images: returns (N, 22)."""
    return np.stack([extract_rgb_features(load_crop(p)) for p in image_paths])
