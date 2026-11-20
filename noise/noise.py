"""Variance features for 512x512 RGB image crops."""

import numpy as np


NOISE_FEATURE_NAMES = [
    "noise_variance_R",
    "noise_variance_G",
    "noise_variance_B",
]


def variance_per_channel(image):
    """Return raw pixel-intensity variance for R, G, and B.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image crop with shape (512, 512, 3), typically uint8.

    Returns
    -------
    numpy.ndarray
        Three float64 values in RGB order.
    """
    image = np.asarray(image)

    if image.shape != (512, 512, 3):
        raise ValueError("Input image must have shape (512, 512, 3).")

    image = image.astype(np.float64)

    if not np.isfinite(image).all():
        raise ValueError("Image contains non-finite pixel values.")

    return np.var(image, axis=(0, 1), dtype=np.float64)


def mean_channel_variance(image):
    """Return the average of the red, green, and blue variances."""
    return float(np.mean(variance_per_channel(image)))


def extract_noise_features(image):
    """Feature-extractor interface: return one variance feature per RGB channel."""
    return variance_per_channel(image)
