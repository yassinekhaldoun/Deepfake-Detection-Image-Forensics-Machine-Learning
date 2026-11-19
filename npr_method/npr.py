import numpy as np
from scipy.stats import skew, kurtosis


def extract_npr_features(image: np.ndarray) -> np.ndarray:
    """
    Extract Neighboring Pixel Relationships (NPR) features from an RGB image.

    For each RGB channel:
        - Compute horizontal, vertical and diagonal pixel differences
          over non-overlapping 2x2 blocks.
        - Extract mean, variance, skewness and kurtosis.
        - Compute a 16-bin normalized histogram.

    Returns:
        A feature vector of shape (180,).
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3) in RGB format.")

    # Convert to float to avoid unsigned integer subtraction issues.
    image = image.astype(np.float64)

    # Keep only complete 2x2 blocks.
    height = (image.shape[0] // 2) * 2
    width = (image.shape[1] // 2) * 2
    image = image[:height, :width, :]

    # Extract the four pixels of each 2x2 block.
    A1 = image[0::2, 0::2, :]
    A2 = image[0::2, 1::2, :]
    A3 = image[1::2, 0::2, :]
    A4 = image[1::2, 1::2, :]

    features = []

    # Process each RGB channel independently.
    for channel in range(3):

        # Compute the three difference distributions.
        differences = [
            (A2[:, :, channel] - A1[:, :, channel]).ravel(),
            (A3[:, :, channel] - A1[:, :, channel]).ravel(),
            (A4[:, :, channel] - A1[:, :, channel]).ravel()
        ]

        for diff in differences:

            # Statistical features.
            mean = np.mean(diff)
            variance = np.var(diff)
            skewness = skew(diff) if variance > 0 else 0.0
            kurt = kurtosis(diff, fisher=True) if variance > 0 else 0.0

            features.extend([mean, variance, skewness, kurt])

            # Normalized histogram.
            hist, _ = np.histogram(
                diff,
                bins=16,
                range=(-255, 255),
                density=False
            )

            hist = hist / hist.sum()
            features.extend(hist)

    return np.asarray(features, dtype=np.float64)