import numpy as np
from scipy.stats import skew, kurtosis

def compute_npr_differences(image):
    height = (image.shape[0] // 2) * 2
    width = (image.shape[1] // 2) * 2

    image = image[:height, :width, :].astype(np.float64)

    A1 = image[0::2, 0::2, :]
    A2 = image[0::2, 1::2, :]
    A3 = image[1::2, 0::2, :]
    A4 = image[1::2, 1::2, :]

    differences = {}

    channels = ["R", "G", "B"]

    for c, channel in enumerate(channels):
        differences[channel] = {
            "horizontal": (A2[:, :, c] - A1[:, :, c]).ravel(),
            "vertical": (A3[:, :, c] - A1[:, :, c]).ravel(),
            "diagonal": (A4[:, :, c] - A1[:, :, c]).ravel()
        }

    return differences

def collect_differences(image_paths):
    collected = {
        channel: {
            direction: []
            for direction in ["horizontal", "vertical", "diagonal"]
        }
        for channel in ["R", "G", "B"]
    }

    for path in image_paths:
        image = load_image(path)
        differences = compute_npr_differences(image)

        for channel in ["R", "G", "B"]:
            for direction in ["horizontal", "vertical", "diagonal"]:
                collected[channel][direction].append(
                    differences[channel][direction]
                )

    # Concatenate all image vectors
    for channel in collected:
        for direction in collected[channel]:
            collected[channel][direction] = np.concatenate(
                collected[channel][direction]
            )

    return collected

def extract_npr_features(image: np.ndarray) -> np.ndarray:
    """
    Extract Neighboring Pixel Relationships (NPR) features from an RGB image.

    For each RGB channel and direction:
        - Compute mean, variance, skewness and kurtosis.
        - Compute a 16-bin normalized histogram.

    Returns:
        A feature vector of shape (180,).
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3) in RGB format.")

    differences = compute_npr_differences(image)

    features = []

    for channel in ["R", "G", "B"]:
        for direction in ["horizontal", "vertical", "diagonal"]:

            diff = differences[channel][direction]

            # Statistical features.
            mean = np.mean(diff)
            variance = np.var(diff)
            skewness = skew(diff) if variance > 0 else 0.0
            kurt = kurtosis(diff, fisher=True) if variance > 0 else 0.0

            features.extend([mean, variance, skewness, kurt])

            # Normalized histogram.
            hist, _ = np.histogram(diff, bins=16, density=False)
            hist = hist / hist.sum()

            features.extend(hist)

    return np.asarray(features, dtype=np.float64)