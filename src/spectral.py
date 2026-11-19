import numpy as np

def compute_fft(image):
    # Step 1: Convert RGB image to grayscale
    image = image.astype(np.float32)

    gray = (
        0.299 * image[:, :, 0] +
        0.587 * image[:, :, 1] +
        0.114 * image[:, :, 2]
    )

    # Step 2: Apply the 2D Fourier transform
    fft = np.fft.fft2(gray)

    return fft

def compute_power_spectrum(image):
    fft = compute_fft(image)
    return np.abs(fft) ** 2