import numpy as np

def hann_window_2d(h, w):
    # Fenêtre 2D séparable : produit externe de deux fenêtres de Hann 1D
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