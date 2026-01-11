# Utility functions

import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image(path, fallback_size=(100, 100)):
    img = cv2.imread(path, 0)
    if img is None:
        img = np.zeros(fallback_size)
        cv2.circle(img, (50, 50), 30, 1, -1)
    return img.astype(np.float64) / 255.0

def save_image(path, image):
    """
    Save an image to the specified path.
    """
    # Ensure the image is in the range [0, 255] and of type uint8
    img_to_save = np.clip(image * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_to_save)

def mse(img1, img2):
    """
    MSE between two images.
    """
    return np.mean((img1 - img2) ** 2)


def add_gaussian_noise(image, sigma, mean=0):
    """
    Add gaussian noise with a normal distribution of given standard deviation and mean.
    """
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gauss)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def estimate_noise(image, cutoff_ratio: float = 0.15):
    """
    Estimate noise of an image. Transform to frequency domain (FFT). Remove low frequencies. Inverse FFT. Compute sigma.
    """

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = image.astype(np.float32)

    # fft
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)

    # create mask
    height, width = img.shape
    center_y, center_x = height // 2, width // 2
    r = int(min(height, width) * cutoff_ratio)
    mask = np.ones((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), r, 0, -1)

    # apply mask
    F_shift_filtered = F_shift * mask

    # ifft to get back to time domain
    noise = np.fft.ifft2(np.fft.ifftshift(F_shift_filtered))
    noise = np.real(noise)

    # remove dc out of noise
    noise -= np.mean(noise)

    # compute standard deviation
    sigma = np.std(noise)

    return sigma, noise


def show_results(original, noisy, denoised):
    """Compares original, noisy and denoised images"""

    plt.figure(figsize=(12, 5))
    for k, (img, title) in enumerate(
        [
            (original, "Original"),
            (noisy, f"Noisy (MSE={mse(original, noisy):.4f})"),
            (denoised, f"Denoised (MSE={mse(original, denoised):.4f})"),
        ]
    ):
        plt.subplot(1, 3, k + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.show()
