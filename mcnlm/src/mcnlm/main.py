import numpy as np
import cv2
import matplotlib.pyplot as plt

from mcnlm.noise_estimator import NoiseEstimator

def add_gaussian_noise(image, sigma, mean=0):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gauss)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def test_noise_extraction():
    """
    Estimate noise with FFT usage example on a grayscale image
    """
    image_path = "imgs/man.tiff"
    
    # Load image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Add noise
    noisy_img = add_gaussian_noise(image, 20)
    
    # Noise estimation
    ne = NoiseEstimator(image=noisy_img, cutoff_ratio=0.18)
    sigma, extracted_noise = ne.estimate_noise()
    
    print(f"Estimated noise std: {sigma:.3f}")
    
    # Plot original noisy image and extracted noise
    plt.figure(figsize=(15, 5))  # wider for 3 images

    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
        
    plt.subplot(1, 4, 2)
    plt.imshow(noisy_img, cmap="gray")
    plt.title("Noisy Image")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(extracted_noise, cmap="gray")
    plt.title(f"Extracted Noise sigma = {sigma:.3f}")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(noisy_img - extracted_noise, cmap="gray")
    plt.title(f"Difference?")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def main():
    import os
    print(os.getcwd())
    print("MCNLM")
    test_noise_extraction()