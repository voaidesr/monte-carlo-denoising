# Estimate noise with fft
import numpy as np
import cv2

class NoiseEstimator:
    def __init__(self, image, cutoff_ratio: float = 0.15) -> None:
        """
        image: Image to estimate noise out of.

        cutoff_ratio: Cutoff ratio for filtering out low frequencies.
        """
        self.image = image
        self.cutoff_ratio = cutoff_ratio

    def estimate_noise(self):
        """
        Transform to frequency domain. Remove low frequencies. Inverse FFT. Compute sigma.
        """

        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        img = self.image.astype(np.float32)

        # fft
        F = np.fft.fft2(img)
        F_shift = np.fft.fftshift(F)

        # create mask
        height, width = img.shape
        center_y, center_x = height // 2, width // 2
        r = int(min(height, width) * self.cutoff_ratio)
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
