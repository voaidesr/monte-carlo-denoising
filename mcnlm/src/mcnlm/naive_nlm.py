import numpy as np
from dataclasses import dataclass
from numba import njit, prange

from mcnlm.utils import load_image, show_results, add_gaussian_noise

# Params
@dataclass
class NLMParams:
    sigma: float
    h_factor: float = 0.4
    patch_radius: int = 2
    search_radius: int = 10

    @property
    def h(self):
        return self.h_factor * self.sigma * (self.patch_radius*2 + 1)

    @property
    def patch_size(self):
        return 2 * self.patch_radius + 1


@njit(parallel=True)
def nlm_denoise_fast(padded, H, W, pad, patch_radius, search_radius, h, sigma2):
    """
    Fast Non-Local Means denoising using Numba JIT compilation and parallelization.
    """
    # Output denoised image
    out = np.zeros((H, W), dtype=np.float32)

    # patch_size = 2 * patch_radius + 1  # Patch size
    r = search_radius
    h2 = h * h

    # Loop over all pixels in the original image
    for i in prange(H):
        for j in range(W):
            # Map pixel coordinates to padded image coordinates
            pi, pj = i + pad, j + pad

            # Extract center patch
            patch_i = padded[
                pi - patch_radius : pi + patch_radius + 1,
                pj - patch_radius : pj + patch_radius + 1,
            ].flatten()

            acc = 0.0
            norm = 0.0

            # Loop over search window
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    qi, qj = pi + di, pj + dj

                    # Extract neighbor patch
                    patch_q = padded[
                        qi - patch_radius : qi + patch_radius + 1,
                        qj - patch_radius : qj + patch_radius + 1,
                    ].flatten()

                    # Compute squared Euclidean distance between patches
                    d2 = np.mean((patch_i - patch_q) ** 2)

                    # Compute Non-Local Means weight
                    w = np.exp(-max(d2 - 2 * sigma2, 0.0) / h2)

                    # Accumulate weighted pixel values
                    acc += w * padded[qi, qj]
                    norm += w  # Accumulate weights for normalization

            # Assign normalized value to output pixel
            out[i, j] = acc / norm

    return out


def nlm_denoise(img, params):
    H, W = img.shape
    # pad the image
    pad = params.patch_radius + params.search_radius
    padded = np.pad(img, pad, mode="reflect")
    return nlm_denoise_fast(
        padded,
        H,
        W,
        pad,
        params.patch_radius,
        params.search_radius,
        params.h,
        params.sigma**2,
    )


# Test naive nlm
def test_naive_nlm():
    image = load_image("imgs/clock.tiff")
    SIGMA = 15
    noisy = add_gaussian_noise(image * 255, sigma=SIGMA).astype(np.float32) / 255.0
    params = NLMParams(sigma=SIGMA / 255, patch_radius=2, search_radius=10)
    denoised = nlm_denoise(noisy, params)
    show_results(image, noisy, denoised)
