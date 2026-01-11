# Montecarlo Non-Local Means

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view

from mcnlm.utils import load_image, show_results


# Params
@dataclass
class MCNLMParams:
    sigma: float
    patch_size: int = 5
    search_radius: int = 15
    spatial_sigma: float = 10.0
    sampling_prob: float = 0.5
    h_factor: float = 0.4

    @property
    def patch_radius(self):
        return self.patch_size // 2

    @property
    def h_r(self):
        return self.h_factor * self.patch_size * self.sigma


# MCNLM Kernel
def mcnlm_local(y_patch, window_patches, window_coords, params):
    n = len(window_patches)

    mask = np.random.rand(n) < params.sampling_prob
    if not np.any(mask):
        return y_patch[len(y_patch) // 2]

    patches = window_patches[mask]
    coords = window_coords[mask]

    diffs = patches - y_patch
    d2 = np.sum(diffs**2, axis=1)
    w_r = np.exp(-d2 / (params.h_r**2))

    #spatial_d2 = np.sum(coords**2, axis=1)
    #w_s = np.exp(-spatial_d2 / (2 * params.spatial_sigma**2))
    w_s = 1
    
    w = w_r * w_s
    s = np.sum(w)

    if s == 0:
        return y_patch[len(y_patch) // 2]

    center = patches[:, len(y_patch) // 2]
    return np.sum(w * center) / s


# Sliding window utils
def extract_search_window(padded, pi, pj, rho, pad):
    r0, r1 = pi - rho - pad, pi + rho + pad + 1
    c0, c1 = pj - rho - pad, pj + rho + pad + 1
    return padded[r0:r1, c0:c1]


def window_patches(window, patch_size):
    patches = sliding_window_view(window, (patch_size, patch_size))
    return patches.reshape(-1, patch_size * patch_size)


def window_coords(rho):
    y, x = np.mgrid[-rho : rho + 1, -rho : rho + 1]
    return np.stack((y.flatten(), x.flatten()), axis=1)


# MCNLM
def mcnlm_denoise(noisy, params):
    h, w = noisy.shape
    out = np.zeros_like(noisy)

    pad = params.patch_radius
    rho = params.search_radius
    total_pad = pad + rho

    padded = np.pad(noisy, total_pad, mode="reflect")
    coords = window_coords(rho)

    print(f"MC-NLM with search window {2 * rho + 1}×{2 * rho + 1}")

    for i in range(h):
        print(f"Row {i + 1}/{h}", end="\r")
        for j in range(w):
            pi, pj = i + total_pad, j + total_pad

            y_patch = padded[pi - pad : pi + pad + 1, pj - pad : pj + pad + 1].flatten()

            window = extract_search_window(padded, pi, pj, rho, pad)
            patches = window_patches(window, params.patch_size)

            out[i, j] = mcnlm_local(y_patch, patches, coords, params)

    print()
    return out


def test_mcnlm():
    img = load_image("imgs/clock.tiff")
    sigma = 17.0 / 255.0
    noisy = img + np.random.normal(0, sigma, img.shape)

    params = MCNLMParams(sigma=sigma)
    params.sampling_prob = 0.3
    params.h_factor = 0.8

    denoised = mcnlm_denoise(noisy, params)

    show_results(img, noisy, denoised)
