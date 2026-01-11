# Montecarlo Non-Local Means

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view

from mcnlm.utils import load_image, show_results, add_gaussian_noise

# Params
@dataclass
class MCNLMParams:
    sigma: float
    h_factor: float = 0.4
    patch_size: int = 5
    search_radius: int = 10
    spatial_sigma: float = 10.0
    sampling_prob: float = 0.5

    @property
    def patch_radius(self):
        return self.patch_size // 2

# MCNLM Kernel 
def mcnlm_local(y_patch, window_patches, window_coords, params):
    n = len(window_patches)

    mask = np.random.rand(n) < params.sampling_prob
    if not np.any(mask):
        return y_patch[len(y_patch) // 2]

    patches = window_patches[mask]
    coords  = window_coords[mask]

    h = params.h_factor * params.sigma
    h2 = h * h
    sigma2 = params.sigma * params.sigma

    diffs = patches - y_patch            
    d2 = np.mean(diffs * diffs, axis=1)      
    d2 = np.maximum(d2 - 2.0 * sigma2, 0.0)
    w_r = np.exp(-d2 / h2)                

    spatial_d2 = np.sum(coords * coords, axis=1)
    w_s = np.exp(-spatial_d2 / (2.0 * params.spatial_sigma**2))
    
    w = w_r
    
    s = np.sum(w)
    if s == 0.0:
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


def test_mcnlm(image_path):
    image = load_image(image_path)
    
    SIGMA = 17
    noisy = add_gaussian_noise(image * 255, sigma=SIGMA).astype(np.float32) / 255.0
    
    params = MCNLMParams(
        sigma = SIGMA / 255.0,
        h_factor = 0.4,
        patch_size = 5,
        search_radius = 10,
        spatial_sigma = 10,
        sampling_prob = 1
    )
    
    denoised = mcnlm_denoise(noisy, params)

    show_results(image, noisy, denoised)

def show_matches(image_path, points, K=10000):
    """
    Show strong non-local matches for a list of points.
    """
    image = load_image(image_path)
    
    SIGMA = 17
    noisy = add_gaussian_noise(image * 255, sigma=SIGMA).astype(np.float32) / 255.0
    image = noisy

    params = MCNLMParams(
        sigma = SIGMA / 255.0,
        h_factor = 0.4,
        patch_size = 5,
        search_radius = 20,
        spatial_sigma = 10,
        sampling_prob = 0.3
    )
    
    pad = params.patch_radius
    rho = params.search_radius
    total_pad = pad + rho
    padded = np.pad(image, total_pad, mode="reflect")
    coords = window_coords(rho)

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")

    for pi, pj in points:
        pi0, pj0 = pi + total_pad, pj + total_pad

        y_patch = padded[
            pi0 - pad : pi0 + pad + 1,
            pj0 - pad : pj0 + pad + 1
        ].flatten()

        window = extract_search_window(padded, pi0, pj0, rho, pad)
        patches = window_patches(window, params.patch_size)

        # --- compute full NLM weights ---
        diffs = patches - y_patch
        d2 = np.mean(diffs * diffs, axis=1)
        d2 = np.maximum(d2 - 2 * params.sigma**2, 0)

        h2 = (params.h_factor * params.sigma * params.patch_size) ** 2
        w_r = np.exp(-d2 / h2)

        spatial_d2 = np.sum(coords * coords, axis=1)
        w_s = np.exp(-spatial_d2 / (2 * params.spatial_sigma**2))

        w = w_r * w_s

        # keep strongest matches
        idx = np.argsort(w)[-K:]
        ys = pi + coords[idx,0]
        xs = pj + coords[idx,1]

        plt.scatter(xs, ys, c=w[idx], s=10, cmap="hot")
        plt.scatter([pj], [pi], c="green", s=10)

    plt.title("Strong non-local matches for multiple points")
    plt.axis("off")
    plt.show()
