# Montecarlo Non-Local Means - Parallelized

import numpy as np
from dataclasses import dataclass
from numba import njit, prange

from mcnlm.utils import load_image, add_gaussian_noise

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

@njit
def mcnlm_local_numba(noisy, pi, pj, patch_radius, search_radius, sigma, h, sampling_prob):
    """Numba-compatible MCNLM kernel"""
    h2 = h * h
    sigma2 = sigma * sigma
    patch_size = 2 * patch_radius + 1
    patch_len = patch_size * patch_size
    center_idx = patch_len // 2
    
    # Extract center patch
    y_patch = noisy[pi - patch_radius : pi + patch_radius + 1, 
                    pj - patch_radius : pj + patch_radius + 1].copy().flatten()
    
    total_weight = 0.0
    weighted_sum = 0.0
    any_sampled = False
    
    # Iterate over search window
    for di in range(-search_radius, search_radius + 1):
        for dj in range(-search_radius, search_radius + 1):
            # Monte Carlo sampling
            if np.random.random() >= sampling_prob:
                continue
            
            any_sampled = True
            qi, qj = pi + di, pj + dj
            
            # Extract comparison patch
            comp_patch = noisy[qi - patch_radius : qi + patch_radius + 1,
                              qj - patch_radius : qj + patch_radius + 1].copy().flatten()
            
            # Compute squared distance
            d2 = 0.0
            for k in range(patch_len):
                diff = comp_patch[k] - y_patch[k]
                d2 += diff * diff
            d2 /= patch_len
            d2 = max(d2 - 2.0 * sigma2, 0.0)
            
            # Compute weight
            w = np.exp(-d2 / h2)
            total_weight += w
            weighted_sum += w * comp_patch[center_idx]
    
    if not any_sampled or total_weight == 0.0:
        return y_patch[center_idx]
    
    return weighted_sum / total_weight


@njit(parallel=True)
def mcnlm_denoise(noisy, search_radius, patch_size, sigma, h, sampling_prob):
    """Parallelized MCNLM denoising"""
    out = np.zeros_like(noisy)
    
    patch_radius = patch_size // 2
    total_pad = search_radius + patch_radius
    
    height = noisy.shape[0] - 2 * total_pad
    width = noisy.shape[1] - 2 * total_pad
    
    for i in prange(height):
        for j in range(width):
            pi, pj = i + total_pad, j + total_pad
            out[pi, pj] = mcnlm_local_numba(
                noisy, pi, pj, patch_radius, search_radius,
                sigma, h, sampling_prob
            )
    
    return out


def test_mcnlm(noisy, params: MCNLMParams):
    total_pad = params.search_radius + params.patch_radius
    padded = np.pad(noisy, total_pad, mode="reflect")
    denoised = mcnlm_denoise(
        padded, 
        params.search_radius, 
        params.patch_size, 
        params.sigma, 
        params.h_factor * params.sigma, 
        params.sampling_prob
    )
    return denoised[total_pad:-total_pad, total_pad:-total_pad]
