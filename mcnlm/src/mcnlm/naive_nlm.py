import numpy as np
from numba import njit, prange
from dataclasses import dataclass

@dataclass
class NLMParams:
    sigma: float          # Standard deviation of the noise
    patch_radius: int    # Radius of the patches (f)
    search_radius: int   # Radius of the search window (r)
    h_factor: float      # Filtering parameter factor

# https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf


# Squared Euclidean distance between two patches
# d^2 = d^2(B(p, f), B(q, f)) of the (2f + 1) x (2f + 1) patches centered at pixels p and q
# Normally there is a 3x for color channels, but this is grayscale so we remove the factor
@njit
def euclidean_distance(noisy_image, p, q, f):
    height, width = noisy_image.shape

    # If patch goes out of bounds, minimize the patch size
    f_adj = min(f,
                p[0],
                height - 1 - p[0],
                p[1],
                width - 1 - p[1],
                q[0],
                height - 1 - q[0],
                q[1],
                width - 1 - q[1])

    patch_p = noisy_image[p[0] - f_adj : p[0] + f_adj + 1, p[1] - f_adj : p[1] + f_adj + 1]
    patch_q = noisy_image[q[0] - f_adj : q[0] + f_adj + 1, q[1] - f_adj : q[1] + f_adj + 1]
    distance = np.sum((patch_p - patch_q) ** 2)

    COLOR_CHANNELS = 1  # Grayscale image
    distance /= COLOR_CHANNELS * (
        (2 * f_adj + 1) ** 2
    )  # Normalize by patch size and color channels
    return distance


# Weight function
@njit
def w(noisy_image, p, q, f, sigma, h):
    d2 = euclidean_distance(noisy_image, p, q, f)
    weight = np.exp(-max(d2 - 2 * sigma**2, 0.0) / (h**2))
    return weight


# Normalizing factor
# C(p, r) = Σ w(p, q) for all q in B(p, r)
# B(p, r) is the (2r + 1) x (2r + 1) search window centered at pixel p
# This research zone is limited to a square neighborhood of fixed size because of computation
# restrictions. This is a 21 x 21 window for small and moderate values of σ. The size of the research
# window is increased to 35 x 35 for large values of σ due to the necessity of finding more similar pixels
# to reduce further the noise
@njit
def C(noisy_image, p, r, f, sigma, h):
    normalizing_factor = 0.0
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            q = (p[0] + i, p[1] + j)
            # Make sure q is within image boundaries
            if 0 <= q[0] < noisy_image.shape[0] and 0 <= q[1] < noisy_image.shape[1]:
                normalizing_factor += w(noisy_image, p, q, f, sigma, h)
    return normalizing_factor


# Non-Local Means Denoising
@njit(parallel=True)
def nlm_denoising(noisy_image, r, f, sigma, h):
    denoised_image = np.zeros_like(noisy_image)

    for i in prange(noisy_image.shape[0]):
        for j in range(noisy_image.shape[1]):
            p = (i, j)
            C_p_r = C(noisy_image, p, r, f, sigma, h)
            pixel_value = 0.0

            for m in range(-r, r + 1):
                for n in range(-r, r + 1):
                    q = (i + m, j + n)
                    # Make sure q is within image boundaries
                    if (
                        0 <= q[0] < noisy_image.shape[0]
                        and 0 <= q[1] < noisy_image.shape[1]
                    ):
                        pixel_value += w(noisy_image, p, q, f, sigma, h) * noisy_image[q[0], q[1]]

            denoised_image[i, j] = pixel_value / C_p_r

    return denoised_image


def test_naive_nlm(noisy_image, params: NLMParams):
    padded = np.pad(noisy_image, params.search_radius + params.patch_radius, mode='reflect')
    denoised = nlm_denoising(padded, params.search_radius, params.patch_radius, params.sigma, params.h_factor * params.sigma)

    # Remove padding
    total_pad = params.search_radius + params.patch_radius
    denoised = denoised[total_pad:-total_pad, total_pad:-total_pad]
    return denoised