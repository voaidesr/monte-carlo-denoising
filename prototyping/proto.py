import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.lib.stride_tricks import sliding_window_view

def get_mse(img1, img2):
    return np.mean((img1 - img2)**2)

def mcnlm_local(noisy_patch: np.ndarray,
                window_patches: np.ndarray,
                window_coords: np.ndarray,
                probs: float,
                h_r: float,
                h_s: float) -> float:
    n = len(window_patches)

    i_vect = np.random.random(n) < probs
    selected_indices = np.where(i_vect)[0]

    if len(selected_indices) == 0:
        return noisy_patch[len(noisy_patch) // 2]

    selected_patches = window_patches[selected_indices]
    selected_coords = window_coords[selected_indices]

    diffs = selected_patches - noisy_patch
    dists_sq = np.sum(diffs**2, axis=1)
    w_r = np.exp(-dists_sq / (h_r**2))

    spatial_dist_sq = np.sum(selected_coords**2, axis=1)
    w_s = np.exp(-spatial_dist_sq / (2 * h_s**2))

    weights = w_r * w_s

    sum_B = np.sum(weights)
    if sum_B == 0:
        return noisy_patch[len(noisy_patch) // 2]

    center_idx = len(noisy_patch) // 2
    center_pixels = selected_patches[:, center_idx]
    sum_A = np.sum(weights * center_pixels)

    return sum_A / sum_B

def trial1():
    try:
        img = cv2.imread('../imgs/moon.tiff', 0)
        if img is None: raise FileNotFoundError
    except:
        img = np.zeros((100, 100))
        cv2.circle(img, (50, 50), 30, 1, -1)

    img = img.astype(np.float64) / 255.0
    sd = 17.0 / 255.0
    noisy_img = img + np.random.normal(0, sd, size=img.shape)

    h, w = noisy_img.shape
    denoised_img = np.zeros_like(noisy_img)

    patch_size = 5
    pad = patch_size // 2

    h_r = 0.4 * 5 * sd
    h_s = 10.0
    rho = 15
    xi = 0.5

    total_pad = pad + rho
    padded_img = np.pad(noisy_img, total_pad, mode='reflect')

    print(f"Denoising with Search Window {2*rho+1}x{2*rho+1}...")

    for i in range(h):
        for j in range(w):
            pi, pj = i + total_pad, j + total_pad
            y_patch = padded_img[pi-pad : pi+pad+1, pj-pad : pj+pad+1].flatten()

            r_min, r_max = pi - rho, pi + rho + 1
            c_min, c_max = pj - rho, pj + rho + 1

            window_slice = padded_img[r_min-pad : r_max+pad, c_min-pad : c_max+pad]

            patches_view = sliding_window_view(window_slice, (patch_size, patch_size))
            window_patches = patches_view.reshape(-1, patch_size * patch_size)

            wy, wx = np.mgrid[-rho:rho+1, -rho:rho+1]
            window_coords = np.stack((wy.flatten(), wx.flatten()), axis=1)

            pixel_val = mcnlm_local(y_patch, window_patches, window_coords, xi, h_r, h_s)
            denoised_img[i, j] = pixel_val

    mse_noisy = get_mse(img, noisy_img)
    mse_denoised = get_mse(img, denoised_img)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title("Original")
    plt.subplot(1, 3, 2); plt.imshow(noisy_img, cmap='gray'); plt.title(f"Noisy (MSE: {mse_noisy:.4f})")
    plt.subplot(1, 3, 3); plt.imshow(denoised_img, cmap='gray'); plt.title(f"MCNLM (MSE: {mse_denoised:.4f})")
    plt.show()

if __name__ == "__main__":
    trial1()