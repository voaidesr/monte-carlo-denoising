# Utility functions

import numpy as np
import matplotlib.pyplot as plt
import cv2

import mcnlm.mc_nlm as mc_nlm

# ---------- General utilites ------------

def load_image(path, fallback_size=(100, 100)):
    """
    Load an image from the specified path.
    """
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
    MSE (Mean square error) between two images.
    """
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2, max_pixel=255.0):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two images.
    """
    error = mse(img1, img2)
    if error == 0:
        return float('inf')  # identical images
    return 10 * np.log10((max_pixel ** 2) / error)


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
    """Compares original, noisy and denoised image"""

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


# ------------ Naive NLM Utilities ------------



# ------------ MCNLM utilites ---------------
def show_mcnlm_result_zoomed(image_path, probs, zoom, output_path):
    image = load_image(image_path)

    sigma = 17.0
    noisy = add_gaussian_noise(image*255, sigma) / 255.0

    x0, y0, w, h = zoom
    x1, y1 = x0 + w, y0 + h

    n_pi = len(probs)
    n_cols = 1 + n_pi
    fig, axs = plt.subplots(2, n_cols, figsize=(3.2*n_cols, 6))

    # Noisy
    axs[0, 0].imshow(noisy, cmap="gray")
    axs[0, 0].set_title(f"Noisy MSE = {mse(image, noisy):.4f}")
    axs[0, 0].axis("off")

    axs[1, 0].imshow(noisy[y0:y1, x0:x1], cmap="gray")
    axs[1, 0].set_title("Zoom")
    axs[1, 0].axis("off")

    # Draw zoom boxes on original and noisy
    for col in [0, 1]:
        rect = plt.Rectangle((x0, y0), w, h,
                             edgecolor="red",
                             facecolor="none",
                             linewidth=1,
                             alpha = 0.5)
        axs[0, col].add_patch(rect)
        
    for i, pi in enumerate(probs):
        col = 1 + i
        params = mc_nlm.MCNLMParams(
            sigma=sigma/255.0,
            h_factor=0.4,
            patch_size=5,
            search_radius=10,
            spatial_sigma=10,
            sampling_prob=pi
        )

        denoised = mc_nlm.test_mcnlm(noisy, params)

        # Full
        axs[0, col].imshow(denoised, cmap="gray")
        axs[0, col].set_title(f"MCNLM Denoised MSE = {mse(image, denoised):.4f}\np = {pi}")
        axs[0, col].axis("off")

        # Zoom
        zoom_d = denoised[y0:y1, x0:x1]
        axs[1, col].imshow(zoom_d, cmap="gray")
        axs[1, col].axis("off")
        axs[1, col].set_title("Zoom")

        # Draw zoom box on denoised
        rect = plt.Rectangle((x0, y0), w, h,
                             edgecolor="red",
                             facecolor="none",
                             linewidth=1,
                             alpha = 0.5)
        axs[0, col].add_patch(rect)

    
    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.85,
        bottom=0.05,
        wspace=0.15,   # horizontal gap between columns
        hspace=0.02    # vertical gap between rows
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    
    
def show_matches(image_path, points, K=3000):
    """
    Visualize strongest Monte-Carlo NLM matches using the numba MC-NLM kernel logic.
    """
    # --- load + noise ---
    image = load_image(image_path)
    sigma_val = 17.0
    noisy = add_gaussian_noise(image*255, sigma_val) / 255.0

    # --- MCNLM params ---
    params = mc_nlm.MCNLMParams(
        sigma = sigma_val / 255.0,
        h_factor = 0.4,
        patch_size = 5,
        search_radius = 20,
        sampling_prob = 0.3
    )

    pad = params.patch_radius
    rho = params.search_radius
    h = params.h_factor * params.sigma
    h2 = h*h
    sigma2 = params.sigma*params.sigma

    total_pad = pad + rho
    padded = np.pad(noisy, total_pad, mode="reflect")

    patch_size = params.patch_size
    patch_len = patch_size * patch_size
    center_idx = patch_len // 2

    # --- offsets for the search window ---
    offsets = []
    for di in range(-rho, rho+1):
        for dj in range(-rho, rho+1):
            offsets.append((di,dj))
    offsets = np.array(offsets)

    plt.figure(figsize=(8,8))
    plt.imshow(noisy, cmap="gray")
    
    for pi, pj in points:
        pi0, pj0 = pi + total_pad, pj + total_pad

        # center patch
        y_patch = padded[pi0-pad:pi0+pad+1, pj0-pad:pj0+pad+1].flatten()

        weights = []
        coords = []

        # Monte-Carlo search
        for di, dj in offsets:
            if np.random.rand() >= params.sampling_prob:
                continue

            qi, qj = pi0 + di, pj0 + dj
            comp_patch = padded[qi-pad:qi+pad+1, qj-pad:qj+pad+1].flatten()

            # --- exact MCNLM weight ---
            diff = comp_patch - y_patch
            d2 = np.mean(diff*diff)
            d2 = max(d2 - 2*sigma2, 0.0)  # exact as numba kernel
            w = np.exp(-d2 / h2)

            weights.append(w)
            coords.append((di,dj))

        if len(weights) == 0:
            continue

        weights = np.array(weights)
        coords = np.array(coords)

        # strongest matches
        idx = np.argsort(weights)[-min(K, len(weights)):]
        xs = pj + coords[idx,1]
        ys = pi + coords[idx,0]
        cs = weights[idx]

        # normalize for colormap
        cs = cs / cs.max()

        plt.scatter(xs, ys, c=cs, cmap="hot", s=12)

        # original point
        plt.scatter([pj], [pi], c="lime", s=30)

    plt.title("MC-NLM strong matches (numba kernel)")
    plt.axis("off")
    plt.show()