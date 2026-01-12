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


def show_matches(image_path, points, output_path, K=3000):
    """
    Visualize MC-NLM matches using Monte-Carlo sampling.
    Only a subset of offsets is used according to sampling_prob.
    """
    # --- load + noisy ---
    image = load_image(image_path)
    SIGMA = 17
    noisy = add_gaussian_noise(image*255, sigma=SIGMA).astype(np.float32)/255.0

    params = mc_nlm.MCNLMParams(
        sigma=SIGMA/255.0,
        h_factor=0.4,
        patch_size=5,
        search_radius=20,
        spatial_sigma=10,
        sampling_prob=1.0 # Monte-Carlo sampling
    )

    pad = params.patch_radius
    rho = params.search_radius
    total_pad = pad + rho
    padded = np.pad(noisy, total_pad, mode='reflect')

    # --- offsets for search window ---
    y, x = np.mgrid[-rho:rho+1, -rho:rho+1]
    coords = np.stack([y.flatten(), x.flatten()], axis=1)

    plt.figure(figsize=(8,8))
    plt.imshow(noisy, cmap='gray')

    for pi, pj in points:
        pi0, pj0 = pi + total_pad, pj + total_pad
        y_patch = padded[pi0-pad:pi0+pad+1, pj0-pad:pj0+pad+1].flatten()

        # --- Monte-Carlo sampling mask ---
        mask = np.random.rand(len(coords)) < params.sampling_prob
        sampled_coords = coords[mask]
        if len(sampled_coords) == 0:
            continue

        # --- compute weights only for sampled offsets ---
        w_r = []
        for di, dj in sampled_coords:
            qi, qj = pi0+di, pj0+dj
            comp_patch = padded[qi-pad:qi+pad+1, qj-pad:qj+pad+1].flatten()
            diff = comp_patch - y_patch
            d2 = np.mean(diff*diff)
            d2 = max(d2 - 2*params.sigma**2, 0.0)
            w_r.append(np.exp(-d2 / (params.h_factor*params.sigma*params.patch_size)**2))
        w_r = np.array(w_r)

        # --- spatial term ---
        spatial_d2 = np.sum(sampled_coords**2, axis=1)
        w_s = np.exp(-spatial_d2 / (2*params.spatial_sigma**2))

        w = w_r * w_s

        # --- strongest matches ---
        idx = np.argsort(w)[-min(K,len(w)):]
        xs = pj + sampled_coords[idx,1]
        ys = pi + sampled_coords[idx,0]
        cs = w[idx]

        # --- scale for visualization ---
        cs = np.clip(cs, 1e-3, None)
        cs = cs**0.5
        cs = cs / cs.max()

        plt.scatter(xs, ys, c=cs, cmap='hot', s=10, edgecolors='none')
        plt.scatter([pj], [pi], c='lime', s=20)

    plt.title(f"Strong matches (Monte-Carlo sampling prob = {params.sampling_prob})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path)



