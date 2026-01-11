import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("mcnlm/imgs/clock.tiff", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Failed to load image")

# Resize to 256x256 for faster computation
image = cv2.resize(image, (256, 256))

SIGMA = 15
gauss = np.random.normal(0, SIGMA, image.shape).astype(np.float32)
noisy_image = cv2.add(image.astype(np.float32), gauss)
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# save image 
cv2.imwrite("mcnlm/imgs/noisy_clock.tiff", noisy_image)

noisy_image = noisy_image.astype(np.float32) / 255.0
image = image.astype(np.float32) / 255.0

CALCULATED_SIGMA = SIGMA / 255.0
h = 0.4 * CALCULATED_SIGMA  # This is taken from the paper linked below

# https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf


# Squared Euclidean distance between two patches
# d^2 = d^2(B(p, f), B(q, f)) of the (2f + 1) x (2f + 1) patches centered at pixels p and q
# Normally there is a 3x for color channels, but this is grayscale so we remove the factor
def euclidean_distance(p, q, f):
    # If patch goes out of bounds, minimize the patch size
    if (
        p[0] - f < 0
        or p[0] + f >= noisy_image.shape[0]
        or p[1] - f < 0
        or p[1] + f >= noisy_image.shape[1]
        or q[0] - f < 0
        or q[0] + f >= noisy_image.shape[0]
        or q[1] - f < 0
        or q[1] + f >= noisy_image.shape[1]
    ):
        f = min(
            p[0],
            noisy_image.shape[0] - 1 - p[0],
            p[1],
            noisy_image.shape[1] - 1 - p[1],
            q[0],
            noisy_image.shape[0] - 1 - q[0],
            q[1],
            noisy_image.shape[1] - 1 - q[1],
        )

    patch_p = noisy_image[p[0] - f : p[0] + f + 1, p[1] - f : p[1] + f + 1]
    patch_q = noisy_image[q[0] - f : q[0] + f + 1, q[1] - f : q[1] + f + 1]
    distance = np.sum((patch_p - patch_q) ** 2)

    COLOR_CHANNELS = 1  # Grayscale image
    distance /= COLOR_CHANNELS * (
        (2 * f + 1) ** 2
    )  # Normalize by patch size and color channels
    return distance


# Weight function
def w(p, q):
    f = 1  # Patch radius
    d2 = euclidean_distance(p, q, f)
    weight = np.exp(-max(d2 - 2 * CALCULATED_SIGMA**2, 0.0) / (h**2))
    return weight


# Normalizing factor
# C(p, r) = Σ w(p, q) for all q in B(p, r)
# B(p, r) is the (2r + 1) x (2r + 1) search window centered at pixel p
# This research zone is limited to a square neighborhood of fixed size because of computation
# restrictions. This is a 21 x 21 window for small and moderate values of σ. The size of the research
# window is increased to 35 x 35 for large values of σ due to the necessity of finding more similar pixels
# to reduce further the noise
def C(p, r):
    normalizing_factor = 0.0
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            q = (p[0] + i, p[1] + j)
            # Make sure q is within image boundaries
            if 0 <= q[0] < noisy_image.shape[0] and 0 <= q[1] < noisy_image.shape[1]:
                normalizing_factor += w(p, q)
    return normalizing_factor


# Non-Local Means Denoising
def nlm_denoising(noisy_image):
    denoised_image = np.zeros_like(noisy_image)
    r = 10  # 21x21 search window

    for i in range(noisy_image.shape[0]):
        print(f"Processing row {i + 1}/{noisy_image.shape[0]}", end="\r")
        for j in range(noisy_image.shape[1]):
            p = (i, j)
            C_p_r = C(p, r)
            pixel_value = 0.0

            for m in range(-r, r + 1):
                for n in range(-r, r + 1):
                    q = (i + m, j + n)
                    # Make sure q is within image boundaries
                    if (
                        0 <= q[0] < noisy_image.shape[0]
                        and 0 <= q[1] < noisy_image.shape[1]
                    ):
                        pixel_value += w(p, q) * noisy_image[q[0], q[1]]

            denoised_image[i, j] = pixel_value / C_p_r

    return denoised_image


def test_naive_nlm():
    denoised_image = nlm_denoising(noisy_image)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Noisy Image")
    plt.imshow(noisy_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Denoised Image (NLM)")
    plt.imshow(denoised_image, cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    test_naive_nlm()
