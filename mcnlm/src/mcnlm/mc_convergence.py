import numpy as np
import matplotlib.pyplot as plt

from mcnlm.utils import mse, add_gaussian_noise, load_image, psnr, show_results
import mcnlm.mc_nlm as mc_nlm
import mcnlm.naive_nlm as naive_nlm

def run_mc_convergence(image_path, xis):
    """
    Run MCNLM for different sampling probabilities and compute MSE between MC method and naive NLM.
    """
    image = load_image(image_path)
    sigma = 17
    noisy = add_gaussian_noise(image * 255, sigma=sigma).astype(np.float32) / 255.0

    nlm_params = naive_nlm.NLMParams(
        sigma =  sigma / 255.0,
        h_factor = 0.4,
        patch_radius = 2,
        search_radius = 10
    )

    # compute naive NLM
    print("Computing naive NLM...")
    nlm_ref = naive_nlm.test_naive_nlm(noisy, nlm_params) * 255.0
    naive_clean_mse = mse(nlm_ref, image)
    native_clean_psnr = psnr(nlm_ref, image)

    print(f"Naive NLM MSE vs clean: {naive_clean_mse}, PSNR vs clean: {native_clean_psnr}")

    mc_clean_mse = []
    mc_clean_psnr = []
    print("Running Monte-Carlo with different sampling probabilities...")

    SEARCH_RADIUS = 10
    print(f"MC-NLM with search window {2 * SEARCH_RADIUS + 1}×{2 * SEARCH_RADIUS + 1}")

    for xi in xis:
        print(f"sampling_prob = {xi}")

        mc_params = mc_nlm.MCNLMParams(
            sigma = sigma / 255.0,
            h_factor = 0.4,
            patch_size = 5,
            search_radius = SEARCH_RADIUS,
            spatial_sigma = 1e10,
            sampling_prob = xi
        )
        mc = mc_nlm.test_mcnlm(noisy, mc_params) * 255.0
        mc_clean_mse.append(mse(mc, image))
        mc_clean_psnr.append(psnr(mc, image))
        print(f"  MSE vs clean: {mc_clean_mse[-1]}, PSNR vs clean: {mc_clean_psnr[-1]}")

    return xis, mc_clean_mse, mc_clean_psnr, naive_clean_mse, native_clean_psnr


def mc_convergence(image_path):
    probs = np.linspace(0, 1, 13)
    # probs = [1]
    print('Testing convergence for probs: ', probs)
    xis, mc_clean_errors, mc_clean_psnr, naive_mse, naive_psnr = run_mc_convergence(image_path, xis=probs)

    plt.figure(figsize=(7,5))

    # MC-NLM MSE
    plt.plot(xis, mc_clean_errors, 'o-', label='MC-NLM vs clean')
    plt.axhline(y=naive_mse, color='r', linestyle='--', label=f'Naive NLM MSE = {naive_mse}')

    plt.xlabel('Sampling probs')
    plt.ylabel('MSE vs clean image')
    plt.title('MC-NLM Denoising vs Naive NLM')
    plt.xticks(xis)
    plt.grid(True)
    plt.legend()
    plt.show()

    # MC-NLM PSNR
    plt.plot(xis, mc_clean_psnr, 'o-', label='MC-NLM vs clean')
    plt.axhline(y=naive_psnr, color='r', linestyle='--', label=f'Naive NLM PSNR = {naive_psnr}')

    plt.xlabel('Sampling probs')
    plt.ylabel('PSNR vs clean image')
    plt.title('MC-NLM Denoising vs Naive NLM')
    plt.xticks(xis)
    plt.grid(True)
    plt.legend()
    plt.show()
