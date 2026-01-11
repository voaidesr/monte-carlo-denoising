import numpy as np
import matplotlib.pyplot as plt

from mcnlm.utils import mse, add_gaussian_noise, load_image
import mcnlm.mc_nlm as mc_nlm
import mcnlm.naive_nlm as naive_nlm


def run_mc_convergence(img_path, sigma, xis, patch_size=5, search_radius=15, h_factor=0.8, seed=0):
    """
    Run MCNLM for different sampling probabilities and compute MSE between MC method and naive NLM.
    """
    #np.random.seed(seed)
    
    # load image and add noise
    img = load_image(img_path)
    noisy = add_gaussian_noise(img * 255, sigma=sigma * 255).astype(np.float32) / 255.0

    nlm_params = naive_nlm.NLMParams(
        sigma=sigma,
        patch_radius=patch_size // 2,
        search_radius=search_radius,
        h_factor=h_factor
    )
    
    # compute naive NLM
    print("Computing naive NLM...")
    nlm_ref = naive_nlm.nlm_denoise(noisy, nlm_params)
    naive_clean_mse = mse(nlm_ref, img)

    mc_clean_errors, mc_naive_errors = [], []
    print("Running Monte-Carlo with different sampling probabilities...")
    for xi in xis:
        print(f"sampling_prob = {xi}")

        mc_params = mc_nlm.MCNLMParams(
            sigma=sigma,
            patch_size=patch_size,
            search_radius=search_radius,
            sampling_prob=xi,
            h_factor=h_factor
        )
        mc = mc_nlm.mcnlm_denoise(noisy, mc_params)

        mc_vs_clean = mse(mc, img)
        mc_vs_naive = mse(mc, nlm_ref)
    
        mc_clean_errors.append(mc_vs_clean)
        mc_naive_errors.append(mc_vs_naive)

        print(f"MC vs clean MSE = {mc_vs_clean:.6e}")
        print(f"MC vs NLM MSE = {mc_vs_naive:.6e}")
        
        print()
    
    return xis, mc_clean_errors, mc_naive_errors, naive_clean_mse


def mc_convergence():
    probs = np.linspace(0, 1, 13)
    print('Testing convergence for probs: ', probs)
    xis, mc_clean_errors, mc_naive_errors, naive_mse = run_mc_convergence(img_path="imgs/man.tiff", sigma=17/255, xis = probs)
    
    plt.figure(figsize=(7,5))
    
    # MC-NLM MSE
    plt.plot(xis, mc_naive_errors, 'o-', label='MC-NLM vs clean')
    plt.axhline(y=naive_mse, color='r', linestyle='--', label=f'Naive NLM MSE = {naive_mse}')

    plt.xlabel('Sampling probs')
    plt.ylabel('MSE vs clean image')
    plt.title('MC-NLM Denoising vs Naive NLM')
    plt.xticks(xis)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(7,5))
    plt.plot(xis, mc_clean_errors, 'o-', label='MC-NLM vs Naive NLM')
    plt.xlabel('Sampling probability')
    plt.ylabel('MSE vs Naive NLM')
    plt.title('MC-NLM convergence to Naive NLM')
    plt.grid(True)
    plt.legend()
    plt.show()