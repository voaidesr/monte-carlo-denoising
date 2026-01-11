import numpy as np
import matplotlib.pyplot as plt

from mcnlm.utils import mse, add_gaussian_noise, load_image, psnr
import mcnlm.mc_nlm as mc_nlm
import mcnlm.naive_nlm as naive_nlm

def run_mc_convergence(image_path, xis):
    """
    Run MCNLM for different sampling probabilities and compute MSE between MC method and naive NLM.
    """
    image = load_image(image_path)
    sigma = 17
    h_factor = 0.4
    noisy = add_gaussian_noise(image * 255, sigma=sigma).astype(np.float32) / 255.0

    # Modify here to integrate matteo
    
    # nlm_params = naive_nlm.NLMParams(
    #     sigma =  sigma / 255.0,
    #     h_factor = 0.4,
    #     patch_radius = 2,
    #     search_radius = 10
    # )
    
    # compute naive NLM
    print("Computing naive NLM...")
    nlm_ref = naive_nlm.nlm_denoising(noisy, sigma, h_factor)
    naive_clean_mse = mse(nlm_ref, image)
    naive_clean_psnr = psnr(nlm_ref, image)

    mc_clean_mse, mc_naive_mse = [], []
    mc_clean_psnr, mc_naive_psnr = [], []
    print("Running Monte-Carlo with different sampling probabilities...")
    for xi in xis:
        print(f"sampling_prob = {xi}")

        mc_params = mc_nlm.MCNLMParams(
            sigma = sigma / 255.0,
            h_factor = 0.4,
            patch_size = 5,
            search_radius = 10,
            spatial_sigma = 10,
            sampling_prob = xi
        )
        
        mc = mc_nlm.mcnlm_denoise(noisy, mc_params)
        
        mc_clean_mse.append(mse(mc, image))
        mc_naive_mse.append(mse(mc, nlm_ref))
        
        mc_clean_psnr.append(psnr(mc, image))
        mc_naive_psnr.append(psnr(mc, nlm_ref))
        
        
    return xis, mc_clean_mse, mc_naive_mse, mc_clean_psnr, mc_naive_psnr, naive_clean_mse, naive_clean_psnr


def mc_convergence(image_path):
    
    probs = np.linspace(0, 1, 13)
    # probs = [1]
    print('Testing convergence for probs: ', probs)
    xis, mc_clean_errors, mc_naive_errors, mc_clean_psnr, mc_naive_psnr, naive_mse, naive_clean_psnr = run_mc_convergence(image_path, xis=probs)
    
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
    plt.axhline(y=naive_clean_psnr, color='r', linestyle='--', label=f'Naive NLM MSE = {naive_mse}')

    plt.xlabel('Sampling probs')
    plt.ylabel('PSNR vs clean image')
    plt.title('MC-NLM Denoising vs Naive NLM')
    plt.xticks(xis)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    # plt.figure(figsize=(7,5))
    # plt.plot(xis, mc_naive_errors, 'o-', label='MC-NLM vs Naive NLM')
    # plt.xlabel('Sampling probability')
    # plt.ylabel('MSE vs Naive NLM')
    # plt.title('MC-NLM convergence to Naive NLM')
    # plt.grid(True)
    # plt.legend()
    # plt.show()