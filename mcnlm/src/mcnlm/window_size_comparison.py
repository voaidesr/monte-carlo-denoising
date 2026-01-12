import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import mcnlm.mc_nlm as mc_nlm
import mcnlm.naive_nlm as naive_nlm
from mcnlm.utils import add_gaussian_noise, load_image, mse


def window_size_comparison(
    image_path,
    output_path,
    search_radii,
    sampling_prob=0.5,
    sigma=17.0,
    h_factor=0.4,
    patch_radius=2,
    resize_to=None,
):
    image = load_image(str(image_path))
    if resize_to is not None:
        image = cv2.resize(image, resize_to, interpolation=cv2.INTER_AREA)
    noisy = add_gaussian_noise(image * 255.0, sigma=sigma).astype(np.float32) / 255.0

    np.random.seed(0)

    nlm_mse = []
    mc_mse = []
    window_sizes = []

    for r in search_radii:
        r = int(r)
        nlm_params = naive_nlm.NLMParams(
            sigma=sigma / 255.0,
            h_factor=h_factor,
            patch_radius=patch_radius,
            search_radius=r,
        )
        nlm = naive_nlm.test_naive_nlm(noisy, nlm_params)
        nlm_mse.append(mse(nlm, image))

        mc_params = mc_nlm.MCNLMParams(
            sigma=sigma / 255.0,
            h_factor=h_factor,
            patch_size=patch_radius * 2 + 1,
            search_radius=r,
            spatial_sigma=1e10,
            sampling_prob=sampling_prob,
        )
        mc = mc_nlm.test_mcnlm(noisy, mc_params)
        mc_mse.append(mse(mc, image))

        window_sizes.append(2 * r + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(window_sizes, nlm_mse, "o-", label="NLM")
    plt.plot(window_sizes, mc_mse, "o-", label=f"MCNLM (p={sampling_prob})")
    plt.xlabel("Search window size (pixels)")
    plt.ylabel("MSE")
    plt.title("NLM vs MCNLM MSE vs window size")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight", dpi=300)
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parents[3]
    window_size_comparison(
        image_path=repo_root / "mcnlm" / "imgs" / "moon.tiff",
        output_path=repo_root / "docs" / "res" / "window_size_mse.pdf",
        search_radii=range(8, 51, 3),
        sampling_prob=0.5,
        resize_to=(64, 64),
    )


if __name__ == "__main__":
    main()
