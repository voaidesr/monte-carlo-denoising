# MCNLM
Monte Carlo methods for image denoising. 

# Monte Carlo Non-Local Means (MCNLM) Optimization

This repository implements an optimized **Non-Local Means (NLM)** denoising algorithm using **Monte Carlo sampling**. By selecting a random subset of patches rather than the entire image, this approach significantly reduces computational overhead while maintaining high visual fidelity.

This is an implementation of the [Monte Carlo Non-Local Means paper by Stanley H. Chan, Todd Zickler, Yue M. Lu](https://arxiv.org/pdf/1312.7366).

## Implementation
The python implementation documentation is available [here](./mcnlm/README.md).

## Features

### Efficiency & Optimization
* **Complexity Reduction**: Lowers computational cost from $\mathcal{O}(mnd)$ to $\mathcal{O}(mkd)$, where $k \ll n$.
* **Bernoulli Sampling**: Uses a probability vector $p$ to generate a reference set of patches.
* **Exponential Convergence**: Theoretical bounds ensure that the estimation error drops exponentially as the sample size increases.

### Improved Methodology
* **Spatial Locality**: Combines structural similarity with spatial proximity weights to preserve distinct features like stars or sharp edges.
* **Noise Estimation (FFT)**: Includes a Fast Fourier Transform module to estimate unknown noise deviation ($\sigma$) in the frequency domain.

## Performance Summary

The MCNLM algorithm achieves quality comparable to full NLM but at a fraction of the cost.

| Metric | Noisy Image | MCNLM ($p=0.3$) | MCNLM ($p=0.8$) | Naive NLM |
| :--- | :--- | :--- | :--- | :--- |
| **MSE** | ~275.4 [cite: 116] | 97.06 [cite: 117] | 58.16 [cite: 117] | 69.14 [cite: 162] |
| **PSNR** | - | ~28.8 dB [cite: 167] | ~29.5 dB [cite: 164] | 29.73 [cite: 173] |

* **Visual Results**: Effectively removes additive white Gaussian noise while preserving textures.
* **Diminishing Returns**: MSE drops rapidly with initial sampling.
* **Reliability**: High reliability is maintained even with only 5% of samples.

*Date: January 13, 2026* [cite: 4]

## Resources
- [Original NLM paper](https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf)
- [Monte Carlo Non-Local Means paper](https://arxiv.org/pdf/1312.7366)
