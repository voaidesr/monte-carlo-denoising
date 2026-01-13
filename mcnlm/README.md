# MCNLM Python Implementation

## Structure
Our project is structured in multiple parts:
- paralelized Naive NLM implementation: `naive_nlm.py`
- paralelized Monte Carlo NLM implementation: `mc_nlm.py`
- Monte Carlo simulator: `mc_convergence.py`

Image processing utilities are in `utils.py`.

## Paralelization
To maximize efficiency, the implementation leverages **Numba-based parallelization**, achieving significant performance improvements by distributing the computationally intensive Monte Carlo sampling and patch weight calculations—which have a complexity of $\mathcal{O}(mkd)$ across multiple CPU cores.

- JIT Compilation: Uses Numba's Just-In-Time (JIT) compilation to run Python code at near-native C++ speeds.
- Multi-Core Execution: Offloads the large volume of independent pixel-wise operations to handle high-resolution images in a fraction of the time.

## Running instructions
We are using **poetry** for managing dependencies.

```bash
# install poetry
pip install --user poetry

# project dir
cd mcnlm

# install deps
poetry install

# activate environment
poetry env activate

# run
poetry run mcnlm
```

## Resources
- [Poetry](https://python-poetry.org/)
- [Numba](https://numba.pydata.org/)
