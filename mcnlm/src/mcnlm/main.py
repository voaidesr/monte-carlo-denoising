import mcnlm.mc_nlm as mc_nlm
import mcnlm.naive_nlm as naive_nlm
from mcnlm.mc_convergence import mc_convergence

import numpy as np

def main():
    mc_convergence('imgs/clock.tiff')
    # mc_nlm.test_mcnlm()
    # naive_nlm.test_naive_nlm()