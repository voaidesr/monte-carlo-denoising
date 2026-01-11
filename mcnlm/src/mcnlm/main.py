import mcnlm.mc_nlm as mc_nlm
import mcnlm.naive_nlm as naive_nlm
from mcnlm.mc_convergence import mc_convergence

import numpy as np

def main():
    # mc_convergence("imgs/land.tiff")
    mc_nlm.test_mcnlm("imgs/clock.tiff")
    #naive_nlm.test_naive_nlm("imgs/clock.tiff")
    # mc_nlm.show_matches("imgs/city.tiff", [(100, 128), (170, 50), (200, 220), (150, 220), (200, 140)])
    # mc_nlm.show_mask("imgs/clock.tiff", 10, 128)
    
    naive_nlm.test_naive_nlm()