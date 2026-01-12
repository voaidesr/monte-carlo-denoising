import mcnlm.mc_nlm as mc_nlm
import mcnlm.naive_nlm as naive_nlm
from mcnlm.mc_convergence import mc_convergence

from mcnlm.utils import show_mcnlm_result_zoomed, show_matches
import numpy as np

def results_mcnlm():
    show_mcnlm_result_zoomed(
        "imgs/land.tiff",
        probs=[0.3, 0.5, 0.8],
        zoom=(120, 100, 64, 64),
        output_path="../docs/res/mcnlm3.pdf",
    )
    
    show_mcnlm_result_zoomed(
        "imgs/clock.tiff",
        probs=[0.3, 0.5, 0.8],
        zoom=(120, 100, 64, 64),
        output_path="../docs/res/mcnlm1.pdf",
    )
    
    show_mcnlm_result_zoomed(
        "imgs/man.tiff",
        probs=[0.3, 0.5, 0.8],
        zoom=(440, 600, 64, 64),
        output_path="../docs/res/mcnlm2.pdf",
    )

def main():
    # results_mcnlm()
    show_matches('imgs/clock.tiff', [(150, 210), (90, 135), (170, 80)], "../docs/res/strong_matches_p1.pdf")