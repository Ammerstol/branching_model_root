# branching_model/utils.py
import numpy as np

def get_thinned_negbin_params(mean_R, disp_k, nu):
    """
    Returns parameters (n, p) for Scipy nbinom corresponding to 
    \tilde{X} ~ Bin(NegBin(R, k), 1-nu).
    
    Relations:
    Mean = n(1-p)/p = R  => p = n / (n+R)
    """
    R_tilde = mean_R * (1 - nu)
    n = disp_k
    p = disp_k / (disp_k + R_tilde)
    return n, p