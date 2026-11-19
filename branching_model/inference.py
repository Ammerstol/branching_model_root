import numpy as np
from scipy.optimize import minimize
from .core import (
    compute_raw_pmf_fft,
    compute_thinned_pmf,
    compute_thinned_pmf_tran_kiem
)

def negative_log_likelihood(params, data, mode, fixed_vals):
    nu = fixed_vals['nu']
    s = fixed_vals.get('s', 1.0)  # Default to 1.0 (no thinning) if not present
    if mode == 'joint':
        R, k = params
    elif mode == 'fix_R':
        R = fixed_vals['R']
        k = params[0]
    elif mode == 'fix_k':
        k = fixed_vals['k']
        R = params[0]
    
    if R <= 1.001 or k <= 0.001:
        return 1e12
    
    max_observed = np.max(data)
    
    # --- BRANCHING LOGIC ---
    if abs(s - 1.0) < 1e-5:
        # Standard Case (s=1)
        pmf = compute_raw_pmf_fft(R, k, nu, max_size=int(max_observed))
    else:
        # Thinned Case (s < 1)
        pmf = compute_thinned_pmf(R, k, nu, s, max_observed_size=int(max_observed))
    
    safe_indices = np.clip(data, 0, len(pmf)-1)
    probs = pmf[safe_indices]
    probs = np.maximum(probs, 1e-20)
    nll = -np.sum(np.log(probs))
    return nll

def run_mle_optimization(data, mode, fixed_vals, initial_guess):
    opt_options = {'disp': False, 'maxiter': 500}
    res = minimize(
        negative_log_likelihood, 
        initial_guess, 
        args=(data, mode, fixed_vals),
        method='Nelder-Mead', 
        options=opt_options
    )
    return res.x

def negative_log_likelihood_tran_kiem(params, data, mode, fixed_vals):

    nu = fixed_vals['nu']
    s = fixed_vals.get('s', 1.0)

    if mode == 'joint':
        R, k = params
    elif mode == 'fix_R':
        R = fixed_vals['R']
        k = params[0]
    elif mode == 'fix_k':
        k = fixed_vals['k']
        R = params[0]

    if R <= 0 or k <= 0:
        return 1e12

    max_observed = np.max(data)

    pmf = compute_thinned_pmf_tran_kiem(
        R,
        k,
        nu,
        s,
        max_observed
    )

    safe_indices = np.clip(data, 0, len(pmf)-1)

    probs = pmf[safe_indices]
    probs = np.maximum(probs, 1e-20)

    nll = -np.sum(np.log(probs))

    return nll


def run_mle_optimization_tran_kiem(
    data, mode, fixed_vals, initial_guess
):

    opt_options = {'disp': False, 'maxiter': 500}

    res = minimize(
        negative_log_likelihood_tran_kiem,
        initial_guess,
        args=(data, mode, fixed_vals),
        method='Nelder-Mead',
        options=opt_options
    )

    return res.x
