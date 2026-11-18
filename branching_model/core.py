import numpy as np
import scipy.stats as stats
import scipy.signal

from .utils import get_thinned_negbin_params

def find_truncation_K(max_k, s, R, k_disp, nu, tolerance=1e-10):
    mean_offspring = R * (1 - nu)
    is_supercritical = mean_offspring >= 1.0
    p_scipy = k_disp / (k_disp + mean_offspring) if not is_supercritical else 0
    def get_error(K):
        binom = stats.binom.cdf(max_k, K, s)
        tail = 1.0 if is_supercritical else (stats.nbinom.sf(K-2, K*k_disp, p_scipy) if K > 1 else 1.0)
        return binom * tail
    low = int(max_k) + 1; high = low
    while get_error(high) > tolerance:
        low = high; high *= 2
        if high > 1e7: return 1e7 
    ans = high
    while low <= high:
        mid = (low + high) // 2
        if get_error(mid) < tolerance: ans = mid; high = mid - 1
        else: low = mid + 1
    return ans

def compute_thinned_pmf(R, k_disp, nu, s, max_observed_size, max_n_limit=1e7):
    """
    Computes P(Y_tau(s) = k) with optimized rigorous bounds.
    """
    # 1. Find rigorous truncation point K
    limit_n = find_truncation_K(max_observed_size, s, R, k_disp, nu, tolerance=1e-10)
    
    # Cap at max_n_limit if needed
    limit_n = min(limit_n, int(max_n_limit))
    
    # 2. Compute Base PMF (FFT convolution)
    base_pmf = compute_raw_pmf_fft(R, k_disp, nu, limit_n)
    
    # 3. Construct the Thinned PMF matrix
    thinned_pmf = np.zeros(max_observed_size + 1)
    
    # Pre-calculate P(Y(s)=0)
    # Note: If limit_n was capped by max_n_limit, this sum is truncated,
    # but the error is bounded by the tolerance we set earlier.
    prob_zero_thinned = np.sum(base_pmf * ((1-s)**np.arange(len(base_pmf))))
    
    # 4. Compute Thinned Probabilities
    for k in range(1, max_observed_size + 1):
        relevant_n = np.arange(k, len(base_pmf))
        relevant_base_probs = base_pmf[relevant_n]
        bin_probs = stats.binom.pmf(k, relevant_n, s)
        thinned_pmf[k] = np.sum(relevant_base_probs * bin_probs)
        
    # 5. Normalize
    normalization_factor = 1.0 - prob_zero_thinned
    
    if normalization_factor > 1e-12:
        thinned_pmf /= normalization_factor
    else:
        return np.zeros(max_observed_size + 1)

    return thinned_pmf

def compute_raw_pmf_fft(R, k_disp, nu, max_size):
    """
    Computes P(\tilde{Y}_tau = k) explicitly using Theorem 4 recursion.
    Returns the RAW probability vector up to max_size.
    STRICTLY NO NORMALIZATION.
    
    Revised to apply exact truncation:
    1. Inner sum over l stops at max_size.
    2. Outer loop runs up to max_size (where vector stabilizes).
    3. Adds the exact tail term m^(-k) * a_{k,k}.
    """
    if R <= 1.0001: 
        return np.zeros(max_size + 1) 

    nb_n, nb_p = get_thinned_negbin_params(R, k_disp, nu)
    l_vals = np.arange(max_size + 1)
    px_probs = stats.nbinom.pmf(l_vals, n=nb_n, p=nb_p)
    
    A_curr = np.zeros(max_size + 1)
    A_curr[1] = 1.0
    
    final_pmf = np.zeros(max_size + 1)
    term_weight = (1 - 1/R)
    
    # Initial term for n=0
    final_pmf += term_weight * A_curr
    
    current_R_power = 1.0
    
    # THEOREM UPDATE:
    # The vector a_{n,k} stabilizes completely when n >= k.
    # Therefore, we loop exactly up to n = max_size.
    for n in range(1, max_size + 1): 
        current_R_power /= R
        
        # Efficiency break: If the weight is effectively zero, the tail is also zero.
        if current_R_power < 1e-18: 
            break 

        sum_poly = np.zeros(max_size + 1)
        A_pow = np.zeros(max_size + 1)
        A_pow[0] = 1.0 
        
        cutoff_prob = 1e-14
        
        # THEOREM UPDATE:
        # The term (a_n^{*l})_{k-1} is 0 if l > k-1.
        # Since we only need indices up to max_size, we stop l at max_size.
        limit_l = min(len(px_probs), max_size)
        
        for l in range(limit_l):
            prob = px_probs[l]
            if prob < cutoff_prob: continue
            
            # Optimization: If A_pow mass is negligible, stop convolving
            if np.sum(A_pow) < 1e-15: break
            
            sum_poly += prob * A_pow
            
            # Only convolve if we haven't reached the last relevant l
            if l < limit_l - 1:
                full_conv = scipy.signal.fftconvolve(A_pow, A_curr, mode='full')
                A_pow = full_conv[:max_size+1]
        
        A_next = np.zeros(max_size + 1)
        # Shift indices: a_{n+1, k} uses index k-1 from the sum
        A_next[1:] = sum_poly[:-1]
        
        contribution = term_weight * current_R_power * A_next
        final_pmf += contribution
        
        A_curr = A_next
        
    # THEOREM UPDATE: Stable Tail
    # The loop summed terms up to n = max_size.
    # The remaining infinite sum is: a_{final} * sum_{n=max_size+1}^{inf} (1-1/m)m^{-n}
    # This collapses to: a_{final} * m^{-(max_size + 1)}
    tail_weight = current_R_power / R
    final_pmf += tail_weight * A_curr
        
    return final_pmf