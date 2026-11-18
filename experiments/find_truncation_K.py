import time
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import multiprocessing
from collections import Counter
from tqdm import tqdm

# --- CONFIGURATION ---
TARGET_BIO_SIZE = 200000
S_FIXED = 1
REPLICATES = 2000         # Runs per parameter set to estimate the 99th percentile
PERCENTILE_CUTOFF = 99.5   # Keep 99.5% of valid dynamics, drop massive outliers
TOLERANCE = 1e-10
NUM_CORES = 20

# Parameters (Master Grid for all experiments)
R_VALUES = [1.1, 4/3, 10/7, 1.5, 2.0, 2.5, 3.0]
K_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0]
NU_VALUES = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# --- 1. EXACT BRANCHING SIMULATION ---
def run_branching_simulation(R, k, nu, s, max_tree_size=10000):
    p_param = k / (k + R)
    n_param = k
    current_gen_ids = [0] 
    all_recorded_ids = [0]
    next_genotype_id = 1
    
    while len(all_recorded_ids) < max_tree_size and len(current_gen_ids) > 0:
        next_gen_ids = []
        for parent_id in current_gen_ids:
            num_offspring = np.random.negative_binomial(n_param, p_param)
            if num_offspring > 0:
                for _ in range(num_offspring):
                    if np.random.rand() < nu:
                        child_id = next_genotype_id
                        next_genotype_id += 1
                    else:
                        child_id = parent_id
                    next_gen_ids.append(child_id)
                    all_recorded_ids.append(child_id)
        current_gen_ids = next_gen_ids

    total_nodes = len(all_recorded_ids)
    mask = np.random.rand(total_nodes) < s
    sampled_genotypes = np.array(all_recorded_ids)[mask]
    
    if len(sampled_genotypes) == 0:
        return [], total_nodes
        
    counts = Counter(sampled_genotypes)
    return list(counts.values()), total_nodes

# --- 2. BARE CDF TRUNCATION SEARCH ---
def find_truncation_K(max_k, s, R, k_disp, nu, tolerance=1e-10):
    """
    Uses ONLY the bare binomial CDF and NB survival function to find truncation.
    """
    mean_offspring = R * (1 - nu)
    is_supercritical = mean_offspring >= 1.0
    p_scipy = k_disp / (k_disp + mean_offspring) if not is_supercritical else 0
    
    def get_error(K):
        binom = stats.binom.cdf(max_k, K, s)
        tail = 1.0 if is_supercritical else (stats.nbinom.sf(K-2, K*k_disp, p_scipy) if K > 1 else 1.0)
        return binom * tail

    low = int(max_k) + 1
    high = low
    while get_error(high) > tolerance:
        low = high
        high *= 2
        if high > 1e7: return int(1e7) 
            
    ans = high
    while low <= high:
        mid = (low + high) // 2
        if get_error(mid) < tolerance: 
            ans = mid; high = mid - 1
        else: 
            low = mid + 1
            
    return 2 ** math.ceil(math.log2(max(ans, 16)))

# --- 3. PARALLEL WORKER ---
def worker(args):
    R, k, nu = args
    
    max_observed_clusters = []
    start_time = time.time()
    
    for _ in range(REPLICATES):
        clusters, _ = run_branching_simulation(R, k, nu, S_FIXED, max_tree_size=TARGET_BIO_SIZE)
        if clusters:
            max_observed_clusters.append(max(clusters))
        else:
            max_observed_clusters.append(0)
            
    elapsed = time.time() - start_time
    
    if max_observed_clusters:
        p99p5_max_cluster = int(np.percentile(max_observed_clusters, PERCENTILE_CUTOFF))
        absolute_max = max(max_observed_clusters)
    else:
        p99p5_max_cluster = 0
        absolute_max = 0
        
    # Compute K using the bare CDF for the 99th percentile maximum cluster
    required_K = find_truncation_K(p99p5_max_cluster, S_FIXED, R, k, nu, TOLERANCE)

    return {
        "R": float(R) if R != 10/7 else "10/7",
        "k": k,
        "nu": nu,
        "Replicates": REPLICATES,
        "Absolute_Max": absolute_max,
        "P99p5_Max_Cluster": p99p5_max_cluster,
        "Required_K_Bare_CDF": required_K,
        "Sim_Time_Sec": round(elapsed, 2)
    }

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print(f"--- STARTING EMPIRICAL K ESTIMATION (Target Size = {TARGET_BIO_SIZE}) ---")
    
    tasks = [(R, k, nu) for R in R_VALUES for k in K_VALUES for nu in NU_VALUES]
    
    results = []
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(NUM_CORES) as pool:
        for res in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
            results.append(res)
            
    df = pd.DataFrame(results).sort_values(by=['R', 'k', 'nu'])
    
    print("\n>>> EMPIRICAL PERCENTILES & REQUIRED K TABLE (S=0.5) <<<")
    pd.set_option('display.max_rows', None)
    print(df.to_string(index=False))
    
    output_file = f"empirical_truncation_K_S{S_FIXED}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")