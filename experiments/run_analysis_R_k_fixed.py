import time
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
import random
from scipy.optimize import minimize
from tqdm import tqdm

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# --- IMPORTS ---
from branching_model.simulation import run_branching_simulation
from branching_model.inference import negative_log_likelihood

# --- CONFIGURATION ---
TARGET_BIO_SIZE = 200000    
S_FIXED = 1
REPLICATES = 500            

# Update this to the exact filename of your new empirical table
EMPIRICAL_LIMITS_CSV = f"empirical_truncation_K_S{S_FIXED}.csv"
OUTPUT_FILE = f"experiment_two_stage/experiment_two_stage_size{TARGET_BIO_SIZE}_S{S_FIXED}.csv"

# Optimizer Settings
MAX_STEPS = 2000            
FTOL = 1e-9                

# Cores
NUM_CORES = 20

# --- PART 1: EMPIRICAL LOOKUP TABLE ---
def build_empirical_table(csv_path):
    """
    Reads the empirical table and returns a dictionary mapping 
    (R, k, nu) -> (Max_Cluster_Allowed, Required_K)
    """
    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: Empirical file {csv_path} not found. Exiting.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    lookup_table = {}
    
    for _, row in df.iterrows():
        # Handle the "10/7" edge case if it was saved as a string, else use float
        r_val = 10/7 if row['R'] == "10/7" or str(row['R']).startswith("1.428") else float(row['R'])
        
        # Round keys slightly to prevent float-matching errors
        key = (round(r_val, 4), round(float(row['k']), 4), round(float(row['nu']), 4))
        
        # Adjust column names here if your CSV named them slightly differently
        max_allowed = int(row['P99p5_Max_Cluster'])
        required_k = int(row['Required_K_Bare_CDF'])
        
        lookup_table[key] = (max_allowed, required_k)
        
    print(f"Empirical limits loaded for {len(lookup_table)} parameter combinations.")
    return lookup_table

# --- PHASE 1 WORKER: SIMULATION ONLY ---
def simulation_worker(args):
    """
    Generates data and checks empirical safety limit. 
    """
    R_true, k_true, nu, rep_id, max_allowed, required_k = args
    
    start_sim = time.time()
    max_attempts = 2000 
    attempts = 0
    
    try:
        while True:
            attempts += 1
            if attempts > max_attempts:
                return {
                    "meta": (R_true, k_true, nu, rep_id),
                    "status": "FAILED_ATTEMPTS",
                    "clusters": None,
                    "sim_time": time.time() - start_sim
                }
            
            clusters, bio_size = run_branching_simulation(
                R_true, k_true, nu, S_FIXED, 
                max_tree_size=TARGET_BIO_SIZE
            )
            if bio_size >= TARGET_BIO_SIZE:
                break
                
        sim_time = time.time() - start_sim
        max_cluster = max(clusters) if clusters else 0
        
        # RIGOROUS SAFETY CHECK (Based on P99.5)
        if max_cluster > max_allowed:
            return {
                "meta": (R_true, k_true, nu, rep_id),
                "status": "DISCARDED_TOO_BIG",
                "clusters": None,
                "max_cluster": max_cluster,
                "limit": max_allowed,
                "sim_time": sim_time
            }
            
        return {
            "meta": (R_true, k_true, nu, rep_id),
            "status": "READY",
            "clusters": clusters,
            "max_cluster": max_cluster,
            "required_k": required_k,  # Pass the optimal K forward
            "sim_time": sim_time
        }
        
    except Exception as e:
        return {
            "meta": (R_true, k_true, nu, rep_id),
            "status": "ERROR",
            "message": str(e),
            "clusters": None
        }

# --- PHASE 2 WORKER: INFERENCE ONLY ---
def inference_worker(args):
    """
    Runs MLE on pre-generated clusters using pre-calculated K.
    """
    R_true, k_true, nu, rep_id, clusters, sim_time, max_cluster, required_k = args
    
    try:
        start_opt = time.time()
        initial_guess = [R_true, k_true] 
        
        # Inject the pre-calculated K into fixed_vals
        fixed_vals = {
            'nu': nu, 
            's': S_FIXED, 
            'precalculated_K': required_k  # Send this to your likelihood module
        }
        
        bnds = ((1.001, 20.0), (0.001, 10.0))
        
        res = minimize(
            negative_log_likelihood,
            initial_guess,
            args=(clusters, 'joint', fixed_vals),
            method='L-BFGS-B',
            bounds=bnds,
            options={'maxiter': MAX_STEPS, 'ftol': FTOL}
        )
        opt_time = time.time() - start_opt
        
        return {
            "R_true": R_true, "k_true": k_true, "nu": nu, "Replicate": rep_id,
            "Success": res.success, 
            "Message": res.message,
            "R_est": res.x[0], "k_est": res.x[1],
            "N_Clusters": len(clusters),
            "Max_Cluster": max_cluster,
            "Required_K": required_k,
            "Sim_Time_Sec": sim_time,
            "Opt_Time_Sec": opt_time,
            "Steps": res.nfev
        }
    except Exception as e:
         return {
            "R_true": R_true, "k_true": k_true, "nu": nu, "Replicate": rep_id,
            "Success": False, "Message": f"Inference Crash: {str(e)}"
        }
    
# --- MAIN LOOP ---
def run_experiment():
    print(f"--- STARTING TWO-STAGE EXPERIMENT (Target={TARGET_BIO_SIZE}) ---")
    
    # 1. SETUP TASKS FROM EMPIRICAL TABLE
    empirical_lookup = build_empirical_table(EMPIRICAL_LIMITS_CSV)
    
    R_VALUES = [10/7, 2.5]
    K_VALUES = [0.1, 0.5]
    NU_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    sim_tasks = []
    for R in R_VALUES:
        for k in K_VALUES:
            for nu in NU_VALUES:
                # Format key to match lookup table
                key = (round(R, 4), round(k, 4), round(nu, 4))
                
                # Fetch limits. If missing, fail safely rather than guessing.
                if key not in empirical_lookup:
                    print(f"Warning: No empirical data for {key}. Skipping.")
                    continue
                    
                max_allowed, required_k = empirical_lookup[key]
                
                for rep in range(REPLICATES):
                    sim_tasks.append((R, k, nu, rep, max_allowed, required_k))
    
    random.shuffle(sim_tasks)
    print(f"Total Simulations Planned: {len(sim_tasks)}")
    
    # 2. RUN PHASE 1: SIMULATION
    print("\n>>> PHASE 1: SIMULATION & SAFETY CHECK <<<")
    valid_data_for_inference = []
    discarded_count = 0
    failed_count = 0
    
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(NUM_CORES) as pool:
        for res in tqdm(pool.imap_unordered(simulation_worker, sim_tasks), total=len(sim_tasks)):
            meta = res['meta']
            
            if res['status'] == 'READY':
                valid_data_for_inference.append((
                    meta[0], meta[1], meta[2], meta[3], 
                    res['clusters'], 
                    res['sim_time'],
                    res['max_cluster'],
                    res['required_k']  # Append the K
                ))
            elif res['status'] == 'DISCARDED_TOO_BIG':
                discarded_count += 1
            else:
                failed_count += 1

    print("\n>>> PHASE 1 REPORT <<<")
    print(f"  Total Simulations: {len(sim_tasks)}")
    print(f"  Valid & Safe:      {len(valid_data_for_inference)}")
    print(f"  Discarded (Huge):  {discarded_count}")
    print(f"  Failed (Error):    {failed_count}")
    
    if len(valid_data_for_inference) == 0:
        print("No valid simulations to process. Exiting.")
        return

    # 3. RUN PHASE 2: INFERENCE
    print("\n>>> PHASE 2: INFERENCE (MLE) <<<")
    
    cols = ["R_true", "k_true", "nu", "Replicate", "R_est", "k_est", 
            "Success", "Message", "N_Clusters", "Max_Cluster", "Required_K",
            "Sim_Time_Sec", "Opt_Time_Sec", "Steps"]
            
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=cols).to_csv(OUTPUT_FILE, index=False)
        
    with ctx.Pool(NUM_CORES) as pool:
        for res in tqdm(pool.imap_unordered(inference_worker, valid_data_for_inference), total=len(valid_data_for_inference)):
            pd.DataFrame([res]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    print(f"\n--- EXPERIMENT COMPLETE ---")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_experiment()