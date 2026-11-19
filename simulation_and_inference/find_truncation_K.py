
import argparse
import time
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- IMPORTS ---
from branching_model.simulation import run_branching_simulation
from simulation_and_inference.experiment_config import load_experiment


# ============================================================
# PARALLEL WORKER
# ============================================================

def worker(args):

    (
        R,
        k,
        nu,
        target_bio_size,
        s_fixed,
        replicates,
        percentile_cutoff,
        max_attempts
    ) = args

    max_observed_clusters = []
    start_time = time.time()

    # --------------------------------------------------------
    # 1. Simulate branching processes
    # --------------------------------------------------------

    for rep in range(replicates):

        attempts = 0

        while True:

            attempts += 1

            if attempts > max_attempts:
                return {
                    "R": float(R) if R != 10/7 else "10/7",
                    "k": k,
                    "nu": nu,
                    "status": "FAILED_ATTEMPTS"
                }

            clusters, bio_size = run_branching_simulation(
                R,
                k,
                nu,
                s_fixed,
                max_tree_size=target_bio_size
            )

            if bio_size >= target_bio_size:
                break

        max_observed_clusters.append(
            max(clusters) if clusters else 0
        )

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # 2. Calculate empirical maximum-cluster percentile
    # --------------------------------------------------------

    p99_max_cluster = int(np.ceil(
        np.percentile(
            max_observed_clusters,
            percentile_cutoff
        )
    ))

    absolute_max = max(max_observed_clusters)

    # --------------------------------------------------------
    # 3. Return empirical safety limit
    # --------------------------------------------------------

    return {
        "R": float(R) if R != 10/7 else "10/7",
        "k": k,
        "nu": nu,
        "Replicates": replicates,
        "Absolute_Max": absolute_max,
        "P99p5_Max_Cluster": p99_max_cluster,
        "Sim_Time_Sec": round(elapsed, 2),
        "status": "READY"
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(experiment_name):

    # --------------------------------------------------------
    # 1. LOAD EXPERIMENT PARAMETERS
    # --------------------------------------------------------

    config, output_dir = load_experiment(experiment_name)

    target_bio_size = config["target_bio_size"]
    s_fixed = config["sampling_fraction"]

    R_values = config["parameter_grid"]["R"]
    k_values = config["parameter_grid"]["k"]
    nu_values = config["parameter_grid"]["nu"]

    replicates = config["empirical"]["replicates"]
    percentile_cutoff = config["empirical"]["percentile_cutoff"]
    max_attempts = config["empirical"]["max_attempts"]
    num_cores = config["empirical"]["num_cores"]

    output_file = output_dir / "empirical_max_cluster.csv"

    print("\n========================================")
    print(f"EMPIRICAL MAXIMUM-CLUSTER ESTIMATION")
    print(f"Experiment:             {experiment_name}")
    print("========================================")
    print(f"Target biological size: {target_bio_size}")
    print(f"Sampling fraction:      {s_fixed}")
    print(f"Replicates:             {replicates}")
    print(f"Percentile:             {percentile_cutoff}")
    print(f"Number of cores:        {num_cores}")
    print(f"Results directory:      {output_dir}")

    # --------------------------------------------------------
    # 2. SET UP SIMULATION TASKS
    # --------------------------------------------------------

    tasks = [
        (
            R,
            k,
            nu,
            target_bio_size,
            s_fixed,
            replicates,
            percentile_cutoff,
            max_attempts
        )
        for R in R_values
        for k in k_values
        for nu in nu_values
    ]

    print(f"\nTotal parameter combinations: {len(tasks)}")

    results = []
    failed_count = 0

    # --------------------------------------------------------
    # 3. RUN SIMULATIONS
    # --------------------------------------------------------

    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(num_cores) as pool:

        for res in tqdm(
            pool.imap_unordered(worker, tasks),
            total=len(tasks)
        ):

            if res["status"] == "READY":

                results.append(res)

            else:

                failed_count += 1

                print(
                    f"Failed parameter combination: "
                    f"R={res['R']}, "
                    f"k={res['k']}, "
                    f"nu={res['nu']}"
                )

    # --------------------------------------------------------
    # 4. RESULTS TABLE
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    if not df.empty:

        # Sort numerically while preserving the "10/7" CSV label.
        df["_R_sort"] = df["R"].apply(
            lambda x: (
                10/7
                if str(x).strip() == "10/7"
                else float(x)
            )
        )

        df = (
            df.sort_values(
                by=["_R_sort", "k", "nu"]
            )
            .drop(columns=["_R_sort", "status"])
        )

    # --------------------------------------------------------
    # 5. SAVE RESULTS
    # --------------------------------------------------------

    df.to_csv(output_file, index=False)

    print("\n========================================")
    print("EMPIRICAL ESTIMATION COMPLETE")
    print("========================================")
    print(f"Successful combinations: {len(df)}")
    print(f"Failed combinations:     {failed_count}")
    print(f"Saved to: {output_file}")


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment_name",
        help="Name of the experiment, e.g. experiment_1"
    )

    args = parser.parse_args()

    run_experiment(args.experiment_name)
