import argparse
import json
import pickle
import random
import time
import multiprocessing
from pathlib import Path

import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from branching_model.simulation import run_branching_simulation
from branching_model.inference import (
    negative_log_likelihood,
    negative_log_likelihood_tran_kiem,
)
from simulation_and_inference.experiment_config import load_experiment


# ============================================================
# FILE HELPERS
# ============================================================

def atomic_write_pickle(path, payload):
    """Write a pickle atomically so an interrupted write is not reused."""
    path = Path(path)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(path)


def atomic_write_json(path, payload):
    """Write JSON metadata atomically."""
    path = Path(path)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    temp_path.replace(path)


def load_manifest(path):
    """Return the latest manifest row for each Simulation_ID."""
    path = Path(path)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if df.empty or "Simulation_ID" not in df.columns:
        return {}

    return {
        str(row["Simulation_ID"]): row.to_dict()
        for _, row in df.iterrows()
    }


def completed_simulation_ids(path):
    """Read IDs already recorded in an inference-results file."""
    path = Path(path)
    if not path.exists():
        return set()

    columns = pd.read_csv(path, nrows=0).columns
    if "Simulation_ID" not in columns:
        raise RuntimeError(
            f"Existing results file has no Simulation_ID column: {path}\n"
            "It was created before simulation datasets were saved. "
            "Rename or move this old file before starting the persistent run."
        )

    try:
        values = pd.read_csv(path, usecols=["Simulation_ID"])
    except pd.errors.EmptyDataError:
        return set()

    return set(values["Simulation_ID"].dropna().astype(str))


def append_result(path, result, columns):
    """Append one inference result using the exact declared column order."""
    pd.DataFrame([result], columns=columns).to_csv(
        path,
        mode="a",
        header=False,
        index=False,
    )


# ============================================================
# EMPIRICAL MAXIMUM-CLUSTER LOOKUP
# ============================================================

def build_empirical_table(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Empirical maximum-cluster file not found: {csv_path}\n"
            "Run find_truncation_K first."
        )

    df = pd.read_csv(csv_path)
    lookup_table = {}

    for _, row in df.iterrows():
        r_string = str(row["R"]).strip()
        r_val = 10 / 7 if r_string == "10/7" else float(row["R"])

        key = (
            round(r_val, 4),
            round(float(row["k"]), 4),
            round(float(row["nu"]), 4),
        )
        lookup_table[key] = int(row["P99p5_Max_Cluster"])

    print(
        f"Empirical maximum-cluster limits loaded for "
        f"{len(lookup_table)} parameter combinations."
    )
    return lookup_table


# ============================================================
# PHASE 1: SIMULATION WORKER
# ============================================================

def simulation_worker(args):
    (
        simulation_id,
        R_true,
        k_true,
        nu,
        rep_id,
        max_allowed,
        target_bio_size,
        s_fixed,
        max_attempts,
        data_path,
        metadata_path,
    ) = args

    start_sim = time.time()
    attempts = 0

    try:
        while True:
            attempts += 1
            if attempts > max_attempts:
                return {
                    "Simulation_ID": simulation_id,
                    "R_true": R_true,
                    "k_true": k_true,
                    "nu": nu,
                    "Replicate": rep_id,
                    "status": "FAILED_ATTEMPTS",
                    "sim_time": time.time() - start_sim,
                }

            clusters, bio_size = run_branching_simulation(
                R_true,
                k_true,
                nu,
                s_fixed,
                max_tree_size=target_bio_size,
            )

            if bio_size >= target_bio_size:
                break

        sim_time = time.time() - start_sim
        max_cluster = max(clusters) if clusters else 0

        # Retain the empirical P99.5 safety filter.
        if max_cluster > max_allowed:
            return {
                "Simulation_ID": simulation_id,
                "R_true": R_true,
                "k_true": k_true,
                "nu": nu,
                "Replicate": rep_id,
                "status": "DISCARDED_TOO_BIG",
                "max_cluster": max_cluster,
                "max_allowed": max_allowed,
                "sim_time": sim_time,
            }

        # Save accepted cluster data immediately in the worker.
        # Both model likelihoods will later read this exact file.
        payload = {
            "Simulation_ID": simulation_id,
            "R_true": R_true,
            "k_true": k_true,
            "nu": nu,
            "Replicate": rep_id,
            "clusters": clusters,
            "bio_size": bio_size,
            "max_cluster": max_cluster,
            "sim_time": sim_time,
            "max_allowed": max_allowed,
        }
        atomic_write_pickle(data_path, payload)

        metadata = {
            "Simulation_ID": simulation_id,
            "R_true": float(R_true),
            "k_true": float(k_true),
            "nu": float(nu),
            "Replicate": int(rep_id),
            "max_cluster": int(max_cluster),
            "max_allowed": int(max_allowed),
            "bio_size": int(bio_size),
            "sim_time": float(sim_time),
            "data_file": str(data_path),
            "status": "READY",
        }
        atomic_write_json(metadata_path, metadata)

        # Do not send the potentially large cluster list back to the parent.
        return metadata

    except Exception as e:
        return {
            "Simulation_ID": simulation_id,
            "R_true": R_true,
            "k_true": k_true,
            "nu": nu,
            "Replicate": rep_id,
            "status": "ERROR",
            "message": str(e),
            "sim_time": time.time() - start_sim,
        }


# ============================================================
# PHASE 2: FIT ONE MODEL
# ============================================================

def fit_one_model(
    model_name,
    likelihood,
    simulation_id,
    R_true,
    k_true,
    nu,
    rep_id,
    clusters,
    max_cluster,
    sim_time,
    s_fixed,
    max_steps,
    ftol,
    bounds,
):
    start_opt = time.time()
    fixed_vals = {"nu": nu, "s": s_fixed}
    initial_guess = [R_true, k_true]

    try:
        res = minimize(
            likelihood,
            initial_guess,
            args=(clusters, "joint", fixed_vals),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_steps, "ftol": ftol},
        )

        return {
            "Simulation_ID": simulation_id,
            "Model": model_name,
            "R_true": R_true,
            "k_true": k_true,
            "nu": nu,
            "Replicate": rep_id,
            "Success": bool(res.success),
            "Message": str(res.message),
            "R_est": res.x[0],
            "k_est": res.x[1],
            "N_Clusters": len(clusters),
            "Max_Cluster": max_cluster,
            "Sim_Time_Sec": sim_time,
            "Opt_Time_Sec": time.time() - start_opt,
            "Steps": res.nfev,
        }

    except Exception as e:
        return {
            "Simulation_ID": simulation_id,
            "Model": model_name,
            "R_true": R_true,
            "k_true": k_true,
            "nu": nu,
            "Replicate": rep_id,
            "Success": False,
            "Message": f"Inference crash: {str(e)}",
            "R_est": None,
            "k_est": None,
            "N_Clusters": len(clusters),
            "Max_Cluster": max_cluster,
            "Sim_Time_Sec": sim_time,
            "Opt_Time_Sec": time.time() - start_opt,
            "Steps": None,
        }


# ============================================================
# PHASE 2: PAIRED INFERENCE WORKER
# ============================================================

def inference_worker(args):
    (
        simulation_id,
        data_path,
        metadata,
        run_ours,
        run_tran_kiem,
        s_fixed,
        max_steps,
        ftol,
        bounds,
    ) = args

    try:
        with Path(data_path).open("rb") as f:
            payload = pickle.load(f)

        if payload.get("Simulation_ID") != simulation_id:
            raise ValueError(
                f"Simulation ID mismatch in saved file {data_path}"
            )

        clusters = payload["clusters"]

    except Exception as e:
        return {
            "Simulation_ID": simulation_id,
            "load_error": str(e),
            "ours": None,
            "tran_kiem": None,
        }

    common = {
        "simulation_id": simulation_id,
        "R_true": metadata["R_true"],
        "k_true": metadata["k_true"],
        "nu": metadata["nu"],
        "rep_id": metadata["Replicate"],
        "clusters": clusters,
        "max_cluster": metadata["max_cluster"],
        "sim_time": metadata["sim_time"],
        "s_fixed": s_fixed,
        "max_steps": max_steps,
        "ftol": ftol,
        "bounds": bounds,
    }

    ours_result = None
    tran_kiem_result = None

    if run_ours:
        ours_result = fit_one_model(
            model_name="Ours",
            likelihood=negative_log_likelihood,
            **common,
        )

    if run_tran_kiem:
        tran_kiem_result = fit_one_model(
            model_name="Tran-Kiem",
            likelihood=negative_log_likelihood_tran_kiem,
            **common,
        )

    return {
        "Simulation_ID": simulation_id,
        "ours": ours_result,
        "tran_kiem": tran_kiem_result,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(experiment_name):
    config, output_dir = load_experiment(experiment_name)

    target_bio_size = config["target_bio_size"]
    s_fixed = config["sampling_fraction"]
    R_values = config["parameter_grid"]["R"]
    k_values = config["parameter_grid"]["k"]
    nu_values = config["parameter_grid"]["nu"]

    replicates = config["inference"]["replicates"]
    max_steps = config["inference"]["max_steps"]
    ftol = config["inference"]["ftol"]
    num_cores = config["inference"]["num_cores"]
    max_attempts = config["empirical"]["max_attempts"]

    bounds = (
        tuple(config["inference"]["bounds"]["R"]),
        tuple(config["inference"]["bounds"]["k"]),
    )

    empirical_file = output_dir / "empirical_max_cluster.csv"
    simulation_dir = output_dir / "simulation_data"
    simulation_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = output_dir / "simulation_manifest.csv"
    ours_file = output_dir / "two_stage_results_ours.csv"
    tran_kiem_file = output_dir / "two_stage_results_tran_kiem.csv"

    print("\n========================================")
    print(f"PAIRED MODEL COMPARISON: {experiment_name}")
    print("========================================")
    print(f"Target biological size: {target_bio_size}")
    print(f"Sampling fraction:      {s_fixed}")
    print(f"Inference replicates:   {replicates}")
    print(f"Number of cores:        {num_cores}")
    print(f"Simulation data folder: {simulation_dir}")

    # Existing output files from the previous, non-persistent script
    # cannot be linked to newly saved simulations. Do not mix them.
    for result_file in (ours_file, tran_kiem_file):
        if result_file.exists():
            header = pd.read_csv(result_file, nrows=0).columns
            if "Simulation_ID" not in header:
                raise RuntimeError(
                    f"Existing results file predates simulation persistence:\n"
                    f"{result_file}\n\n"
                    "Rename or move this old file before running this version. "
                    "The old estimates cannot be paired with newly generated "
                    "simulation files."
                )

    empirical_lookup = build_empirical_table(empirical_file)
    manifest_records = load_manifest(manifest_file)

    simulation_tasks = []
    accepted_simulations = []
    reused_count = 0
    previously_discarded = 0
    previously_failed = 0

    # --------------------------------------------------------
    # Build deterministic IDs and either reuse or simulate data.
    # --------------------------------------------------------
    for r_idx, R in enumerate(R_values):
        for k_idx, k in enumerate(k_values):
            for nu_idx, nu in enumerate(nu_values):
                key = (round(R, 4), round(k, 4), round(nu, 4))

                if key not in empirical_lookup:
                    print(f"Warning: No empirical limit for {key}; skipping.")
                    continue

                max_allowed = empirical_lookup[key]

                for rep in range(replicates):
                    simulation_id = (
                        f"r{r_idx:02d}_k{k_idx:02d}_nu{nu_idx:02d}_rep{rep:04d}"
                    )
                    data_path = simulation_dir / f"{simulation_id}.pkl"
                    metadata_path = simulation_dir / f"{simulation_id}.json"

                    # Reuse complete saved datasets on subsequent runs.
                    if data_path.exists() and metadata_path.exists():
                        with metadata_path.open("r", encoding="utf-8") as f:
                            metadata = json.load(f)

                        if metadata.get("Simulation_ID") != simulation_id:
                            raise ValueError(
                                f"Incorrect metadata for {simulation_id}: {metadata_path}"
                            )

                        accepted_simulations.append(metadata)
                        manifest_records[simulation_id] = {
                            **metadata,
                            "status": "READY",
                        }
                        reused_count += 1
                        continue

                    # A previous terminal discard/failure is not silently
                    # resimulated when resuming the same experiment.
                    previous = manifest_records.get(simulation_id, {})
                    previous_status = previous.get("status")

                    if previous_status == "DISCARDED_TOO_BIG":
                        previously_discarded += 1
                        continue

                    if previous_status in ("FAILED_ATTEMPTS", "ERROR"):
                        previously_failed += 1
                        continue

                    # Remove incomplete/corrupt partial files before retrying.
                    if data_path.exists():
                        data_path.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()

                    simulation_tasks.append((
                        simulation_id,
                        R,
                        k,
                        nu,
                        rep,
                        max_allowed,
                        target_bio_size,
                        s_fixed,
                        max_attempts,
                        str(data_path),
                        str(metadata_path),
                    ))

    random.shuffle(simulation_tasks)

    print(f"\nTotal simulation tasks to run: {len(simulation_tasks)}")
    print(f"Saved simulations reused:      {reused_count}")
    print(f"Previously discarded:          {previously_discarded}")
    print(f"Previously failed:              {previously_failed}")

    # --------------------------------------------------------
    # PHASE 1: Generate and save accepted simulations.
    # --------------------------------------------------------
    print("\n>>> PHASE 1: SIMULATION & PERSISTENCE <<<")

    discarded_count = 0
    failed_count = 0

    ctx = multiprocessing.get_context("spawn")
    if simulation_tasks:
        with ctx.Pool(num_cores) as pool:
            for res in tqdm(
                pool.imap_unordered(simulation_worker, simulation_tasks),
                total=len(simulation_tasks),
            ):
                simulation_id = res["Simulation_ID"]
                manifest_records[simulation_id] = res

                if res["status"] == "READY":
                    accepted_simulations.append(res)
                elif res["status"] == "DISCARDED_TOO_BIG":
                    discarded_count += 1
                else:
                    failed_count += 1

    # Persist the audit manifest after the phase. Accepted cluster data
    # has already been written separately, immediately by each worker.
    manifest_df = pd.DataFrame(list(manifest_records.values()))
    if not manifest_df.empty:
        manifest_df.to_csv(manifest_file, index=False)

    print("\n>>> PHASE 1 REPORT <<<")
    print(f"New simulation tasks:  {len(simulation_tasks)}")
    print(f"Saved datasets reused: {reused_count}")
    print(f"Accepted datasets:     {len(accepted_simulations)}")
    print(f"Newly discarded:       {discarded_count}")
    print(f"Newly failed:          {failed_count}")
    print(f"Saved data location:   {simulation_dir}")

    if not accepted_simulations:
        print("No accepted simulations available for inference. Exiting.")
        return

    # --------------------------------------------------------
    # PHASE 2: Paired inference, loading clusters from disk.
    # --------------------------------------------------------
    print("\n>>> PHASE 2: PAIRED INFERENCE <<<")
    print("Both likelihoods use the same saved cluster dataset.")

    columns = [
        "Simulation_ID",
        "Model",
        "R_true",
        "k_true",
        "nu",
        "Replicate",
        "Success",
        "Message",
        "R_est",
        "k_est",
        "N_Clusters",
        "Max_Cluster",
        "Sim_Time_Sec",
        "Opt_Time_Sec",
        "Steps",
    ]

    # Create new files with headers, or resume files that already have IDs.
    for result_file in (ours_file, tran_kiem_file):
        if not result_file.exists():
            pd.DataFrame(columns=columns).to_csv(result_file, index=False)

    ours_done = completed_simulation_ids(ours_file)
    tran_done = completed_simulation_ids(tran_kiem_file)

    inference_tasks = []
    for metadata in accepted_simulations:
        simulation_id = metadata["Simulation_ID"]
        run_ours = simulation_id not in ours_done
        run_tran_kiem = simulation_id not in tran_done

        if not run_ours and not run_tran_kiem:
            continue

        inference_tasks.append((
            simulation_id,
            metadata["data_file"],
            metadata,
            run_ours,
            run_tran_kiem,
            s_fixed,
            max_steps,
            ftol,
            bounds,
        ))

    print(f"Datasets requiring inference: {len(inference_tasks)}")
    print(f"Original-model fits already saved: {len(ours_done)}")
    print(f"Tran-Kiem fits already saved:      {len(tran_done)}")

    load_errors = 0
    if inference_tasks:
        with ctx.Pool(num_cores) as pool:
            for res in tqdm(
                pool.imap_unordered(inference_worker, inference_tasks),
                total=len(inference_tasks),
            ):
                if res.get("load_error"):
                    load_errors += 1
                    print(
                        f"Could not load {res['Simulation_ID']}: "
                        f"{res['load_error']}"
                    )
                    continue

                if res["ours"] is not None:
                    append_result(ours_file, res["ours"], columns)

                if res["tran_kiem"] is not None:
                    append_result(tran_kiem_file, res["tran_kiem"], columns)

    print("\n========================================")
    print("PAIRED MODEL COMPARISON COMPLETE")
    print("========================================")
    print(f"Simulation data: {simulation_dir}")
    print(f"Simulation manifest: {manifest_file}")
    print(f"Original model results: {ours_file}")
    print(f"Tran-Kiem results: {tran_kiem_file}")
    print(f"Dataset loading errors: {load_errors}")
    print("Saved simulations can be reused without simulating again.")


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_name",
        help="Experiment name, e.g. experiment_1",
    )
    args = parser.parse_args()
    run_experiment(args.experiment_name)
