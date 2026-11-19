import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from simulation_and_inference.experiment_config import load_experiment


# ============================================================
# ORIGINAL VISUALISATION STYLE
# ============================================================

# Keep the colours from the original figures.
COLOR_MAP_R = {1.43: "#1f77b4", 2.5: "#d62728"}
COLOR_MAP_K = {0.1: "#2ca02c", 0.5: "#9467bd"}

# Fonts compatible with the surrounding LaTeX text.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,
})


# ============================================================
# LOAD ORIGINAL-MODEL RESULTS
# ============================================================

def load_and_fix_data(filename):
    """Load the original-model results and retain successful fits."""

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Results file not found: {filename}")

    df = pd.read_csv(filename)

    required_columns = [
        "R_true", "k_true", "nu", "R_est", "k_est", "Success"
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {filename}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # CSVs produced by the paired inference script contain a Model column.
    # If so, keep only the original model.
    if "Model" in df.columns:
        df = df[df["Model"].astype(str).str.strip().str.lower().isin(
            ["ours", "original", "original model"]
        )].copy()

    df["Success"] = (
        df["Success"].astype(str).str.strip().str.lower().isin(["true", "1"])
    )
    df_clean = df[df["Success"]].copy()

    numerical_columns = ["R_true", "k_true", "nu", "R_est", "k_est"]
    for column in numerical_columns:
        # Also tolerate a legacy R value written as "10/7".
        if column == "R_true":
            df_clean[column] = df_clean[column].replace({"10/7": str(10 / 7)})
        df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")

    df_clean = df_clean.dropna(subset=numerical_columns)

    # Preserve the grouping precision used in the original figures.
    df_clean["R_true"] = df_clean["R_true"].round(2)
    df_clean["k_true"] = df_clean["k_true"].round(2)
    df_clean["nu"] = df_clean["nu"].round(4)

    if df_clean.empty:
        raise ValueError(
            f"No successful original-model estimates found in {filename}."
        )

    print(
        f"Loaded {len(df_clean)} successful original-model fits "
        f"out of {len(df)} rows from {filename}."
    )
    return df_clean


# ============================================================
# ORIGINAL BOX PLOTS -- RESTORED
# ============================================================

def plot_raw_estimates(df, param, true_col, est_col, color_map, output_dir):
    """Reproduce the original one-row, faceted boxplot design."""

    unique_vals = sorted(df[true_col].unique())
    if len(unique_vals) == 0:
        return

    fig, axes = plt.subplots(
        1,
        len(unique_vals),
        figsize=(7.2, 2.8),
        sharey=False,
        layout="none",
        squeeze=False,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.19,
        top=0.82,
        wspace=0.26,
    )

    hue_col = "k_true" if param == "R" else "R_true"
    legend_title = r"True $\kappa$" if param == "R" else r"True $R$"

    # Keep the original colours for the parameter values for which
    # they were defined, and add distinct colours for any new values
    # in the experiment configuration (e.g. kappa=1.0).
    base_palette = {
        round(float(key), 2): color
        for key, color in color_map.items()
    }
    hue_values = sorted(df[hue_col].dropna().unique())
    palette = {}
    missing_values = []

    for hue_value in hue_values:
        key = round(float(hue_value), 2)
        if key in base_palette:
            palette[hue_value] = base_palette[key]
        else:
            missing_values.append(hue_value)

    if missing_values:
        extra_colors = sns.color_palette(
            "husl", n_colors=len(missing_values)
        ).as_hex()
        for hue_value, color in zip(missing_values, extra_colors):
            palette[hue_value] = color

    for ax, val in zip(axes.ravel(), unique_vals):
        subset = df[df[true_col] == val]

        sns.boxplot(
            data=subset,
            x="nu",
            y=est_col,
            hue=hue_col,
            palette=palette,
            width=0.55,
            gap=0.15,
            dodge=True,
            linewidth=0.55,
            boxprops={
                "edgecolor": "0.35",
                "linewidth": 0.55,
            },
            whiskerprops={
                "color": "0.45",
                "linewidth": 0.6,
            },
            capprops={
                "color": "0.45",
                "linewidth": 0.6,
            },
            medianprops={
                "color": "0.25",
                "linewidth": 1.25,
            },
            flierprops={
                "marker": "o",
                "markersize": 2,
                "markerfacecolor": "none",
                "markeredgecolor": "0.45",
                "markeredgewidth": 0.45,
                "alpha": 0.6,
            },
            ax=ax,
        )

        ax.axhline(
            val,
            color="0.55",
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
            label="_nolegend_",
        )

        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles,
            labels,
            title=legend_title,
            loc="upper left",
            fontsize=7,
            title_fontsize=7,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.8",
            borderpad=0.35,
            labelspacing=0.25,
        )
        legend.get_frame().set_linewidth(0.5)

        q_low = subset[est_col].quantile(0.05)
        q_high = subset[est_col].quantile(0.95)
        ax.set_ylim(
            max(0, q_low - val * 0.2),
            q_high + val * 0.2,
        )

        ax.tick_params(
            axis="both", labelsize=8, width=0.5, length=3
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("0.35")

        ax.set_axisbelow(True)
        ax.grid(False)
        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.2,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"{param}_overpic_v3.pdf"
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(filename, format="pdf", bbox_inches=None)

    plt.close(fig)
    print(f"Saved: {filename}")


# ============================================================
# MAIN
# ============================================================

def run_visualisations(experiment_name):
    _, output_dir = load_experiment(experiment_name)

    # Paired inference writes the original-model estimates here.
    # Fall back to the legacy filename for older one-model runs.
    ours_file = output_dir / "two_stage_results_ours.csv"
    legacy_file = output_dir / "two_stage_results.csv"

    if ours_file.exists():
        input_file = ours_file
    elif legacy_file.exists():
        input_file = legacy_file
    else:
        raise FileNotFoundError(
            "Could not find original-model results. Looked for:\n"
            f"  {ours_file}\n"
            f"  {legacy_file}"
        )

    df = load_and_fix_data(input_file)

    # Keep these figures separate from the model-comparison figures.
    output_dir = output_dir / "visualisations" / "original_style"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_raw_estimates(
        df,
        param="R",
        true_col="R_true",
        est_col="R_est",
        color_map=COLOR_MAP_K,
        output_dir=output_dir,
    )

    plot_raw_estimates(
        df,
        param="k",
        true_col="k_true",
        est_col="k_est",
        color_map=COLOR_MAP_R,
        output_dir=output_dir,
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recreate the original parameter-estimate boxplots."
    )
    parser.add_argument(
        "experiment_name",
        help="Experiment name, e.g. experiment_1"
    )
    args = parser.parse_args()

    run_visualisations(args.experiment_name)
