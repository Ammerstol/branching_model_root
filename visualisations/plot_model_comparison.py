
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch

from simulation_and_inference.experiment_config import load_experiment


# ============================================================
# CONFIGURATION
# ============================================================

COLOR_MAP_R = {
    1.43: "#1f77b4",
    2.5: "#d62728"
}

COLOR_MAP_K = {
    0.1: "#2ca02c",
    0.5: "#9467bd"
}

MODEL_FILES = {
    "Ours": "two_stage_results.csv",
    "Tran-Kiem": "two_stage_results_tran_kiem.csv"
}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,
})


# ============================================================
# LOAD RESULTS
# ============================================================

def load_model_results(filename, model_name):

    if not filename.exists():
        raise FileNotFoundError(
            f"{model_name} results not found:\n{filename}\n\n"
            f"Expected one results CSV for each model."
        )

    df = pd.read_csv(filename)

    required_columns = [
        "R_true",
        "k_true",
        "nu",
        "R_est",
        "k_est",
        "Success"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{model_name} results are missing columns: {missing}"
        )

    total = len(df)

    # Retain successful optimisation runs, as in the original figures.
    success = (
        df["Success"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )

    df = df[success].copy()

    numerical_columns = [
        "R_true",
        "k_true",
        "nu",
        "R_est",
        "k_est"
    ]

    for col in numerical_columns:
        if col == "R_true":
            df[col] = df[col].replace("10/7", str(10 / 7))

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numerical_columns)

    for col in ["R_true", "k_true", "nu"]:
        df[col] = df[col].round(4)

    df["Model"] = model_name

    print(
        f"{model_name}: {len(df)} successful fits "
        f"out of {total}."
    )

    return df


# ============================================================
# COLOUR PALETTES
# ============================================================

def get_palette(values, hue_col):

    values = sorted(values)

    if hue_col == "R_true":
        original_map = COLOR_MAP_R
    else:
        original_map = COLOR_MAP_K

    fallback = sns.color_palette(
        "colorblind",
        n_colors=max(len(values), 1)
    ).as_hex()

    palette = {}

    for i, value in enumerate(values):

        rounded_value = round(float(value), 2)

        if rounded_value in original_map:
            palette[value] = original_map[rounded_value]
        else:
            palette[value] = fallback[i]

    return palette


# ============================================================
# ORIGINAL BOX PLOT STYLE, EXTENDED TO COMPARE MODELS
# ============================================================

def plot_model_comparison(
    df,
    param,
    output_file
):

    if param == "R":

        true_col = "R_true"
        est_col = "R_est"
        hue_col = "k_true"

        y_label = r"Estimated $R$"
        legend_title = r"True $\kappa$"

    else:

        true_col = "k_true"
        est_col = "k_est"
        hue_col = "R_true"

        y_label = r"Estimated $\kappa$"
        legend_title = r"True $R$"

    models = ["Ours", "Tran-Kiem"]

    unique_vals = sorted(df[true_col].unique())
    hue_values = sorted(df[hue_col].unique())
    nu_values = sorted(df["nu"].unique())

    if len(unique_vals) == 0:
        print(f"No values available for parameter {param}.")
        return

    color_map = get_palette(hue_values, hue_col)

    # Two model rows, one column for each true parameter value.
    fig, axes = plt.subplots(
        2,
        len(unique_vals),
        figsize=(
            max(7.2, 2.0 * len(unique_vals)),
            5.6
        ),
        sharey=False,
        squeeze=False
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.11,
        top=0.82,
        wspace=0.26,
        hspace=0.35
    )

    for col, val in enumerate(unique_vals):

        # Calculate common y-axis limits for both models.
        # This makes the comparison fair and visually direct.
        pooled = df[
            np.isclose(df[true_col], val)
        ][est_col].dropna()

        if pooled.empty:
            continue

        q_low = pooled.quantile(0.05)
        q_high = pooled.quantile(0.95)

        padding = max(abs(val) * 0.2, 0.05)

        y_min = max(0, q_low - padding)
        y_max = q_high + padding

        # Ensure the true value remains visible.
        y_min = min(y_min, val)
        y_max = max(y_max, val)

        for row, model in enumerate(models):

            ax = axes[row, col]

            subset = df[
                (df["Model"] == model)
                & np.isclose(df[true_col], val)
            ]

            if subset.empty:

                ax.text(
                    0.5,
                    0.5,
                    "No successful fits",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8
                )

            else:

                sns.boxplot(
                    data=subset,
                    x="nu",
                    y=est_col,
                    hue=hue_col,
                    hue_order=hue_values,
                    order=nu_values,
                    palette=color_map,
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

            # Remove individual legends. We use one shared legend.
            legend = ax.get_legend()

            if legend is not None:
                legend.remove()

            # True parameter reference line.
            ax.axhline(
                val,
                color="0.55",
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
            )

            ax.set_ylim(y_min, y_max)

            # Facet title.
            if row == 0:
                ax.set_title(
                    f"True {param} = {val:g}",
                    fontsize=9,
                    pad=6
                )

            ax.set_xlabel(r"Mutation probability $\nu$", fontsize=8)

            if col == 0:
                ax.set_ylabel(y_label, fontsize=9)

                # Explicit model label along the left side.
                ax.annotate(
                    model,
                    xy=(-0.30, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold"
                )
            else:
                ax.set_ylabel("")

            # Original formatting.
            ax.tick_params(
                axis="both",
                labelsize=8,
                width=0.5,
                length=3
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

    # --------------------------------------------------------
    # Shared legend: same colours as the original figures
    # --------------------------------------------------------

    handles = [
        Patch(
            facecolor=color_map[value],
            edgecolor="0.35",
            label=f"{value:g}"
        )
        for value in hue_values
    ]

    fig.legend(
        handles=handles,
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(handles),
        fontsize=8,
        title_fontsize=8,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.8",
    )

    fig.suptitle(
        f"Model comparison: estimated {param}",
        fontsize=11,
        y=0.995
    )

    # Save in a separate folder; original figures remain untouched.
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_file,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# MAIN
# ============================================================

def run_visualisations(experiment_name, ours_filename, tran_filename):

    config, output_dir = load_experiment(experiment_name)

    ours_path = output_dir / ours_filename
    tran_path = output_dir / tran_filename

    # Load the two models independently.
    ours_df = load_model_results(
        ours_path,
        "Ours"
    )

    tran_df = load_model_results(
        tran_path,
        "Tran-Kiem"
    )

    # Combine results without modifying either source CSV.
    combined_df = pd.concat(
        [ours_df, tran_df],
        ignore_index=True
    )

    # Keep the original figure directories untouched.
    comparison_dir = (
        output_dir / "visualisations" / "model_comparison"
    )

    # Estimated R.
    plot_model_comparison(
        combined_df,
        param="R",
        output_file=comparison_dir / "R_model_comparison.pdf"
    )

    # Estimated kappa.
    plot_model_comparison(
        combined_df,
        param="k",
        output_file=comparison_dir / "kappa_model_comparison.pdf"
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare original and Tran-Kiem inference results."
    )

    parser.add_argument(
        "experiment_name",
        help="Experiment name, e.g. experiment_1"
    )

    parser.add_argument(
        "--ours-file",
        default="two_stage_results_ours.csv"
    )

    parser.add_argument(
        "--tran-kiem-file",
        default="two_stage_results_tran_kiem.csv"
    )

    args = parser.parse_args()

    run_visualisations(
        args.experiment_name,
        args.ours_file,
        args.tran_kiem_file
    )
