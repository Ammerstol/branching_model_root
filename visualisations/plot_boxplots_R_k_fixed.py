import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# --- CONFIGURATION ---
S = "1"
INPUT_FILE = f"experiment_two_stage/experiment_two_stage_size100000_S{S}.csv"
OUTPUT_DIR = f"plots_raw_estimates_S{S}"

COLOR_MAP_R = {1.43: "#1f77b4", 2.5: "#d62728"}
COLOR_MAP_K = {0.1: "#2ca02c", 0.5: "#9467bd"}

# Fonts compatible with the surrounding LaTeX text.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,
})


def load_and_fix_data(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None

    col_names = [
        "R_true", "k_true", "nu", "Replicate",
        "Success", "Message", "R_est", "k_est",
        "N_Clusters", "Max_Cluster", "Required_K",
        "Sim_Time_Sec", "Opt_Time_Sec", "Steps",
    ]

    df = pd.read_csv(filename, names=col_names, header=0)

    df["Success"] = (
        df["Success"].astype(str).str.strip().str.lower() == "true"
    )
    df_clean = df[df["Success"]].copy()

    df_clean["R_true"] = df_clean["R_true"].round(2)
    df_clean["k_true"] = df_clean["k_true"].round(2)

    df_clean["R_est"] = pd.to_numeric(
        df_clean["R_est"], errors="coerce"
    )
    df_clean["k_est"] = pd.to_numeric(
        df_clean["k_est"], errors="coerce"
    )

    return df_clean.dropna(subset=["R_est", "k_est"])


def plot_raw_estimates(
    df, param, true_col, est_col, color_map, output_dir
):
    unique_vals = sorted(df[true_col].unique())
    if not unique_vals:
        return

    fig, axes = plt.subplots(
        1,
        len(unique_vals),
        figsize=(7.2, 2.8),  # Wider, flatter panels
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

    for ax, val in zip(axes.ravel(), unique_vals):
        subset = df[df[true_col] == val]

        sns.boxplot(
            data=subset,
            x="nu",
            y=est_col,
            hue=hue_col,
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

    filename = os.path.join(output_dir, f"{param}_overpic_v3.pdf")
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(filename, format="pdf", bbox_inches=None)

    plt.close(fig)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_fix_data(INPUT_FILE)

    if df is not None:
        plot_raw_estimates(
            df,
            param="R",
            true_col="R_true",
            est_col="R_est",
            color_map=COLOR_MAP_K,
            output_dir=OUTPUT_DIR,
        )

        plot_raw_estimates(
            df,
            param="k",
            true_col="k_true",
            est_col="k_est",
            color_map=COLOR_MAP_R,
            output_dir=OUTPUT_DIR,
        )