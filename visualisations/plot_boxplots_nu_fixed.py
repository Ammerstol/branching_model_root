import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# --- CONFIGURATION ---
S = "1"
NU_FIXED = "0.25"
INPUT_FILE = (
    f"experiment_two_stage/"
    f"experiment_two_stage_size100000_S{S}_nu{NU_FIXED}.csv"
)
OUTPUT_DIR = f"plots_relative_error_S{S}_nu{NU_FIXED}"

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

    df_clean["R_est"] = pd.to_numeric(
        df_clean["R_est"], errors="coerce"
    )
    df_clean["k_est"] = pd.to_numeric(
        df_clean["k_est"], errors="coerce"
    )
    df_clean = df_clean.dropna(subset=["R_est", "k_est"])

    df_clean["R_true"] = df_clean["R_true"].round(3)
    df_clean["k_true"] = df_clean["k_true"].round(3)

    df_clean["R_rel_err"] = (
        (df_clean["R_est"] - df_clean["R_true"])
        / df_clean["R_true"]
    )
    df_clean["k_rel_err"] = (
        (df_clean["k_est"] - df_clean["k_true"])
        / df_clean["k_true"]
    )

    return df_clean


def plot_single_frame_relative_error(df, output_dir):
    if df.empty:
        print("No successful estimates available.")
        return

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.2, 2.8),
        layout="none",
    )

    # Fixed geometry matching the overpic coordinates below.
    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.19,
        top=0.82,
        wspace=0.26,
    )

    panels = [
    ("R_true", "R_rel_err", "k_true", "Set2", r"True $\kappa$"),
    ("k_true", "k_rel_err", "R_true", "Set2", r"True $R$"),
]

    for ax, (x_col, y_col, hue_col, palette, legend_title) in zip(
        axes, panels
    ):
        sns.boxplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=hue_col,
            palette=palette,
            width=0.6,
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
                "color": "0.15",
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
            0,
            color="0.55",
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
            label="_nolegend_",
        )

        # Supplied by LaTeX overpic.
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

        # Preserve the original quantile-based viewing window.
        q_low, q_high = df[y_col].quantile([0.005, 0.995])
        margin = (q_high - q_low) * 0.1
        if margin == 0:
            margin = max(abs(q_low) * 0.1, 0.01)

        ax.set_ylim(q_low - margin, q_high + margin)

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

    filename = os.path.join(
        output_dir, "relative_errors_overpic.pdf"
    )

    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(filename, format="pdf", bbox_inches=None)

    plt.close(fig)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_fix_data(INPUT_FILE)

    if df is not None:
        plot_single_frame_relative_error(df, OUTPUT_DIR)