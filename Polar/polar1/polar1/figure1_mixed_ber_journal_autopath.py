from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Figure 1: Mixed-channel BER curves (2x2)
# This version automatically searches for:
#   results_mean_std_by_snr.csv
# in the current folder and all subfolders.
# ------------------------------------------------------------

def find_results_csv():
    matches = sorted(Path(".").rglob("results_mean_std_by_snr.csv"))

    if not matches:
        raise FileNotFoundError(
            "Cannot find 'results_mean_std_by_snr.csv'.\n"
            "Please extract the FINAL LOCKED result ZIP first,\n"
            "then run this script again."
        )

    # Prefer a path inside a 'tables' directory if available
    preferred = [p for p in matches if p.parent.name == "tables"]
    return preferred[0] if preferred else matches[0]


CSV_PATH = find_results_csv()

print("Using result file:")
print(CSV_PATH.resolve())

df = pd.read_csv(CSV_PATH)

# Mixed-channel experiment only
mixed = df[
    (df["regime"] == "mixed")
    & (df["job_tag"] == "MIXED")
].copy()

if mixed.empty:
    raise ValueError(
        "No mixed-channel rows were found in the CSV file."
    )

channels = [
    "AWGN",
    "Rayleigh",
    "RicianLOS",
    "Impulsive",
]

panel_titles = {
    "AWGN": "(a) AWGN",
    "Rayleigh": "(b) Rayleigh",
    "RicianLOS": "(c) Rician LOS",
    "Impulsive": "(d) Impulsive",
}

method_order = [
    "SC",
    "ML",
    "PNND",
    "RNND_Zhu",
    "CNN_RNND",
    "Proposed_CA_CNN_RNND",
]

method_labels = {
    "SC": "SC",
    "ML": "ML",
    "PNND": "PNND",
    "RNND_Zhu": "RNND-Zhu",
    "CNN_RNND": "CNN-RNND",
    "Proposed_CA_CNN_RNND": "Proposed CA-CNN-RNND",
}

styles = {
    "SC": dict(
        marker="o",
        linestyle="--",
        linewidth=1.25,
        markersize=4,
    ),
    "ML": dict(
        marker="s",
        linestyle=":",
        linewidth=1.35,
        markersize=4,
    ),
    "PNND": dict(
        marker="^",
        linestyle="-.",
        linewidth=1.25,
        markersize=4,
    ),
    "RNND_Zhu": dict(
        marker="D",
        linestyle="--",
        linewidth=1.35,
        markersize=3.8,
    ),
    "CNN_RNND": dict(
        marker="v",
        linestyle="-.",
        linewidth=1.35,
        markersize=4,
    ),
    "Proposed_CA_CNN_RNND": dict(
        marker="*",
        linestyle="-",
        linewidth=2.2,
        markersize=7,
    ),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.2, 5.8),
    sharex=True,
    sharey=True,
)

axes = axes.ravel()

legend_handles = None
legend_labels = None

for ax, channel in zip(axes, channels):
    sub = mixed[mixed["channel"] == channel].copy()

    if sub.empty:
        raise ValueError(
            f"No rows found for channel '{channel}'."
        )

    for method in method_order:
        g = sub[sub["method"] == method].sort_values("snr_db")

        if g.empty:
            print(f"Warning: method '{method}' not found for channel '{channel}'.")
            continue

        ax.semilogy(
            g["snr_db"],
            g["BER_mean"],
            label=method_labels[method],
            **styles[method],
        )

    ax.set_title(panel_titles[channel])
    ax.set_xlim(-0.15, 7.15)
    ax.set_ylim(1e-5, 4e-1)
    ax.set_xticks(range(8))

    ax.grid(
        True,
        which="major",
        linewidth=0.45,
        alpha=0.35,
    )

    ax.grid(
        True,
        which="minor",
        linewidth=0.25,
        alpha=0.18,
    )

    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

fig.supxlabel(
    r"$E_b/N_0$ (dB)",
    y=0.055,
)

fig.supylabel(
    "Bit Error Rate (BER)",
    x=0.045,
)

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    frameon=False,
)

fig.tight_layout(
    rect=[0.06, 0.07, 1.0, 0.89]
)

png_path = Path("Figure1_Mixed_BER_2x2.png")
pdf_path = Path("Figure1_Mixed_BER_2x2.pdf")

plt.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.show()

print("\nSaved files:")
print(png_path.resolve())
print(pdf_path.resolve())
