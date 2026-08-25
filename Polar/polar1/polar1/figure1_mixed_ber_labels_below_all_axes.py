from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Figure 1: Mixed-channel BER curves (2x2)
#
# Changes:
#   1) Move subplot labels UNDER each graph:
#        (a) AWGN
#        (b) Rayleigh
#        (c) Rician LOS
#        (d) Impulsive
#
#   2) Every subplot has its own:
#        x-axis = Eb/N0 (dB)
#        y-axis = BER
#
#   3) Every subplot shows its own x/y tick labels.
#
#   4) Move common legend to TOP-LEFT.
#
#   5) Rename:
#        "Proposed CA-CNN-RNND"
#        -> "Proposed"
#
# Input:
#   results_mean_std_by_snr.csv
#
# Output:
#   Figure1_Mixed_BER_2x2.png
#   Figure1_Mixed_BER_2x2.pdf
# ============================================================


# ------------------------------------------------------------
# Find results CSV automatically
# ------------------------------------------------------------
def find_results_csv():

    matches = sorted(
        Path(".").rglob("results_mean_std_by_snr.csv")
    )

    if not matches:
        raise FileNotFoundError(
            "Cannot find 'results_mean_std_by_snr.csv'.\n"
            "Please extract the FINAL LOCKED result ZIP first,\n"
            "then run this script again."
        )

    # Prefer a file inside a folder named "tables"
    preferred = [
        p for p in matches
        if p.parent.name.lower() == "tables"
    ]

    return preferred[0] if preferred else matches[0]


# ------------------------------------------------------------
# Load result CSV
# ------------------------------------------------------------
CSV_PATH = find_results_csv()

print("Using result file:")
print(CSV_PATH.resolve())

df = pd.read_csv(CSV_PATH)


# ------------------------------------------------------------
# Keep MIXED-channel experiment only
# ------------------------------------------------------------
mixed = df[
    (df["regime"] == "mixed")
    & (df["job_tag"] == "MIXED")
].copy()

if mixed.empty:
    raise ValueError(
        "No mixed-channel rows were found in the CSV file."
    )


# ------------------------------------------------------------
# Channel order
# ------------------------------------------------------------
channels = [
    "AWGN",
    "Rayleigh",
    "RicianLOS",
    "Impulsive",
]


# ------------------------------------------------------------
# Panel labels
# ------------------------------------------------------------
panel_titles = {
    "AWGN": "(a) AWGN",
    "Rayleigh": "(b) Rayleigh",
    "RicianLOS": "(c) Rician LOS",
    "Impulsive": "(d) Impulsive",
}


# ------------------------------------------------------------
# Method order
# ------------------------------------------------------------
method_order = [
    "SC",
    "ML",
    "PNND",
    "RNND_Zhu",
    "CNN_RNND",
    "Proposed_CA_CNN_RNND",
]


# ------------------------------------------------------------
# Display names
# ------------------------------------------------------------
method_labels = {
    "SC": "SC",
    "ML": "ML",
    "PNND": "PNND",
    "RNND_Zhu": "RNND-Zhu",
    "CNN_RNND": "CNN-RNND",

    # Changed from:
    # "Proposed CA-CNN-RNND"
    "Proposed_CA_CNN_RNND": "Proposed",
}


# ------------------------------------------------------------
# Plot styles
# ------------------------------------------------------------
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


# ============================================================
# Journal-style typography
# ============================================================
plt.rcParams.update({

    "font.family": "serif",

    "font.size": 9,

    "axes.labelsize": 9,

    "legend.fontsize": 8,

    "xtick.labelsize": 8,

    "ytick.labelsize": 8,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,
})


# ============================================================
# Create 2x2 figure
# ============================================================
#
# sharex=False and sharey=False are used so that EVERY graph
# shows its own x-axis and y-axis labels/tick labels.
#
fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.8, 6.4),
    sharex=False,
    sharey=False,
)

axes = axes.ravel()


# ------------------------------------------------------------
# Legend storage
# ------------------------------------------------------------
legend_handles = None
legend_labels = None


# ============================================================
# Plot each channel
# ============================================================
for ax, channel in zip(axes, channels):

    sub = mixed[
        mixed["channel"] == channel
    ].copy()

    if sub.empty:
        raise ValueError(
            f"No rows found for channel '{channel}'."
        )


    # --------------------------------------------------------
    # Plot all methods
    # --------------------------------------------------------
    for method in method_order:

        g = (
            sub[
                sub["method"] == method
            ]
            .sort_values("snr_db")
        )

        if g.empty:

            print(
                f"Warning: method '{method}' "
                f"not found for channel '{channel}'."
            )

            continue


        ax.semilogy(
            g["snr_db"],
            g["BER_mean"],
            label=method_labels[method],
            **styles[method],
        )


    # --------------------------------------------------------
    # Axis limits
    # --------------------------------------------------------
    ax.set_xlim(
        -0.15,
        7.15,
    )

    ax.set_ylim(
        1e-5,
        4e-1,
    )


    # --------------------------------------------------------
    # X ticks
    # --------------------------------------------------------
    ax.set_xticks(
        range(8)
    )


    # ========================================================
    # Every graph has x-axis = Eb/N0
    # ========================================================
    ax.set_xlabel(
        r"$E_b/N_0$ (dB)",
        labelpad=3,
    )


    # ========================================================
    # Every graph has y-axis = BER
    # ========================================================
    ax.set_ylabel(
        "BER",
        labelpad=4,
    )


    # ========================================================
    # Move panel label UNDER each graph
    # ========================================================
    #
    # Change -0.28 if you want the text higher/lower:
    #
    #   -0.24 = closer to graph
    #   -0.28 = current
    #   -0.32 = farther below
    #
    ax.text(
        0.5,
        -0.28,
        panel_titles[channel],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        clip_on=False,
    )


    # --------------------------------------------------------
    # Major grid
    # --------------------------------------------------------
    ax.grid(
        True,
        which="major",
        linewidth=0.45,
        alpha=0.35,
    )


    # --------------------------------------------------------
    # Minor grid
    # --------------------------------------------------------
    ax.grid(
        True,
        which="minor",
        linewidth=0.25,
        alpha=0.18,
    )


    # --------------------------------------------------------
    # Save legend handles only once
    # --------------------------------------------------------
    if legend_handles is None:

        legend_handles, legend_labels = (
            ax.get_legend_handles_labels()
        )


# ============================================================
# Common legend at TOP-LEFT
# ============================================================
#
# Original:
#
#   loc="upper center"
#   bbox_to_anchor=(0.5, 0.995)
#
# New:
#
#   loc="upper left"
#   bbox_to_anchor=(0.095, 0.995)
#
fig.legend(
    legend_handles,
    legend_labels,
    loc="upper left",
    bbox_to_anchor=(0.095, 0.995),
    ncol=3,
    frameon=False,
    fontsize=11,
)

# ============================================================
# Layout
# ============================================================
#
# Extra vertical space is required because:
#
#   - Every graph has Eb/N0 below the x-axis
#   - Panel label is also below each graph
#
fig.subplots_adjust(

    left=0.10,

    right=0.985,

    top=0.86,

    bottom=0.11,

    wspace=0.25,

    hspace=0.62,
)


# ============================================================
# Save figure
# ============================================================
png_path = Path(
    "Figure1_Mixed_BER_2x2.png"
)

pdf_path = Path(
    "Figure1_Mixed_BER_2x2.pdf"
)


# ------------------------------------------------------------
# Save PNG
# ------------------------------------------------------------
fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.05,
)


# ------------------------------------------------------------
# Save PDF
# ------------------------------------------------------------
fig.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.05,
)


# ============================================================
# Show figure
# ============================================================
plt.show()


# ============================================================
# Print saved files
# ============================================================
print("\nSaved files:")

print(
    png_path.resolve()
)

print(
    pdf_path.resolve()
)