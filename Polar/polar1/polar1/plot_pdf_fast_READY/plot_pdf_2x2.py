from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Combined PDF figure: 2x2
#
# (a) AWGN
# (b) Rayleigh
# (c) Rician LOS
# (d) Impulsive
#
# FAST VERSION:
# - NO training
# - NO checkpoint loading
# - Reads saved .npz files directly
# ============================================================


# ------------------------------------------------------------
# Current folder
# ------------------------------------------------------------
BASE = Path(__file__).resolve().parent


# ------------------------------------------------------------
# Input NPZ files
# ------------------------------------------------------------
FILES = {
    "AWGN": BASE / "PDF_Proposed_AWGN_3dB.npz",

    "Rayleigh": BASE / "PDF_Proposed_Rayleigh_3dB.npz",

    "Rician LOS": BASE / "PDF_Proposed_RicianLOS_3dB.npz",

    "Impulsive": BASE / "PDF_Proposed_Impulsive_3dB.npz",
}


# ============================================================
# SETTINGS
# ============================================================

N_BINS = 180

# Figure size
FIGSIZE = (9.0, 7.2)

# Font sizes
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 15
PANEL_FONTSIZE = 15

# Transparency
FILL_ALPHA = 0.42

# Y-label position
# Make "Probability density" aligned in every subplot
Y_LABEL_X = -0.13
Y_LABEL_Y = 0.50


# ============================================================
# Journal-style typography
# ============================================================
plt.rcParams.update({
    "font.family": "serif",

    "font.size": 13,

    "axes.labelsize": AXIS_LABEL_FONTSIZE,

    "xtick.labelsize": TICK_FONTSIZE,

    "ytick.labelsize": TICK_FONTSIZE,

    "legend.fontsize": LEGEND_FONTSIZE,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,
})


# ============================================================
# Function: read NPZ and calculate probability density
# ============================================================
def get_density(npz_path):

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------
    if not npz_path.exists():

        raise FileNotFoundError(
            f"\nMissing file:\n{npz_path}\n"
        )


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    data = np.load(npz_path)

    received = data["received"].ravel()

    denoised = data["denoised"].ravel()


    # --------------------------------------------------------
    # Determine plotting range
    # --------------------------------------------------------
    lo = min(
        -5.0,

        float(
            np.quantile(
                received,
                0.001,
            )
        ),

        float(
            np.quantile(
                denoised,
                0.001,
            )
        ),
    )


    hi = max(
        5.0,

        float(
            np.quantile(
                received,
                0.999,
            )
        ),

        float(
            np.quantile(
                denoised,
                0.999,
            )
        ),
    )


    # --------------------------------------------------------
    # Histogram bins
    # --------------------------------------------------------
    bins = np.linspace(
        lo,
        hi,
        N_BINS,
    )


    # --------------------------------------------------------
    # Received density
    # --------------------------------------------------------
    h_received, edges = np.histogram(
        received,
        bins=bins,
        density=True,
    )


    # --------------------------------------------------------
    # Denoised density
    # --------------------------------------------------------
    h_denoised, _ = np.histogram(
        denoised,
        bins=bins,
        density=True,
    )


    # --------------------------------------------------------
    # Bin centers
    # --------------------------------------------------------
    centers = (
        0.5
        * (
            edges[:-1]
            + edges[1:]
        )
    )


    return (
        centers,
        h_received,
        h_denoised,
    )


# ============================================================
# Channel order
# ============================================================
channels = [
    "AWGN",
    "Rayleigh",
    "Rician LOS",
    "Impulsive",
]


# ============================================================
# Panel labels
# ============================================================
panel_labels = {
    "AWGN": "(a) AWGN",

    "Rayleigh": "(b) Rayleigh",

    "Rician LOS": "(c) Rician LOS",

    "Impulsive": "(d) Impulsive",
}


# ============================================================
# Create 2 x 2 figure
# ============================================================
fig, axes = plt.subplots(
    2,
    2,

    figsize=FIGSIZE,

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
for ax, channel in zip(
    axes,
    channels,
):


    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------
    centers, h_received, h_denoised = (
        get_density(
            FILES[channel]
        )
    )


    # ========================================================
    # Received symbols
    # ========================================================
    ax.fill_between(
        centers,

        h_received,

        alpha=FILL_ALPHA,

        label="Received symbols",
    )


    # ========================================================
    # Denoised symbols
    # ========================================================
    ax.fill_between(
        centers,

        h_denoised,

        alpha=FILL_ALPHA,

        label="Denoised symbols",
    )


    # ========================================================
    # X-axis label
    # ========================================================
    ax.set_xlabel(
        "Symbol value",

        fontsize=AXIS_LABEL_FONTSIZE,

        labelpad=5,
    )


    # ========================================================
    # Y-axis label
    # ========================================================
    ax.set_ylabel(
        "Probability density",

        fontsize=AXIS_LABEL_FONTSIZE,
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Force all y-axis labels to exactly the same location
    # --------------------------------------------------------
    ax.yaxis.set_label_coords(
        Y_LABEL_X,
        Y_LABEL_Y,
    )


    # ========================================================
    # Tick size
    # ========================================================
    ax.tick_params(
        axis="both",

        which="major",

        labelsize=TICK_FONTSIZE,
    )


    # ========================================================
    # Grid
    # ========================================================
    ax.grid(
        True,

        which="major",

        alpha=0.20,

        linewidth=0.6,
    )


    # ========================================================
    # Panel label below graph
    # ========================================================
    ax.text(
        0.5,

        -0.27,

        panel_labels[channel],

        transform=ax.transAxes,

        ha="center",

        va="top",

        fontsize=PANEL_FONTSIZE,

        clip_on=False,
    )


    # --------------------------------------------------------
    # Save legend handles once
    # --------------------------------------------------------
    if legend_handles is None:

        legend_handles, legend_labels = (
            ax.get_legend_handles_labels()
        )


# ============================================================
# Align all Y labels
# ============================================================
fig.align_ylabels(axes)


# ============================================================
# COMMON LEGEND
# ============================================================
fig.legend(
    legend_handles,

    legend_labels,

    loc="upper left",

    bbox_to_anchor=(
        0.085,
        0.995,
    ),

    ncol=2,

    frameon=False,

    fontsize=LEGEND_FONTSIZE,

    columnspacing=2.2,

    handlelength=1.8,
)


# ============================================================
# Layout
# ============================================================
fig.subplots_adjust(

    # Left margin
    left=0.10,

    # Right margin
    right=0.985,

    # Space for legend
    top=0.87,

    # Bottom margin
    bottom=0.10,

    # Horizontal spacing
    wspace=0.28,

    # Vertical spacing
    hspace=0.62,
)


# ============================================================
# OUTPUT FILES
# ============================================================

png_path = (
    BASE
    / "PDF_Proposed_All_2x2_ALIGNED.png"
)

pdf_path = (
    BASE
    / "PDF_Proposed_All_2x2_ALIGNED.pdf"
)


# ============================================================
# Save PNG
# ============================================================
fig.savefig(
    png_path,

    dpi=600,

    bbox_inches="tight",

    pad_inches=0.05,
)


# ============================================================
# Save PDF
# ============================================================
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
print(
    "\nSaved files:"
)

print(
    png_path.resolve()
)

print(
    pdf_path.resolve()
)