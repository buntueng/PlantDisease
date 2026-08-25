
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FAST plotting script
# Reads saved .npz files only — NO training, NO checkpoint load.
# ============================================================

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

FILES = {
    "AWGN": BASE / "PDF_Proposed_AWGN_3dB.npz",
    "Rayleigh": BASE / "PDF_Proposed_Rayleigh_3dB.npz",
    "Rician LOS": BASE / "PDF_Proposed_RicianLOS_3dB.npz",
    "Impulsive": BASE / "PDF_Proposed_Impulsive_3dB.npz",
}

TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 12

N_BINS = 180
SNR_DB = 3

plt.rcParams.update({
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def density_data(npz_path):
    d = np.load(npz_path)
    received = d["received"].ravel()
    denoised = d["denoised"].ravel()

    lo = min(
        -5.0,
        float(np.quantile(received, 0.001)),
        float(np.quantile(denoised, 0.001)),
    )
    hi = max(
        5.0,
        float(np.quantile(received, 0.999)),
        float(np.quantile(denoised, 0.999)),
    )

    bins = np.linspace(lo, hi, N_BINS)

    h_received, edges = np.histogram(
        received,
        bins=bins,
        density=True,
    )
    h_denoised, _ = np.histogram(
        denoised,
        bins=bins,
        density=True,
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, h_received, h_denoised


def style_axis(ax, channel, show_title=True):
    ax.set_xlabel(
        "Symbol value",
        fontsize=LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        "Probability density",
        fontsize=LABEL_FONTSIZE,
    )
    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
    )
    ax.grid(
        True,
        alpha=0.20,
        linewidth=0.6,
    )

    if show_title:
        ax.set_title(
            f"Proposed — {channel}, "
            rf"$E_b/N_0={SNR_DB}$ dB",
            fontsize=TITLE_FONTSIZE,
            pad=8,
        )


# ============================================================
# 1) Four individual figures
# ============================================================
for channel, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    centers, h0, h1 = density_data(path)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.fill_between(
        centers,
        h0,
        alpha=0.42,
        label="Received symbols",
    )
    ax.fill_between(
        centers,
        h1,
        alpha=0.42,
        label="Denoised symbols",
    )

    style_axis(ax, channel)

    ax.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
    )

    fig.tight_layout()

    file_channel = channel.replace(" ", "")
    out_png = BASE / f"PDF_Proposed_{file_channel}_3dB_EDITED.png"
    out_pdf = BASE / f"PDF_Proposed_{file_channel}_3dB_EDITED.pdf"

    fig.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        out_pdf,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)

    print(f"Saved: {out_png.name}")
    print(f"Saved: {out_pdf.name}")


# ============================================================
# 2) Combined 2x2 figure
# ============================================================
channels = [
    "AWGN",
    "Rayleigh",
    "Rician LOS",
    "Impulsive",
]

panel_labels = {
    "AWGN": "(a) AWGN",
    "Rayleigh": "(b) Rayleigh",
    "Rician LOS": "(c) Rician LOS",
    "Impulsive": "(d) Impulsive",
}

fig, axes = plt.subplots(
    2,
    2,
    figsize=(8.2, 6.4),
)

axes = axes.ravel()

legend_handles = None
legend_labels = None

for ax, channel in zip(axes, channels):

    centers, h0, h1 = density_data(FILES[channel])

    ax.fill_between(
        centers,
        h0,
        alpha=0.42,
        label="Received symbols",
    )
    ax.fill_between(
        centers,
        h1,
        alpha=0.42,
        label="Denoised symbols",
    )

    style_axis(
        ax,
        channel,
        show_title=False,
    )

    ax.text(
        0.5,
        -0.27,
        panel_labels[channel],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        clip_on=False,
    )

    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()


fig.legend(
    legend_handles,
    legend_labels,
    loc="upper left",
    bbox_to_anchor=(0.095, 0.995),
    ncol=2,
    frameon=False,
    fontsize=LEGEND_FONTSIZE,
)

fig.subplots_adjust(
    left=0.10,
    right=0.985,
    top=0.88,
    bottom=0.11,
    wspace=0.28,
    hspace=0.62,
)

combined_png = BASE / "PDF_Proposed_All_2x2_EDITED.png"
combined_pdf = BASE / "PDF_Proposed_All_2x2_EDITED.pdf"

fig.savefig(
    combined_png,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.05,
)

fig.savefig(
    combined_pdf,
    bbox_inches="tight",
    pad_inches=0.05,
)

plt.close(fig)

print(f"Saved: {combined_png.name}")
print(f"Saved: {combined_pdf.name}")
print("\nDone — no model training was used.")
