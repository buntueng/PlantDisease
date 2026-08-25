from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ============================================================
# Figure 2: PDF plots at Eb/N0 = 3 dB (2x2)
#
# This script:
#   1) Automatically searches for the four source PNG figures.
#   2) Combines them into a 2x2 layout.
#   3) Places panel labels UNDER each graph:
#        (a) AWGN
#        (b) Rayleigh
#        (c) Rician LOS
#        (d) Impulsive
#   4) Uses Times New Roman for all NEW text added by this script.
#
# IMPORTANT:
# Text already embedded inside the source PNG files is rasterized
# and cannot be changed by matplotlib rcParams. To change those
# internal plot labels/titles/legends to Times New Roman, regenerate
# the four source plots from their original plotting code.
#
# Output:
#   Figure2_PDF_3dB_2x2.png
#   Figure2_PDF_3dB_2x2.pdf
# ============================================================


# ------------------------------------------------------------
# Find a source figure automatically
# ------------------------------------------------------------
def find_file(filename):
    matches = sorted(Path(".").rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Cannot find '{filename}'.\n"
            "Please extract the FINAL LOCKED result ZIP first,\n"
            "then run this script from that folder or a parent folder."
        )

    # Prefer files inside a folder named "figures"
    preferred = [
        p for p in matches
        if p.parent.name.lower() == "figures"
    ]

    return preferred[0] if preferred else matches[0]


# ------------------------------------------------------------
# Source images and panel labels
# ------------------------------------------------------------
plot_files = [
    ("(a) AWGN",       "PDF_Proposed_AWGN_3dB.png"),
    ("(b) Rayleigh",   "PDF_Proposed_Rayleigh_3dB.png"),
    ("(c) Rician LOS", "PDF_Proposed_RicianLOS_3dB.png"),
    ("(d) Impulsive",  "PDF_Proposed_Impulsive_3dB.png"),
]


# ------------------------------------------------------------
# Resolve source paths
# ------------------------------------------------------------
resolved = []

print("Using source figures:")

for title, filename in plot_files:
    path = find_file(filename)
    resolved.append((title, path))
    print(f"{title}: {path.resolve()}")


# ============================================================
# Times New Roman typography
# ============================================================
#
# On Windows/macOS, Times New Roman is usually available.
# Fallback fonts are included in case it is not installed.
#
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "serif",
    ],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Create 2x2 figure
# ============================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(8.6, 6.8),
)

axes = axes.ravel()


# ============================================================
# Draw each source plot
# ============================================================
for ax, (title, path) in zip(axes, resolved):

    # Read source PNG
    img = mpimg.imread(path)

    # Display source image
    ax.imshow(img)

    # Remove matplotlib axes around the embedded image
    ax.axis("off")

    # --------------------------------------------------------
    # Panel label UNDER the graph
    # --------------------------------------------------------
    ax.text(
        0.5,                 # horizontal center
        -0.055,              # below graph
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontfamily="serif",
        clip_on=False,
    )


# ============================================================
# Layout
# ============================================================
#
# hspace is intentionally enlarged because the labels for the
# top row are now placed below each graph.
#
fig.subplots_adjust(
    left=0.025,
    right=0.985,
    top=0.985,
    bottom=0.055,
    wspace=0.10,
    hspace=0.28,
)


# ============================================================
# Save outputs
# ============================================================
png_path = Path(
    "Figure2_PDF_3dB_2x2.png"
)

pdf_path = Path(
    "Figure2_PDF_3dB_2x2.pdf"
)


fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.05,
)


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
# Print saved paths
# ============================================================
print("\nSaved files:")
print(png_path.resolve())
print(pdf_path.resolve())
