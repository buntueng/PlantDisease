from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ------------------------------------------------------------
# Figure 2: PDF plots at Eb/N0 = 3 dB (2x2)
# Automatically searches the current folder and all subfolders
# for the four Proposed-model PNG figures:
#
#   PDF_Proposed_AWGN_3dB.png
#   PDF_Proposed_Rayleigh_3dB.png
#   PDF_Proposed_RicianLOS_3dB.png
#   PDF_Proposed_Impulsive_3dB.png
#
# Output:
#   Figure2_PDF_3dB_2x2.png
#   Figure2_PDF_3dB_2x2.pdf
# ------------------------------------------------------------

def find_file(filename):
    matches = sorted(Path(".").rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Cannot find '{filename}'.\n"
            "Please extract the FINAL LOCKED result ZIP first,\n"
            "then run this script from that folder or a parent folder."
        )

    # Prefer files inside a 'figures' directory
    preferred = [p for p in matches if p.parent.name == "figures"]
    return preferred[0] if preferred else matches[0]


plot_files = [
    ("(a) AWGN", "PDF_Proposed_AWGN_3dB.png"),
    ("(b) Rayleigh", "PDF_Proposed_Rayleigh_3dB.png"),
    ("(c) Rician LOS", "PDF_Proposed_RicianLOS_3dB.png"),
    ("(d) Impulsive", "PDF_Proposed_Impulsive_3dB.png"),
]

resolved = []

print("Using source figures:")
for title, filename in plot_files:
    path = find_file(filename)
    resolved.append((title, path))
    print(f"{title}: {path.resolve()}")

# Journal-friendly typography
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    2,
    2,
    figsize=(8.2, 6.6),
)

axes = axes.ravel()

for ax, (title, path) in zip(axes, resolved):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.set_title(title, pad=5)
    ax.axis("off")

# Do not add a large figure title inside the graphic.
# The journal caption should be added in LaTeX.
fig.tight_layout(
    pad=0.7,
    w_pad=0.5,
    h_pad=0.8,
)

png_path = Path("Figure2_PDF_3dB_2x2.png")
pdf_path = Path("Figure2_PDF_3dB_2x2.pdf")

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
