from pathlib import Path
import io
import re
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Figure 3: Training convergence under MIXED-channel training
# 1x2 panels:
#   (a) Denoising loss
#   (b) Decoding loss
#
# Uses mean ± std over independent repeats.
# The script first searches extracted history CSVs. If they are
# not found, it automatically searches ZIP files and reads the
# histories directly from the ZIP without extraction.
#
# Output:
#   Figure3_Training_Convergence_MIXED_1x2.png
#   Figure3_Training_Convergence_MIXED_1x2.pdf
# ============================================================

METHODS = [
    "PNND",
    "RNND_Zhu",
    "CNN_RNND",
    "Proposed_CA_CNN_RNND",
]

DISPLAY = {
    "PNND": "PNND",
    "RNND_Zhu": "RNND-Zhu",
    "CNN_RNND": "CNN-RNND",
    "Proposed_CA_CNN_RNND": "Proposed CA-CNN-RNND",
}

DENOISE_METHODS = [
    "RNND_Zhu",
    "CNN_RNND",
    "Proposed_CA_CNN_RNND",
]

EXPECTED_REPEATS = 5


def repeat_from_name(name):
    m = re.search(r"repeat(\d+)_MIXED_", name)
    return int(m.group(1)) if m else None


def load_from_extracted_files():
    frames = []

    for method in METHODS:
        pattern = f"repeat*_MIXED_{method}.csv"
        paths = sorted(Path(".").rglob(pattern))

        for path in paths:
            try:
                h = pd.read_csv(path)
            except Exception:
                continue

            if "method" not in h.columns:
                h["method"] = method

            if "repeat" not in h.columns:
                rep = repeat_from_name(path.name)
                if rep is not None:
                    h["repeat"] = rep

            h["_source"] = str(path)
            frames.append(h)

    return frames


def load_from_zip_files():
    frames = []

    zip_paths = sorted(Path(".").rglob("*.zip"))

    for zpath in zip_paths:
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                names = zf.namelist()

                for method in METHODS:
                    regex = re.compile(
                        rf"(^|/)histories/repeat\d+_MIXED_{re.escape(method)}\.csv$"
                    )

                    for name in names:
                        if not regex.search(name):
                            continue

                        try:
                            raw = zf.read(name)
                            h = pd.read_csv(io.BytesIO(raw))
                        except Exception:
                            continue

                        if "method" not in h.columns:
                            h["method"] = method

                        if "repeat" not in h.columns:
                            rep = repeat_from_name(Path(name).name)
                            if rep is not None:
                                h["repeat"] = rep

                        h["_source"] = f"{zpath}::{name}"
                        frames.append(h)

        except zipfile.BadZipFile:
            continue

    return frames


def load_histories():
    frames = load_from_extracted_files()

    source_mode = "extracted CSV files"

    if not frames:
        frames = load_from_zip_files()
        source_mode = "ZIP archive"

    if not frames:
        raise FileNotFoundError(
            "No MIXED-channel training-history CSV files were found.\n"
            "Expected files such as:\n"
            "  histories/repeat0_MIXED_Proposed_CA_CNN_RNND.csv\n"
            "Please run this script from the FINAL LOCKED result folder,\n"
            "a parent folder, or a folder containing the FINAL result ZIP."
        )

    df = pd.concat(frames, ignore_index=True)

    required = {
        "repeat",
        "method",
        "epoch",
        "denoise_loss",
        "decode_loss",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Training history is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Keep the MIXED regime only if the column exists.
    if "regime" in df.columns:
        df = df[
            df["regime"].astype(str).str.lower() == "mixed"
        ].copy()

    # Remove accidental duplicates if the same histories are discovered
    # more than once.
    df = (
        df.sort_values(["method", "repeat", "epoch"])
          .drop_duplicates(
              subset=["method", "repeat", "epoch"],
              keep="first",
          )
          .reset_index(drop=True)
    )

    print(f"Loaded histories from: {source_mode}")
    print("Repeats detected by method:")
    print(
        df.groupby("method")["repeat"]
          .nunique()
          .reindex(METHODS)
    )

    return df


history = load_histories()

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
available_methods = set(history["method"].astype(str))

missing_methods = [
    m for m in METHODS
    if m not in available_methods
]

if missing_methods:
    print(
        "\nWarning: missing MIXED histories for:",
        ", ".join(missing_methods)
    )

for method in METHODS:
    if method not in available_methods:
        continue

    nrep = history.loc[
        history["method"] == method,
        "repeat"
    ].nunique()

    if nrep < EXPECTED_REPEATS:
        print(
            f"Warning: {method} has {nrep} repeats; "
            f"expected {EXPECTED_REPEATS}."
        )

max_epoch = int(history["epoch"].max())

# Same visualization strategy as the FINAL LOCKED notebook:
# downsample only for plotting; raw histories are unchanged.
plot_stride = max(1, max_epoch // 400)

plot_history = history[
    (history["epoch"] == 1)
    | (history["epoch"] % plot_stride == 0)
    | (history["epoch"] == max_epoch)
].copy()

print(f"\nMaximum epoch: {max_epoch}")
print(f"Plot stride: {plot_stride} epochs")

# ------------------------------------------------------------
# Journal-style typography
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.4, 3.25),
)

# ------------------------------------------------------------
# (a) Denoising loss
# ------------------------------------------------------------
ax = axes[0]

for method in DENOISE_METHODS:
    if method not in available_methods:
        continue

    g = (
        plot_history[
            plot_history["method"] == method
        ]
        .groupby("epoch", as_index=False)
        .agg(
            mean_loss=("denoise_loss", "mean"),
            std_loss=("denoise_loss", "std"),
        )
    )

    g["std_loss"] = g["std_loss"].fillna(0.0)

    line = ax.plot(
        g["epoch"],
        g["mean_loss"],
        label=DISPLAY[method],
        linewidth=1.35,
    )[0]

    lower = np.maximum(
        g["mean_loss"] - g["std_loss"],
        1e-12,
    )
    upper = g["mean_loss"] + g["std_loss"]

    ax.fill_between(
        g["epoch"],
        lower,
        upper,
        alpha=0.14,
        color=line.get_color(),
        linewidth=0,
    )

ax.set_yscale("log")
ax.set_xlabel("Training epoch")
ax.set_ylabel("Denoising MSE")
ax.set_title("(a) Denoising loss")
ax.grid(True, which="both", alpha=0.25, linewidth=0.45)
ax.legend(frameon=False)

# ------------------------------------------------------------
# (b) Decoding loss
# ------------------------------------------------------------
ax = axes[1]

for method in METHODS:
    if method not in available_methods:
        continue

    g = (
        plot_history[
            plot_history["method"] == method
        ]
        .groupby("epoch", as_index=False)
        .agg(
            mean_loss=("decode_loss", "mean"),
            std_loss=("decode_loss", "std"),
        )
    )

    g["std_loss"] = g["std_loss"].fillna(0.0)

    line = ax.plot(
        g["epoch"],
        g["mean_loss"],
        label=DISPLAY[method],
        linewidth=1.35,
    )[0]

    lower = np.maximum(
        g["mean_loss"] - g["std_loss"],
        1e-12,
    )
    upper = g["mean_loss"] + g["std_loss"]

    ax.fill_between(
        g["epoch"],
        lower,
        upper,
        alpha=0.14,
        color=line.get_color(),
        linewidth=0,
    )

ax.set_yscale("log")
ax.set_xlabel("Training epoch")
ax.set_ylabel("Decoding loss")
ax.set_title("(b) Decoding loss")
ax.grid(True, which="both", alpha=0.25, linewidth=0.45)
ax.legend(frameon=False)

fig.tight_layout(w_pad=1.5)

png_path = Path("Figure3_Training_Convergence_MIXED_1x2.png")
pdf_path = Path("Figure3_Training_Convergence_MIXED_1x2.pdf")

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.show()

print("\nSaved files:")
print(png_path.resolve())
print(pdf_path.resolve())
