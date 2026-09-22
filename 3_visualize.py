"""
Module: 3_visualize.py
One figure per bucket from the .npy files 2_batch_extract.sh writes, as a
preliminary check that the extraction is sound before any geometry is derived
from it.

Each figure carries the two power observables and the amplitude of the
beamforming matrix: the monitor's received signal, which describes the
transmitter-to-monitor path; the reported average SNR per space-time stream,
which describes the beamformer-to-beamformee path and so moves independently of
it; and the share of each stream's power every transmit antenna carries, which
is the part of V that shifts as the reporting station moves.

The share is shown twice: averaged over subcarriers against time, one panel per
stream, and per subcarrier as one map per (stream, antenna). Figure height and
panel count follow Nr and Nc.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from main import NATIVE_NG, subcarrier_count, subcarrier_indices

ANALYSIS_ROOT = "../bfi-workspace/analysis"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedcd6"
# Categorical slots in fixed order, one per antenna or stream; a configuration
# never has more than four of either.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# Single hue, light to dark: the heatmap encodes a magnitude, not a category.
SEQUENTIAL = LinearSegmentedColormap.from_list("bfi_blue", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
# One line height at the label font size, the closest two end labels may sit.
LABEL_LINE_PX = 12.0


def style(ax, ylabel, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(GRID)


def label_ends(ax, x_last, points):
    """
    Name each series beside its own last point, so identity does not rest on
    colour alone. Labels are pushed apart to one line height where series end
    close together, which four antennas holding similar shares routinely do.
    Spacing is measured in pixels, so call this once the layout is settled.
    """
    to_pixels, to_data = ax.transData, ax.transData.inverted()
    placed = []
    for y_pixels, text in sorted((to_pixels.transform((x_last, y))[1], text)
                                 for y, text in points):
        if placed and y_pixels - placed[-1][0] < LABEL_LINE_PX:
            y_pixels = placed[-1][0] + LABEL_LINE_PX
        placed.append((y_pixels, text))
    for y_pixels, text in placed:
        ax.annotate(text, (x_last, to_data.transform((0, y_pixels))[1]),
                    xytext=(4, 0), textcoords="offset points", color=INK_MUTED,
                    fontsize=8, va="center", clip_on=False)


def last_finite(values):
    """The final value a series actually carries, or None if it carries none."""
    finite = np.flatnonzero(np.isfinite(values))
    return float(values[finite[-1]]) if finite.size else None


def plot_bucket(key, samples, feedback, out_path):
    transmitter, receiver, shape = key.split("_")
    config, bw = shape.split("@")
    nr, nc = (int(x) for x in config.split("x"))

    times = np.array([s[0] for s in samples])
    order = np.argsort(times)
    elapsed = times[order] - times[order][0]
    v = np.stack([samples[i][1] for i in order])
    rssi = np.array([np.nan if samples[i][2] is None else samples[i][2] for i in order])
    # One SNR per space-time stream, padded where a truncated frame carried fewer
    # than the configuration calls for, so the stack stays rectangular.
    snr = np.full((len(order), nc), np.nan)
    for row, i in enumerate(order):
        reported = list(samples[i][3])[:nc]
        snr[row, :len(reported)] = reported

    # Columns of V are orthonormal, so the squared magnitudes of one column are
    # the share of that stream's power on each transmit antenna and sum to 1
    # over the antenna axis.
    share = np.abs(v) ** 2                              # (report, subcarrier, Nr, Nc)

    nsubc = v.shape[1]
    ng = samples[0][5]["ng"]
    standard = next((std for std in ("AC", "AX")
                     if subcarrier_count(std, int(bw), ng) == nsubc), None)
    # Grouped feedback defines no index positions, so the heatmap falls back to
    # an ordinal axis rather than labelling rows with subcarriers it cannot name.
    grouped = standard is None or ng != NATIVE_NG[standard]
    subcarriers = (np.arange(nsubc) if grouped
                   else subcarrier_indices(standard, int(bw)))

    pending_labels = []
    # Three bands, each its own subfigure so every grid inside one is regular:
    # a shared-width pair of line plots, Nc per-stream panels, then an Nc by Nr
    # map grid. One gridspec spanning all three would mix column spans and the
    # layout engine cannot align them.
    fig = plt.figure(figsize=(13, 5.2 + 2.6 * nc), facecolor=SURFACE,
                     layout="constrained")
    fig.suptitle(f"{transmitter} → {receiver}   {config} @ {bw} MHz   "
                 f"{standard or '?'} {feedback}   ·   {len(samples)} reports "
                 f"over {elapsed[-1]:.1f} s", color=INK, fontsize=13)
    band_signal, band_share, band_maps = fig.subfigures(
        3, 1, height_ratios=[2.0, 1.3, 1.9 * nc])
    for band in (band_signal, band_share, band_maps):
        band.set_facecolor(SURFACE)
    grid = band_signal.add_gridspec(2, 1)

    ax = band_signal.add_subplot(grid[0, :])
    ax.plot(elapsed, rssi, color=SERIES[0], linewidth=1.5)
    ax.set_title("Monitor received signal", color=INK, fontsize=10, loc="left")
    style(ax, "dBm")

    ax = band_signal.add_subplot(grid[1, :])
    ends = []
    for c in range(nc):
        ax.plot(elapsed, snr[:, c], color=SERIES[c], linewidth=1.5, label=f"stream {c + 1}")
        end = last_finite(snr[:, c])
        if end is not None:
            ends.append((end, f"s{c + 1}"))
    ax.set_title("Reported average SNR, beamformer to beamformee", color=INK,
                 fontsize=10, loc="left")
    style(ax, "dB")
    pending_labels.append((ax, elapsed[-1], ends))
    if nc > 1:
        # Headroom for the legend, as the share panels get from their fixed
        # 0..1.15 limits; the autoscaled range leaves none.
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.32 * (high - low))
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=nc,
                  labelcolor=INK_MUTED)

    # One panel per stream: a stream's shares sum to 1 across antennas, so
    # streams are not comparable on one pair of axes.
    per_antenna = share.mean(axis=1)                    # (report, Nr, Nc)
    share_grid = band_share.add_gridspec(1, nc)
    for c in range(nc):
        ax = band_share.add_subplot(share_grid[0, c])
        for r in range(nr):
            ax.plot(elapsed, per_antenna[:, r, c], color=SERIES[r], linewidth=1.5,
                    label=f"antenna {r + 1}")
        ax.set_title(f"Antenna power share, stream {c + 1}" if nc > 1 else
                     "Antenna power share of stream 1, mean over subcarriers",
                     color=INK, fontsize=10, loc="left")
        style(ax, "share of 1\nmean over subcarriers" if c == 0 else "",
              "Seconds since first report")
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        if c:
            ax.set_yticklabels([])
        pending_labels.append((ax, elapsed[-1],
                               [(per_antenna[-1, r, c], f"a{r + 1}") for r in range(nr)]))
        if c == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=nr,
                      labelcolor=INK_MUTED)

    # Reports are irregularly spaced, so the maps are drawn against report
    # index: spreading one report across a long gap would show feedback that
    # was never sent. The span above gives the timing.
    #
    # One map per (stream, antenna). A common 0..1 scale makes cells comparable
    # within a stream, which is where they sum to 1.
    edges_x = np.arange(len(samples) + 1)
    edges_y = np.append(subcarriers, subcarriers[-1] + 1)
    maps = []
    map_grid = band_maps.add_gridspec(nc, nr)
    for c in range(nc):
        for r in range(nr):
            ax = band_maps.add_subplot(map_grid[c, r])
            mesh = ax.pcolormesh(edges_x, edges_y, share[:, :, r, c].T,
                                 cmap=SEQUENTIAL, vmin=0, vmax=1,
                                 shading="flat", rasterized=True)
            if c == 0:
                ax.set_title(f"antenna {r + 1}", color=INK, fontsize=10, loc="left")
            axis = "Matrix index" if grouped else "Subcarrier index"
            style(ax, f"stream {c + 1}\n{axis}" if r == 0 else "",
                  "Report index" if c == nc - 1 else None)
            if r:
                ax.set_yticklabels([])
            ax.grid(False)
            maps.append(ax)

    bar = band_maps.colorbar(mesh, ax=maps, pad=0.012, fraction=0.02)
    bar.set_label("share of 1", color=INK_MUTED, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    bar.outline.set_visible(False)

    # The end labels are spaced in pixels, so they go on once the layout is
    # settled; constrained_layout resolves positions on the first draw.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for target, x_last, points in pending_labels:
        label_ends(target, x_last, points)

    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def generate_reports(root):
    root = Path(root)
    if not root.exists():
        print(f"Error: {root} directory not found.")
        return

    for path in sorted(root.glob("*/*/v_matrix.npy")):
        feedback = path.parent.name
        buckets = np.load(path, allow_pickle=True).item()
        for key, samples in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            # Two reports are the minimum a shift can be read from.
            if len(samples) < 2:
                print(f"   ! skipping {key} ({feedback}): {len(samples)} report")
                continue
            if len({s[1].shape for s in samples}) > 1:
                print(f"   ! skipping {key} ({feedback}): stack is ragged")
                continue
            name = f"{key.replace(':', '').replace('@', '-')}.png"
            out_path = path.parent / name
            plot_bucket(key, samples, feedback, out_path)
            print(f"   [OK] {path.parent.relative_to(root)}/{name}")


if __name__ == "__main__":
    generate_reports(sys.argv[1] if len(sys.argv) > 1 else ANALYSIS_ROOT)
