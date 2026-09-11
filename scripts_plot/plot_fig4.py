import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
JSON_DIR_C = BASE_DIR / "res_c" / "res_test_snr"
JSON_DIR_Q = BASE_DIR / "res_q" / "res_test_snr"
SAVE_DIRS = {
    "png": BASE_DIR / "fig_paper2",
    "pdf": BASE_DIR / "fig_paper2",
}

SNR_GROUPS = ["low", "high"]
PANEL_LABELS = ["a", "b"]
HEAD_VALUES = [3, 4, 5]

MARKER_Q = "o"
MARKER_C = "s"
LINE_STYLE_Q = "-"
LINE_STYLE_C = "-"
COLORS_Q = {
    3: "#f28e8e",
    4: "#e15759",
    5: "#b22222",
}
COLORS_C = {
    3: "#9ecae1",
    4: "#4292c6",
    5: "#08519c",
}
LEGEND_NCOL = 3

PANEL_TITLES = {
    "low": r"Low SNR ($\mathrm{SNR}<12$)",
    "high": r"High SNR ($\mathrm{SNR}\geq12$)",
}

FIG_COMB_W_IN = 7.0
FIG_COMB_H_IN = 2.8

LEFT_FRAC = 0.45
RIGHT_FRAC = 0.45
GAP = 0.10
Q_VIS_START = 0.0
C_VIS_START = LEFT_FRAC + GAP

Q_REAL_MIN, Q_REAL_MAX = 0, 200
C_REAL_MIN, C_REAL_MAX = 200, 3000
Q_TICKS = [0, 100, 200]
C_TICKS = [1000, 2000, 3000]

ERRORBAR_KW = {
    "capsize": 2.0,
    "capthick": 0.6,
    "elinewidth": 0.6,
    "markeredgewidth": 0.4,
    "markersize": 2.5,
    "linewidth": 1.0,
}


def setup_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.0,
        "lines.markersize": 2.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def compute_param_count_c(emb, inner):
    return emb * inner * 3


def compute_param_count_q(depth, qubits, end):
    if end < qubits:
        num = 2 * end
    else:
        num = qubits * (depth + 1) * 2
    return num * 3


def parse_key_c(key):
    match = re.search(r"emb(\d+)_inner(\d+)", key)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_key_q(key):
    match = re.search(r"q(\d+)_circuit_depth(\d+)_end(\d+)", key)
    if not match:
        return None
    qubits, depth, end = map(int, match.groups())
    return depth, qubits, end


def mean_std(values, snr_group):
    metric_values = np.array(
        [float(trial[snr_group]["recall"]) for trial in values],
        dtype=float,
    )
    return float(np.mean(metric_values)), float(np.std(metric_values, ddof=1))


def _sort_by_x(x_vals, means, stds):
    order = np.argsort(x_vals)
    return (
        np.array(x_vals)[order],
        np.array(means)[order],
        np.array(stds)[order],
    )


def to_visual_x(real_x, real_min, real_max, vis_start, vis_width):
    real_x = np.asarray(real_x, dtype=float)
    if real_max == real_min:
        n = len(real_x)
        if n == 1:
            return np.array([vis_start + vis_width / 2.0])
        return vis_start + np.linspace(0.0, vis_width, n)
    frac = (real_x - real_min) / (real_max - real_min)
    return vis_start + frac * vis_width


def _c_to_visual_x(cx):
    cx = np.asarray(cx, dtype=float)
    vx = np.empty_like(cx)
    left = cx < C_REAL_MIN
    if left.any():
        vx[left] = to_visual_x(
            cx[left], Q_REAL_MIN, Q_REAL_MAX, Q_VIS_START, LEFT_FRAC,
        )
    if (~left).any():
        vx[~left] = to_visual_x(
            cx[~left], C_REAL_MIN, C_REAL_MAX, C_VIS_START, RIGHT_FRAC,
        )
    return vx


def clipped_yerr(means, stds):
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    lower = np.minimum(stds, means)
    upper = np.minimum(stds, 100.0 - means)
    return np.vstack([lower, upper])


def _load_curve_q(qubits, snr_group):
    json_path = JSON_DIR_Q / f"res_test_snr_q{qubits}.json"
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    x_vals, means, stds = [], [], []
    for key, values in data.items():
        if not isinstance(values, list) or not values:
            continue
        parsed = parse_key_q(key)
        if parsed is None:
            continue
        depth, q, end = parsed
        x_vals.append(compute_param_count_q(depth, q, end))
        mean, std = mean_std(values, snr_group)
        means.append(mean)
        stds.append(std)
    return _sort_by_x(x_vals, means, stds)


def _load_curve_c(head_value, snr_group):
    json_path = JSON_DIR_C / f"res_test_snr_c{head_value}.json"
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    x_vals, means, stds = [], [], []
    for key, values in data.items():
        if not isinstance(values, list) or not values:
            continue
        parsed = parse_key_c(key)
        if parsed is None:
            continue
        emb, inner = parsed
        x_vals.append(compute_param_count_c(emb, inner))
        mean, std = mean_std(values, snr_group)
        means.append(mean)
        stds.append(std)
    return _sort_by_x(x_vals, means, stds)


def _ylim_for_curves(q_curves, c_curves, pad=0.6):
    lows, highs = [], []
    for _, _, means, stds in q_curves:
        yerr = clipped_yerr(means, stds)
        lows.extend(means - yerr[0])
        highs.extend(means + yerr[1])
    for _, _, means, stds in c_curves:
        yerr = clipped_yerr(means, stds)
        lows.extend(means - yerr[0])
        highs.extend(means + yerr[1])
    return max(0.0, min(lows) - pad), min(100.0, max(highs) + pad)


def _draw_compare_on_ax(ax, snr_group):
    q_curves = [(v, *_load_curve_q(v, snr_group)) for v in HEAD_VALUES]
    c_curves = [(v, *_load_curve_c(v, snr_group)) for v in HEAD_VALUES]

    q_handles, q_labels = [], []
    c_handles, c_labels = [], []

    for v, qx, qm, qs in q_curves:
        vx = to_visual_x(qx, Q_REAL_MIN, Q_REAL_MAX, Q_VIS_START, LEFT_FRAC)
        handle = ax.errorbar(
            vx,
            qm,
            yerr=clipped_yerr(qm, qs),
            fmt=f"{MARKER_Q}{LINE_STYLE_Q}",
            color=COLORS_Q[v],
            zorder=3,
            **ERRORBAR_KW,
        )
        q_handles.append(handle[0])
        q_labels.append(f"QViT qubits={v}")

    for v, cx, cm, cs in c_curves:
        vx = _c_to_visual_x(cx)
        handle = ax.errorbar(
            vx,
            cm,
            yerr=clipped_yerr(cm, cs),
            fmt=f"{MARKER_C}{LINE_STYLE_C}",
            color=COLORS_C[v],
            zorder=3,
            **ERRORBAR_KW,
        )
        c_handles.append(handle[0])
        c_labels.append(f"ViT dim_head={v}")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(_ylim_for_curves(q_curves, c_curves))

    q_tick_vis = to_visual_x(
        np.array(Q_TICKS), Q_REAL_MIN, Q_REAL_MAX, Q_VIS_START, LEFT_FRAC,
    )
    c_tick_vis = to_visual_x(
        np.array(C_TICKS), C_REAL_MIN, C_REAL_MAX, C_VIS_START, RIGHT_FRAC,
    )
    ax.set_xticks(np.concatenate([q_tick_vis, c_tick_vis]))
    ax.set_xticklabels([str(v) for v in Q_TICKS + C_TICKS], rotation=0, fontsize=7)

    ax.set_title(PANEL_TITLES[snr_group])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    handles, labels = [], []
    for q_handle, c_handle, q_label, c_label in zip(q_handles, c_handles, q_labels, c_labels):
        handles.extend([q_handle, c_handle])
        labels.extend([q_label, c_label])
    return handles, labels


def _save_figure(fig, stem):
    for ext, save_dir in SAVE_DIRS.items():
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{stem}.{ext}"
        try:
            fig.savefig(save_path)
            print(f"[OK] saved to: {save_path}")
        except PermissionError as exc:
            print(f"[SKIP] cannot write {save_path}: {exc}")


def plot_fig4():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_COMB_W_IN, FIG_COMB_H_IN))
    legend_handles, legend_labels = None, None

    for ax, snr_group, panel in zip(axes, SNR_GROUPS, PANEL_LABELS):
        handles, labels = _draw_compare_on_ax(ax, snr_group)
        ax.set_ylabel("Recall (%)")
        ax.set_xlabel("Number of QKV parameters")
        ax.text(
            0.02,
            0.98,
            panel,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
            ha="left",
        )
        if legend_handles is None:
            legend_handles, legend_labels = handles, labels

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=LEGEND_NCOL,
        frameon=False,
        handlelength=2.5,
        handletextpad=0.5,
        columnspacing=0.9,
    )
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.35)

    _save_figure(fig, "fig4")
    plt.close(fig)


def main():
    setup_nature_style()
    print("[Plotting] fig4 SNR-stratified recall comparison")
    plot_fig4()


if __name__ == "__main__":
    main()
