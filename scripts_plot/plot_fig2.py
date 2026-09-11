"""
Plot the schematic of the Enhanced Parameterized Quantum Circuit (Enhanced PQC)
for the paper.

Environment: pytorch virtual env
    python scripts_fig/plot_fig2.py

Circuit structure (identical to _circuit_enhanced + _block2 in model/q_layer.py):
    1. Enhancement layer U_enh (trainable variational layer before data encoding,
       corresponding to type_qc='zy'):
         - RZ + RY on qubits 0~4 (qubit 5 has no RZ/RY, i.e. index_qubit_end=5)
         - CZ nearest-neighbour chain 0-1,1-2,2-3,3-4,4-5 (open chain, no 5-0)
    2. Encoding layer U_enc(x): a single RX layer (AngleEmbedding, rotation='X')
       that encodes one input row into the circuit.
    3. Conventional variational layers U_PQC: depth L = 0 (no variational layer
       after the RX layer).

Overall architecture: |psi(x)> = U_enh . U_enc(x) . U_PQC |0>^{otimes n_q}

Data flow (quantum attention):
    input (N, d) --split into 4 heads--> [n_h, N, d_h] (n_h=4 fixed, d_h=6 best).
    Each [N, d_h] (N=17) enters the circuit row by row: every row x_i in R^{d_h}
    is encoded onto RX, and measuring all qubits gives one output row in R^{d_h}.
    Hence [N, d_h] in, [N, d_h] out.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyBboxPatch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SAVE_ROOT = os.path.join(PROJECT_ROOT, "fig_paper2")
SAVE_DIRS = {
    "png": SAVE_ROOT,
    "pdf": SAVE_ROOT,
}

# ----------------------------------------------------------------------------
# Circuit hyper-parameters
# ----------------------------------------------------------------------------
N_QUBITS = 6           # n_q = d_h = 6
INDEX_QUBIT_END = 5    # qubits 0~4 carry RZ/RY, qubit 5 does not
N_HEADS = 4            # n_h fixed to 4
N_TOKENS = 17          # N = 17
PQC_LAYERS = 0         # depth L = 0 of conventional variational layers after RX

# ----------------------------------------------------------------------------
# Colors (project colorblind-friendly palette)
# ----------------------------------------------------------------------------
C_RZ = "#56B4E9"   # sky blue
C_RY = "#009E73"   # green
C_RX = "#D55E00"   # orange (encoding layer, emphasized)
C_CZ = "#222222"   # nearest-neighbour entanglement
C_MEAS = "#7F7F7F" # measurement
C_ENH = "#0072B2"  # enhancement-layer annotation
C_ENC = "#D55E00"  # encoding-layer annotation
C_PQC = "#999999"  # variational-layer annotation

GATE_W = 0.62
GATE_H = 0.50


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "DejaVu Sans",
        ],
        "font.size": 9,
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def y_of(qubit: int) -> float:
    """Qubit 0 is placed at the top."""
    return -qubit


def draw_gate(ax, x, qubit, label, facecolor, text_color="white"):
    y = y_of(qubit)
    box = FancyBboxPatch(
        (x - GATE_W / 2, y - GATE_H / 2),
        GATE_W,
        GATE_H,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        linewidth=1.0,
        edgecolor="black",
        facecolor=facecolor,
        zorder=4,
    )
    ax.add_patch(box)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=9.5, fontweight="bold",
        color=text_color, zorder=5,
    )


def draw_cz(ax, x, qubit_a, qubit_b):
    ya, yb = y_of(qubit_a), y_of(qubit_b)
    ax.plot([x, x], [ya, yb], color=C_CZ, linewidth=1.4, zorder=3)
    for yy in (ya, yb):
        dot = Circle((x, yy), radius=0.075, facecolor=C_CZ,
                     edgecolor=C_CZ, zorder=5)
        ax.add_patch(dot)


def draw_measure(ax, x, qubit):
    y = y_of(qubit)
    box = FancyBboxPatch(
        (x - GATE_W / 2, y - GATE_H / 2),
        GATE_W,
        GATE_H,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        linewidth=1.0,
        edgecolor="black",
        facecolor=C_MEAS,
        zorder=4,
    )
    ax.add_patch(box)
    arc = Arc((x, y - 0.07), GATE_W * 0.58, GATE_H * 0.70,
              theta1=0, theta2=180, color="white", linewidth=1.2, zorder=5)
    ax.add_patch(arc)
    ax.plot([x, x + 0.14], [y - 0.07, y + 0.11],
            color="white", linewidth=1.2, zorder=5)


def draw_top_bracket(ax, x0, x1, y, text, color, drop=0.16):
    """Draw a square bracket above the circuit with a centered label."""
    ax.plot([x0, x1], [y, y], color=color, linewidth=1.6,
            zorder=2, clip_on=False)
    ax.plot([x0, x0], [y, y - drop], color=color, linewidth=1.6,
            zorder=2, clip_on=False)
    ax.plot([x1, x1], [y, y - drop], color=color, linewidth=1.6,
            zorder=2, clip_on=False)
    ax.text((x0 + x1) / 2, y + 0.12, text,
            ha="center", va="bottom", color=color,
            fontsize=10.5, fontweight="bold", zorder=2, clip_on=False)


def draw_right_brace(ax, x, y0, y1, color, tick=0.16):
    """Draw a vertical square bracket on the right, ticks pointing left."""
    ax.plot([x, x], [y0, y1], color=color, linewidth=1.6,
            zorder=2, clip_on=False)
    ax.plot([x, x - tick], [y0, y0], color=color, linewidth=1.6,
            zorder=2, clip_on=False)
    ax.plot([x, x - tick], [y1, y1], color=color, linewidth=1.6,
            zorder=2, clip_on=False)


def build_figure():
    # ---- column coordinates ----
    x_rz = 1.0
    x_ry = 2.0
    x_cz = [3.0, 4.0, 5.0, 6.0, 7.0]   # CZ(0,1)..CZ(4,5)
    x_rx = 8.0
    x_meas = 10.4

    x_wire_l = 0.2
    x_clast = x_meas + GATE_W / 2      # classical wires start at the meter's right edge
    x_brace = 12.4                     # right vertical brace grouping all outputs
    x_center = (x_wire_l + x_meas) / 2 # horizontal center of the circuit

    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_axis_off()

    # ---- quantum wires (with |0> kets and qubit labels), up to the meters ----
    for q in range(N_QUBITS):
        y = y_of(q)
        ax.plot([x_wire_l, x_meas], [y, y],
                color="black", linewidth=1.0, zorder=1)
        ax.text(x_wire_l - 0.65, y, r"$|0\rangle$",
                ha="center", va="center", fontsize=10)
        ax.text(x_wire_l - 1.35, y, rf"$q_{{{q}}}$",
                ha="center", va="center", fontsize=10, color="#444444")

    # ---- enhancement layer: RZ + RY (qubits 0~4) ----
    for q in range(INDEX_QUBIT_END):
        draw_gate(ax, x_rz, q, "RZ", C_RZ)
        draw_gate(ax, x_ry, q, "RY", C_RY)

    # ---- enhancement layer: CZ nearest-neighbour chain (open chain) ----
    for x, q in zip(x_cz, range(N_QUBITS - 1)):
        draw_cz(ax, x, q, q + 1)

    # ---- encoding layer: a single RX layer (AngleEmbedding, all qubits) ----
    for q in range(N_QUBITS):
        draw_gate(ax, x_rx, q, "RX", C_RX)

    # ---- measurement: every qubit is measured ----
    for q in range(N_QUBITS):
        draw_measure(ax, x_meas, q)

    # ---- classical output wires + per-qubit <Z_q>, grouped into one row ----
    y_top, y_bot = y_of(0), y_of(N_QUBITS - 1)
    off = 0.045
    for q in range(N_QUBITS):
        y = y_of(q)
        ax.plot([x_clast, x_brace], [y + off, y + off],
                color=C_MEAS, linewidth=0.9, zorder=1)
        ax.plot([x_clast, x_brace], [y - off, y - off],
                color=C_MEAS, linewidth=0.9, zorder=1)
        ax.text(x_clast + 0.12, y + 0.16, rf"$\langle Z_{{{q}}}\rangle$",
                ha="left", va="bottom", fontsize=8.5, color="#444444")

    draw_right_brace(ax, x_brace, y_top + 0.32, y_bot - 0.32, C_MEAS)
    y_mid = (y_top + y_bot) / 2
    ax.text(x_brace + 0.22, y_mid + 0.62,
            rf"Measure all $n_q={N_QUBITS}$ qubits",
            ha="left", va="center", fontsize=9.5, color="#444444")
    ax.text(x_brace + 0.22, y_mid + 0.12,
            r"Output row $\in \mathbb{R}^{d_h}$",
            ha="left", va="center", fontsize=10.5, color="#222222",
            fontweight="bold")
    ax.text(x_brace + 0.22, y_mid - 0.42,
            r"$=(\langle Z_0\rangle,\,\langle Z_1\rangle,\,\dots,\,"
            r"\langle Z_{n_q-1}\rangle)$",
            ha="left", va="center", fontsize=10, color="#444444")

    # ---- conventional variational layers (L=0): dashed empty box ----
    pqc_x0 = x_rx + GATE_W / 2 + 0.15
    pqc_x1 = x_meas - GATE_W / 2 - 0.15
    ax.add_patch(FancyBboxPatch(
        (pqc_x0, y_of(N_QUBITS - 1) - 0.35),
        pqc_x1 - pqc_x0,
        (N_QUBITS - 1) + 0.70,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.1, linestyle=(0, (4, 3)),
        edgecolor=C_PQC, facecolor="none", zorder=2,
    ))
    ax.text((pqc_x0 + pqc_x1) / 2, y_of(N_QUBITS - 1) - 0.62,
            "empty", ha="center", va="center",
            fontsize=8.5, color=C_PQC, style="italic")

    # ---- top brackets (encoding/PQC regions are adjacent; short labels) ----
    y_bracket = 0.95
    draw_top_bracket(
        ax, x_rz - GATE_W / 2 - 0.15, x_cz[-1] + 0.15, y_bracket,
        r"Enhancement layer  $U_{\mathrm{enh}}$", C_ENH,
    )
    draw_top_bracket(
        ax, x_rx - GATE_W / 2 - 0.05, x_rx + GATE_W / 2 + 0.05, y_bracket,
        r"$U_{\mathrm{enc}}(\mathbf{x})$", C_ENC,
    )
    # PQC bracket is inset on the left so it is clearly separated from U_enc,
    # and its label is staggered to a higher line with a leader
    pqc_bx0 = pqc_x0 + 0.28
    draw_top_bracket(ax, pqc_bx0, pqc_x1, y_bracket, "", C_PQC)
    pqc_cx = (pqc_bx0 + pqc_x1) / 2
    ax.plot([pqc_cx, pqc_cx], [y_bracket + 0.14, y_bracket + 0.52],
            color=C_PQC, linewidth=1.0, clip_on=False)
    ax.text(pqc_cx, y_bracket + 0.58, r"$U_{\mathrm{PQC}}\ (L{=}0)$",
            ha="center", va="bottom", color=C_PQC,
            fontsize=10.5, fontweight="bold")

    # ---- three call-out notes ----
    notes = [
        (C_ENH, r"The part before RX (RZ + RY + CZ) is the enhancement layer"),
        (C_ENC, r"A single RX layer is the encoding layer"),
        (C_PQC, r"No conventional variational layer after RX ($L=0$)"),
    ]

    # ---- encoding-input arrow (pointing at the RX column) ----
    ax.annotate(
        "",
        xy=(x_rx, y_bot - 0.30),
        xytext=(x_rx, y_bot - 1.05),
        arrowprops=dict(arrowstyle="-|>", color=C_RX, linewidth=1.6),
    )
    ax.text(
        x_rx, y_bot - 1.22,
        r"Input row $\mathbf{x}_i \in \mathbb{R}^{d_h}$" "\n"
        r"encoded row by row onto RX",
        ha="center", va="top", fontsize=9, color=C_RX,
    )

    # ---- title + master equation ----
    ax.text(
        x_center, 2.85,
        "Enhanced parameterized quantum circuit",
        ha="center", va="center", fontsize=14, fontweight="bold",
    )
    ax.text(
        x_center, 2.25,
        r"$|\psi(\mathbf{x})\rangle = U_{\mathrm{PQC}}\,U_{\text{enc}}(\mathbf{x})\,"
        r"U_{\text{enh}}\,|0\rangle^{\otimes n_q}$"
        rf"$\quad(n_q={N_QUBITS})$",
        ha="center", va="center", fontsize=11.5,
    )

    # ---- bottom: data flow + notes ----
    flow_text = (
        rf"Data flow: input $(N,d)$ is split into $n_h={N_HEADS}$ heads as "
        rf"$[n_h,\,N,\,d_h]$; each $[N,d_h]$ "
        rf"($N={N_TOKENS},\ d_h={N_QUBITS}$) is processed row by row: "
        r"$[N,d_h]\rightarrow[N,d_h]$"
    )
    ax.text(x_center, y_bot - 2.25, flow_text,
            ha="center", va="top", fontsize=9.2, color="#222222")

    for i, (color, txt) in enumerate(notes):
        ax.text(x_wire_l - 1.35, y_bot - 2.95 - i * 0.42,
                rf"({i + 1})  {txt}",
                ha="left", va="top", fontsize=9.2,
                color=color, fontweight="bold")

    # ---- view limits ----
    ax.set_xlim(x_wire_l - 2.0, x_brace + 4.2)
    ax.set_ylim(y_bot - 4.4, 3.2)
    ax.set_aspect("equal")

    return fig


def save_figure(fig, name="fig2"):
    for ext, save_dir in SAVE_DIRS.items():
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.{ext}")
        fig.savefig(path)
        print(f"[OK] saved to: {path}")


def main():
    setup_style()
    fig = build_figure()
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
