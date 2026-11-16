"""Two figures of the first tree of a gradient-boosted ALL/AML classifier on the Golub data:

  tree_training    how the 38 training samples build the tree: class counts per node, leaf values γ,
                   with the start-value and leaf-value formulas below.
  tree_prediction  one held-out test sample dropped through the finished tree, with the model formula
                   and its worked prediction below.

    python src/fig_single_tree.py   # writes figures/tree_{training,prediction}.{png,svg,caption.md}

The model uses max_features="sqrt" so the first tree is a readable depth-2 tree. With every gene
available, the first split (Zyxin, X95735_at) separates all 38 training samples on its own and the
tree degenerates to a stump. min_impurity_decrease=1e-9 stops sklearn splitting already-pure nodes
into twin leaves with identical values (float-noise gains); 1e-6 was too high and left the late
trees, whose residuals are tiny, as single leaves.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.ensemble import GradientBoostingClassifier

from golub import load, load_labels

SEED = 203            # chosen so tree 1 has two mixed children and four leaves of n >= 3
EXAMPLE = 51          # test sample traced through the tree
NU = 0.1              # learning rate
OUT = Path(__file__).resolve().parent.parent / "figures"

# Gene symbols for the probes this seed's tree uses; any other probe is shown by its ID.
SYMBOLS = {"M26679_at": "HOXA5", "M29610_s_at": "GYPE", "D28473_s_at": "IARS"}

# Reference palette (dataviz skill): slots 1-2 for class identity, neutral inks for text.
C = dict(ALL="#2a78d6", AML="#eb6834", surface="#fcfcfb", node="#ffffff", ink="#0b0b0b",
         ink2="#52514e", muted="#898781", faint="#b9b7b0", line="#c3c2b7")

plt.rcParams.update({
    # Arial rather than Helvetica: matplotlib only finds the regular weight of the macOS Helvetica .ttc.
    "font.family": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})

# Geometry, in inches, shared by both figures so they line up when flipped between on a slide.
W, H_KEY, H_TREE, H_PANELS = 14.0, 0.65, 6.69, 2.94
SPLIT_BOX, LEAF_BOX = (23.0, 15.0), (21.0, 17.0)  # tree-axes units


def symbol(probe):
    return SYMBOLS.get(probe, probe)


# ---- model and numbers ---------------------------------------------------------------------------

def fit():
    Xtr, ytr = load("train")
    Xte, _ = load("test")
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=NU, max_features="sqrt",
        min_impurity_decrease=1e-9, random_state=SEED,
    ).fit(Xtr, ytr)
    return model, Xtr, ytr, Xte


def node_stats(est, Xtr, ytr, p0):
    """Per node: probe, threshold, children, n, ALL/AML counts, leaf value (checked by hand)."""
    t = est.tree_
    path = est.decision_path(Xtr.values).toarray().astype(bool)
    nodes = []
    for i in range(t.node_count):
        n, aml = int(path[:, i].sum()), int(ytr[path[:, i]].sum())
        leaf = t.feature[i] < 0
        gamma = (aml - n * p0) / (n * p0 * (1 - p0))  # Newton step on log-loss, first round
        if leaf:
            assert np.isclose(gamma, t.value[i, 0, 0]), (i, gamma, t.value[i, 0, 0])
        nodes.append(dict(
            i=i, leaf=leaf, probe=None if leaf else Xtr.columns[t.feature[i]],
            thr=t.threshold[i], left=t.children_left[i], right=t.children_right[i],
            n=n, all=n - aml, aml=aml, gamma=gamma,
        ))
    return nodes


def layout(nodes):
    """x, y for each node: root top-centre, children spread by depth."""
    pos = {0: (50.0, 77.0)}
    spread = {1: 25.0, 2: 12.5}

    def walk(i, depth):
        x, y = pos[i]
        nd = nodes[i]
        if nd["leaf"]:
            return
        for child, sign in ((nd["left"], -1), (nd["right"], +1)):
            pos[child] = (x + sign * spread[depth + 1], y - 21.5)
            walk(child, depth + 1)

    walk(0, 0)
    return pos


# ---- drawing -------------------------------------------------------------------------------------

def new_figure():
    """Figure with a key strip on top, the tree axes, and a panel axes underneath."""
    H = H_KEY + H_TREE + H_PANELS
    fig = plt.figure(figsize=(W, H), facecolor=C["surface"])
    tree_ax = fig.add_axes([0.02, H_PANELS / H, 0.96, H_TREE / H])
    tree_ax.set_xlim(0, 100), tree_ax.set_ylim(23, 87), tree_ax.axis("off")
    # Panel axes: x 0..1 across the width; y in units where 0.28 spans the panel strip height.
    panel_ax = fig.add_axes([0, 0, 1, H_PANELS / H])
    panel_ax.set_xlim(0, 1), panel_ax.set_ylim(0.02, 0.30), panel_ax.axis("off")
    return fig, tree_ax, panel_ax, H


def key(fig, H, items):
    """Key at the top-left, positioned in inches. items: ("swatch", label, colour) or ("line", label)."""
    x, y = 0.5, H - 0.33
    for kind, label, *rest in items:
        if kind == "swatch":
            fig.patches.append(Rectangle((x, y - 0.1), 0.18, 0.2, transform=fig.dpi_scale_trans,
                                         fc=rest[0], ec="none"))
        else:
            fig.lines.append(plt.Line2D([x, x + 0.22], [y, y], transform=fig.dpi_scale_trans,
                                        color=C["ink"], lw=2.6, solid_capstyle="round"))
        fig.text(x + 0.27, y, label, fontsize=12, va="center", color=C["ink"], transform=fig.dpi_scale_trans)
        x += 0.27 + 0.11 * len(label) + 0.45


def class_bar(ax, cx, y, width, n_all, n_aml):
    """Horizontal stacked bar of ALL vs AML counts, with a small surface gap between segments."""
    h, gap = 1.7, 0.35
    segs = [s for s in ((n_all, C["ALL"]), (n_aml, C["AML"])) if s[0] > 0]
    usable = width - gap * (len(segs) - 1)
    x = cx - width / 2
    for count, colour in segs:
        w = usable * count / (n_all + n_aml)
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.45",
                                    fc=colour, ec="none", mutation_aspect=1.6, zorder=3))
        x += w + gap


def box(ax, x, y, w, h, border, lw):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                mutation_aspect=1.6, fc=C["node"], ec=border, lw=lw, zorder=2))


def txt(ax, x, y, s, size, color, bold=False):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, color=color,
            weight="bold" if bold else "normal", zorder=3)


def draw_edges(ax, nodes, pos, on_path=None):
    """Parent-to-child lines with yes/no threshold labels; on_path edges drawn bold, the rest faded."""
    for nd in nodes:
        if nd["leaf"]:
            continue
        px, py = pos[nd["i"]]
        for child, op in ((nd["left"], "≤"), (nd["right"], ">")):
            cx, cy = pos[child]
            hot = on_path is not None and nd["i"] in on_path and child in on_path
            faded = on_path is not None and not hot
            y_from = py - SPLIT_BOX[1] / 2
            y_to = cy + (LEAF_BOX if nodes[child]["leaf"] else SPLIT_BOX)[1] / 2
            ax.plot([px, cx], [y_from, y_to], color=C["ink"] if hot else C["line"],
                    lw=2.6 if hot else 1.2, zorder=1, solid_capstyle="round")
            mx, my = (px + cx) / 2, (y_from + y_to) / 2
            word = "yes" if op == "≤" else "no"
            ax.text(mx + (-1.2 if op == "≤" else 1.2), my, f"{word}: {op} {nd['thr']:g}",
                    ha="right" if op == "≤" else "left", va="center", fontsize=10,
                    color=C["ink"] if hot else (C["faint"] if faded else C["ink2"]),
                    weight="bold" if hot else "normal", zorder=3,
                    bbox=dict(fc=C["surface"], ec="none", pad=1.5))


def draw_training_node(ax, nd, x, y):
    w, h = LEAF_BOX if nd["leaf"] else SPLIT_BOX
    box(ax, x, y, w, h, C["line"], 1.0)
    top = y + h / 2
    counts = f"n = {nd['n']}  ·  {nd['all']} ALL / {nd['aml']} AML"
    if not nd["leaf"]:
        txt(ax, x, top - 3.0, symbol(nd["probe"]), 14, C["ink"], bold=True)
        txt(ax, x, top - 6.6, f"expression ≤ {nd['thr']:g} ?", 11.5, C["ink"])
        class_bar(ax, x, top - 9.9, w * 0.78, nd["all"], nd["aml"])
        txt(ax, x, top - 12.4, counts, 9, C["ink2"])
    else:
        txt(ax, x, top - 2.4, "Leaf", 10, C["muted"])
        class_bar(ax, x, top - 5.0, w * 0.78, nd["all"], nd["aml"])
        txt(ax, x, top - 7.3, counts, 9, C["ink2"])
        txt(ax, x, top - 10.4, rf"$\gamma$ = {nd['gamma']:+.3f}", 13, C["ink"], bold=True)
        txt(ax, x, top - 13.6, rf"adds $\nu\gamma$ = {NU * nd['gamma']:+.3f}", 10, C["ink2"])
        txt(ax, x, top - 15.6, "to the log-odds of AML", 8.5, C["muted"])


def draw_prediction_node(ax, nd, x, y, visited, value):
    w, h = LEAF_BOX if nd["leaf"] else SPLIT_BOX
    box(ax, x, y, w, h, C["ink"] if visited else C["line"], 2.0 if visited else 1.0)
    top = y + h / 2
    main, sub = (C["ink"], C["ink2"]) if visited else (C["faint"], C["faint"])
    if not nd["leaf"]:
        txt(ax, x, top - 3.0, symbol(nd["probe"]), 14, main, bold=True)
        txt(ax, x, top - 6.6, f"expression ≤ {nd['thr']:g} ?", 11.5, main)
        if visited:
            answer = "yes" if value <= nd["thr"] else "no"
            txt(ax, x, top - 10.6, f"sample {EXAMPLE}: {value:g}  →  {answer}", 12, C["ink"], bold=True)
    else:
        txt(ax, x, top - 2.6, "Leaf", 10, C["muted"] if visited else C["faint"])
        txt(ax, x, top - 7.0, rf"$\gamma$ = {nd['gamma']:+.3f}", 13, main, bold=True)
        txt(ax, x, top - 11.0, rf"adds $\nu\gamma$ = {NU * nd['gamma']:+.3f}", 10, sub)
        txt(ax, x, top - 13.4, "to the log-odds of AML", 8.5, C["muted"] if visited else C["faint"])


def panels(ax, specs):
    """Side-by-side formula panels. specs: list of (x0, width, heading, [(y, text, size, colour), ...])."""
    top, bottom = 0.285, 0.035
    for x0, w, heading, lines in specs:
        ax.add_patch(FancyBboxPatch((x0, bottom), w - 0.015, top - bottom,
                                    mutation_aspect=(0.28 / H_PANELS) * W,
                                    boxstyle="round,pad=0,rounding_size=0.008", fc=C["node"],
                                    ec=C["line"], lw=1.0))
        ax.text(x0 + 0.015, top - 0.03, heading, fontsize=13, weight="bold", color=C["ink"])
        for y, s, size, colour in lines:
            ax.text(x0 + 0.015, y, s, fontsize=size, color=colour, va="center")


def save(fig, name, caption):
    """Write <name>.png/.svg (no title in the image) and <name>.caption.md beside them."""
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, facecolor=C["surface"], bbox_inches="tight", pad_inches=0.2)
    (OUT / f"{name}.caption.md").write_text(caption + "\n")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---- figures -------------------------------------------------------------------------------------

def main():
    model, Xtr, ytr, Xte = fit()
    labels = load_labels()
    est = model.estimators_[0, 0]
    n_aml, n_all = int(ytr.sum()), int(len(ytr) - ytr.sum())
    p0 = ytr.mean()
    F0 = np.log(p0 / (1 - p0))
    assert np.isclose(F0, model._raw_predict_init(Xtr.iloc[:1].values)[0, 0])
    nodes = node_stats(est, Xtr, ytr, p0)
    pos = layout(nodes)

    # Example test sample.
    x_ex = Xte.loc[[EXAMPLE]].values
    on_path = set(np.flatnonzero(est.decision_path(x_ex).toarray()[0]))
    ex_leaf = int(est.apply(x_ex)[0])
    gamma_ex = nodes[ex_leaf]["gamma"]
    F1 = F0 + NU * gamma_ex
    assert np.isclose(F1, next(model.staged_decision_function(Xte.loc[[EXAMPLE]]))[0])
    F_all = model.decision_function(Xte.loc[[EXAMPLE]])[0]
    sig = lambda f: 1 / (1 + np.exp(-f))
    ex_class = labels.loc[EXAMPLE, "class"]
    path_splits = [nodes[i] for i in sorted(on_path) if not nodes[i]["leaf"]]

    # Leaf used for the worked leaf-value example: the most mixed one, so both terms matter.
    lf = max((nd for nd in nodes if nd["leaf"]), key=lambda nd: min(nd["all"], nd["aml"]))

    # ---- Figure 1: training ----------------------------------------------------------------------
    fig, tree_ax, panel_ax, H = new_figure()
    key(fig, H, [("swatch", "ALL", C["ALL"]), ("swatch", "AML", C["AML"])])
    draw_edges(tree_ax, nodes, pos)
    for nd in nodes:
        draw_training_node(tree_ax, nd, *pos[nd["i"]])
    panels(panel_ax, [
        (0.035, 0.465, "Start value: the same for every sample", [
            (0.205, rf"$F_0 = \ln\dfrac{{n_{{\mathrm{{AML}}}}}}{{n_{{\mathrm{{ALL}}}}}}"
                    rf" = \ln\dfrac{{{n_aml}}}{{{n_all}}} = {F0:.4f}$", 15, C["ink"]),
            (0.125, rf"$p_0 = \dfrac{{1}}{{1+e^{{-F_0}}}} = {p0:.4f}$", 15, C["ink"]),
            (0.058, "The log-odds of AML before any tree, from the training class balance.", 10, C["ink2"]),
        ]),
        (0.515, 0.45, "Leaf value: the leaf's errors, converted to log-odds", [
            (0.205, r"$\gamma_{\mathrm{leaf}} = \dfrac{\sum\,(y_i - p_0)}{\sum\,p_0(1-p_0)}"
                    r" = \dfrac{n_{\mathrm{AML}} - n\,p_0}{n\,p_0(1-p_0)}$", 15, C["ink"]),
            (0.125, rf"e.g. the {lf['all']} ALL / {lf['aml']} AML leaf:  "
                    rf"$\dfrac{{{lf['aml']} - {lf['n']}\times{p0:.4f}}}"
                    rf"{{{lf['n']}\times{p0:.4f}\times{1 - p0:.4f}}} = {lf['gamma']:+.4f}$", 13, C["ink2"]),
            (0.058, r"Sums run over the training samples in the leaf; $y_i$ = 1 for AML, 0 for ALL.",
             10, C["ink2"]),
        ]),
    ])
    save(fig, "tree_training",
         "**Training: the first tree of a gradient-boosted ALL/AML classifier.** "
         f"Tree 1 of {model.n_estimators}, depth 2, fitted to the {len(ytr)} training samples "
         f"({n_all} ALL, {n_aml} AML) with learning rate ν = {NU} and a random subset of genes considered "
         f"at each split (seed {SEED}). Each box shows how the training samples divide; each leaf shows the "
         "value γ the tree outputs for any sample that lands there, and the step νγ it adds to the log-odds "
         "of AML. Below: the start value F₀ shared by every sample, and the leaf-value formula worked for "
         f"the {lf['n']}-sample mixed leaf.")

    # ---- Figure 2: prediction --------------------------------------------------------------------
    fig, tree_ax, panel_ax, H = new_figure()
    key(fig, H, [("line", f"path of test sample {EXAMPLE} ({ex_class})")])
    draw_edges(tree_ax, nodes, pos, on_path)
    for nd in nodes:
        value = None if nd["leaf"] else Xte.loc[EXAMPLE, nd["probe"]]
        draw_prediction_node(tree_ax, nd, *pos[nd["i"]], visited=nd["i"] in on_path, value=value)
    panels(panel_ax, [
        (0.035, 0.465, "The model", [
            (0.200, r"$F(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x)$", 15, C["ink"]),
            (0.120, r"$P(\mathrm{AML} \mid x) = \dfrac{1}{1 + e^{-F(x)}}$", 15, C["ink"]),
            (0.058, f"F is the log-odds of AML; each tree $h_m$ adds its leaf's γ, scaled by ν = {NU}. "
                    f"M = {model.n_estimators} trees.", 10, C["ink2"]),
        ]),
        (0.515, 0.45, f"Test sample {EXAMPLE} ({ex_class}), not used in training", [
            (0.205, rf"After tree 1:   $F_1 = {F0:.4f} + {NU}\times{gamma_ex:.4f} = {F1:.4f}$", 13, C["ink"]),
            (0.150, rf"$P(\mathrm{{AML}}) = 1/(1+e^{{{-F1:.4f}}}) = {sig(F1):.3f}$", 13, C["ink"]),
            (0.095, rf"After all {model.n_estimators} trees:   $F = {F_all:.2f}$,   "
                    rf"$P(\mathrm{{AML}}) = {sig(F_all):.3f}$", 13, C["ink"]),
            (0.050, f"One tree moves it only from {p0:.3f} to {sig(F1):.3f}; the ensemble does the work.",
             10, C["ink2"]),
        ]),
    ])
    steps = "; ".join(
        f"its {symbol(nd['probe'])} expression ({Xte.loc[EXAMPLE, nd['probe']]:g}) is "
        f"{'at most' if Xte.loc[EXAMPLE, nd['probe']] <= nd['thr'] else 'above'} {nd['thr']:g}"
        for nd in path_splits)
    save(fig, "tree_prediction",
         f"**Prediction: test sample {EXAMPLE} ({ex_class}) dropped through the trained tree.** "
         f"The sample was not used in training. At each split the tree checks one gene: {steps}. "
         f"It lands in the leaf with γ = {gamma_ex:.3f}, which adds νγ = {NU * gamma_ex:.3f} to the "
         "starting log-odds. Faded boxes are not visited. Below: the model formula, and the sample's "
         f"probability of AML after tree 1 and after all {model.n_estimators} trees.")

    print(f"F0={F0:.4f} leaf={ex_leaf} gamma={gamma_ex:.4f} F1={F1:.4f} p1={sig(F1):.4f} "
          f"F100={F_all:.4f} p100={sig(F_all):.4f}")


if __name__ == "__main__":
    main()
