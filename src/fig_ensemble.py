"""Figure: one test sample run through every tree of the boosted model, and the outputs summed.

    python src/fig_ensemble.py   # writes figures/tree_ensemble.{png,svg,caption.md}

Same model as fig_single_tree.py. Top: F0 + tree 1 + tree 2 + ... = F, each tree drawn as its real
structure with the sample's path. Bottom: a waterfall of all per-tree steps in log-odds.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from fig_single_tree import C, EXAMPLE, NU, fit, save
from golub import load_labels

TOWARD = {+1: C["AML"], -1: C["ALL"]}
START = "#898781"                      # neutral for F0, the start value


def sig(f):
    return 1 / (1 + np.exp(-f))


def mini_tree(ax, est, x_sample):
    """Draw one fitted tree as dots and lines, with the sample's path in ink and its leaf coloured."""
    t = est.tree_
    path = set(np.flatnonzero(est.decision_path(x_sample).toarray()[0]))
    leaf = int(est.apply(x_sample)[0])
    pos = {0: (0.0, 0.0)}

    def walk(i, depth):
        if t.feature[i] < 0:
            return
        x, y = pos[i]
        for child, sign in ((t.children_left[i], -1), (t.children_right[i], +1)):
            pos[child] = (x + sign * 0.5 / (depth + 1), y - 1)
            walk(child, depth + 1)

    walk(0, 0)
    for i in range(t.node_count):
        if t.feature[i] < 0:
            continue
        for child in (t.children_left[i], t.children_right[i]):
            hot = i in path and child in path
            ax.plot(*zip(pos[i], pos[child]), color=C["ink"] if hot else C["line"], lw=2.2 if hot else 1.0,
                    zorder=1, solid_capstyle="round")
    for i, (x, y) in pos.items():
        if i == leaf:
            colour = TOWARD[np.sign(t.value[i, 0, 0])]
            ax.scatter([x], [y], s=150, color=colour, edgecolor=C["surface"], linewidth=2, zorder=3)
        else:
            on = i in path
            ax.scatter([x], [y], s=55, color=C["node"], edgecolor=C["ink"] if on else C["line"],
                       linewidth=1.6 if on else 1.0, zorder=2)
    ax.set_xlim(-0.95, 0.95), ax.set_ylim(-2.45, 0.35), ax.axis("off")


def main():
    model, Xtr, ytr, Xte = fit()
    ex_class = load_labels().loc[EXAMPLE, "class"]
    x = Xte.loc[[EXAMPLE]]
    F0 = model._raw_predict_init(x.values).ravel()[0]
    F = np.array([s.ravel()[0] for s in model.staged_decision_function(x)])
    steps = np.diff(np.r_[F0, F])
    M = len(F)
    assert np.isclose(F0 + steps.sum(), model.decision_function(x)[0])
    cross = int(np.argmax(F > 0)) + 1
    n_up, n_down = int((steps > 0).sum()), int((steps < 0).sum())
    # Trees drawn in the top strip (None = ellipsis): the first three, the first that pushes toward
    # ALL, and the last.
    first_down = int(np.argmax(steps < 0)) + 1
    assert 4 < first_down < M - 1, first_down
    SHOW = [1, 2, 3, None, first_down, None, M]

    fig = plt.figure(figsize=(14, 9.6), facecolor=C["surface"])

    # ---- Key ------------------------------------------------------------------------------------
    kx, ky = 0.5, 9.6 - 0.33
    for colour, label in ((C["AML"], "tree pushes toward AML (νγ > 0)"),
                          (C["ALL"], "tree pushes toward ALL (νγ < 0)"),
                          (START, "start value F₀"), (C["ink"], "total F")):
        fig.patches.append(Rectangle((kx, ky - 0.1), 0.18, 0.2, transform=fig.dpi_scale_trans, fc=colour, ec="none"))
        fig.text(kx + 0.27, ky, label, fontsize=11.5, va="center", color=C["ink"], transform=fig.dpi_scale_trans)
        kx += 0.27 + 0.095 * len(label) + 0.5

    # ---- Top strip: the sum, tree by tree -----------------------------------------------------------
    strip = fig.add_axes([0.03, 0.66, 0.94, 0.25])
    strip.set_xlim(0, 100), strip.set_ylim(0, 10), strip.axis("off")
    widths = {"F0": 8, "tree": 9.5, "op": 3.0, "dots": 6.0, "F": 14}
    items = ["F0"]
    for m in SHOW:
        items += ["op+", "dots" if m is None else ("tree", m)]
    items += ["op=", "F"]
    # collapse "+ … +" so the ellipsis reads as one operator
    total = sum(widths["op"] if isinstance(it, str) and it.startswith("op") else
                widths[it if isinstance(it, str) else it[0]] for it in items)
    x0 = (100 - total) / 2
    for it in items:
        if isinstance(it, str) and it.startswith("op"):
            strip.text(x0 + widths["op"] / 2, 5.0, it[2], ha="center", va="center", fontsize=20, color=C["ink2"])
            x0 += widths["op"]
        elif it == "dots":
            strip.text(x0 + widths["dots"] / 2, 5.0, "…", ha="center", va="center", fontsize=22, color=C["ink2"])
            x0 += widths["dots"]
        elif it == "F0":
            w = widths["F0"]
            strip.text(x0 + w / 2, 8.6, "start", ha="center", fontsize=11, color=C["muted"])
            strip.text(x0 + w / 2, 5.0, r"$F_0$", ha="center", va="center", fontsize=20, color=C["ink"])
            strip.text(x0 + w / 2, 1.2, f"{F0:+.3f}", ha="center", fontsize=12, color=C["ink"], weight="bold")
            x0 += w
        elif it == "F":
            w = widths["F"]
            strip.text(x0 + w / 2, 6.3, rf"$F = {F[-1]:.3f}$", ha="center", va="center", fontsize=17,
                       color=C["ink"])
            strip.text(x0 + w / 2, 3.4, rf"$P(\mathrm{{AML}}) = {sig(F[-1]):.3f}$", ha="center", va="center",
                       fontsize=13, color=C["ink"])
            x0 += w
        else:
            m, w = it[1], widths["tree"]
            strip.text(x0 + w / 2, 8.6, f"tree {m}", ha="center", fontsize=11, color=C["muted"])
            inset = strip.inset_axes([x0 + 0.6, 2.6, w - 1.2, 5.6], transform=strip.transData)
            mini_tree(inset, model.estimators_[m - 1, 0], x.values)
            strip.text(x0 + w / 2, 1.2, f"{steps[m - 1]:+.3f}", ha="center", fontsize=12, color=C["ink"],
                       weight="bold")
            x0 += w

    # ---- Waterfall ----------------------------------------------------------------------------------
    ax = fig.add_axes([0.075, 0.07, 0.90, 0.53])
    ax.bar([0], [F0], bottom=0, width=0.8, color=START)
    lo = np.minimum(np.r_[F0, F[:-1]], F)
    ax.bar(np.arange(1, M + 1), np.abs(steps), bottom=lo, width=0.75,
           color=[TOWARD[int(np.sign(s))] for s in steps])
    ax.bar([M + 4], [F[-1]], bottom=0, width=1.6, color=C["ink"])
    ax.plot(np.arange(0, M + 1), np.r_[F0, F], drawstyle="steps-mid", color=C["ink2"], lw=0.6, alpha=0.5)

    ax.axhline(0, color=C["line"], lw=1.2, zorder=0)
    ax.text(M + 0.5, 0.12, "F = 0  ⇔  P(AML) = 0.5", ha="right", va="bottom", fontsize=10, color=C["ink2"])
    ax.set_xlim(-1.5, M + 5.5)
    ax.set_ylim(min(F0, 0) - 0.4, F[-1] + 0.3)
    ticks = [0] + list(range(10, M + 1, 10)) + [M + 4]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["F₀"] + [str(t) for t in ticks[1:-1]] + ["total"])
    ax.set_xlabel("tree", fontsize=11, color=C["ink2"])
    ax.set_ylabel("log-odds of AML, F", fontsize=11, color=C["ink2"])
    ax.grid(axis="y", color=C["hair"] if "hair" in C else "#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["line"])
    ax.tick_params(colors=C["ink2"], labelsize=10)
    ax.set_facecolor(C["surface"])

    save(fig, "tree_ensemble",
         f"**Prediction with the whole ensemble: test sample {EXAMPLE} ({ex_class}) through all {M} trees.** "
         f"Each tree routes the sample to one leaf and contributes νγ for that leaf (ν = {NU}); the "
         f"prediction is the start value F₀ = {F0:.3f} plus the sum of all {M} contributions. Top: the sum "
         "written out, with each tree drawn as its actual structure, the sample's path in black and the leaf "
         "it lands in coloured by the direction it pushes. Bottom: every step in log-odds as a waterfall; "
         f"{n_up} trees push toward AML and {n_down} toward ALL. The running total crosses F = 0 "
         f"(P = 0.5) at tree {cross} and ends at F = {F[-1]:.2f}, P(AML) = {sig(F[-1]):.3f}. Contributions "
         "shrink as the training samples become well fitted and their residuals fall.")
    print(f"F0={F0:.4f} F={F[-1]:.4f} cross={cross} up={n_up} down={n_down}")


if __name__ == "__main__":
    main()
