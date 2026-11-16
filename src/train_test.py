"""Train a gradient-boosted ALL/AML classifier on the 38 training samples and test it on the 34
independent samples.

    python src/train_test.py                 # the default model: every gene available at each split
    python src/train_test.py --random-genes  # a random 84 genes at each split; the model in the figures
    python src/train_test.py --tuned         # tuned settings, chosen by cross-validation; 20 seeds

Prints training and test performance, every test sample's P(AML) next to the paper's own call,
and the genes the model relies on most. With more than one seed (--seeds), the model is refitted
with each seed and the results are summarised across them.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from golub import gene_descriptions, load, load_labels, paper_predictions

MODEL = dict(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=0)

# The random-genes model, drawn in the figures. A random 84 genes (sqrt of 7,071) are considered at
# each split, so that tree 1 is a readable depth-2 tree; with every gene, tree 1 is a single Zyxin split.
# min_impurity_decrease=1e-9 stops sklearn splitting pure nodes into twin leaves on float noise.
RANDOM_GENES_MODEL = MODEL | dict(max_features="sqrt", min_impurity_decrease=1e-9, random_state=203)

# The tuned model. Settings chosen by repeated 5-fold cross-validation on the training set only
# (96 combinations of depth, genes per split, row subsampling, learning rate and gene prefilter);
# the test set played no part. Stumps make the model a sum of single-gene votes, like the paper's
# predictor; a random 7 of the 50 prefiltered genes at each split spreads the evidence across genes.
TUNED_MODEL = dict(n_estimators=300, max_depth=1, learning_rate=0.05, max_features="sqrt", random_state=0)
TUNED_GENES = 50  # keep the 50 genes with the highest signal-to-noise first, as the paper did


class SignalToNoise(BaseEstimator, TransformerMixin):
    """Golub et al.'s gene filter: keep the k/2 genes most ALL-like and the k/2 most AML-like.

    The score is (mean_AML - mean_ALL) / (sd_AML + sd_ALL), computed from the samples passed to fit,
    so inside a pipeline it only ever sees the training data.
    """

    def __init__(self, k=50):
        self.k = k

    def fit(self, X, y):
        X, y = pd.DataFrame(X), np.asarray(y)
        aml, all_ = X[y == 1], X[y == 0]
        s2n = (aml.mean() - all_.mean()) / (aml.std(ddof=0) + all_.std(ddof=0) + 1e-12)
        order = s2n.sort_values()
        self.genes_ = list(order.index[: self.k // 2]) + list(order.index[-(self.k // 2):])
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.genes_]


def make_model(params, n_genes=None):
    gbt = GradientBoostingClassifier(**params)
    return Pipeline([("s2n", SignalToNoise(n_genes)), ("gbt", gbt)]) if n_genes else gbt


def fit(params=MODEL, n_genes=None):
    """Fit on the training set. Returns (model, X_train, y_train, X_test, y_test)."""
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    model = make_model(params, n_genes).fit(Xtr, ytr)
    return model, Xtr, ytr, Xte, yte


def importances(model, genes):
    """Per-gene importance (share of the total loss reduction), over all genes."""
    if isinstance(model, Pipeline):
        imp = pd.Series(model["gbt"].feature_importances_, index=model["s2n"].genes_)
        return imp.reindex(genes, fill_value=0.0)
    return pd.Series(model.feature_importances_, index=genes)


def report(name, P, y):
    """P: seeds x samples matrix of P(AML)."""
    errors = ((P >= 0.5) != np.asarray(y)).sum(1)
    ll = np.mean([log_loss(y, p, labels=[0, 1]) for p in P])
    auc = np.mean([roc_auc_score(y, p) for p in P])
    if len(P) == 1:
        pred = P[0] >= 0.5
        fn, fp = int((~pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        print(f"{name}: {len(y) - errors[0]}/{len(y)} correct, {errors[0]} errors "
              f"({fn} AML called ALL, {fp} ALL called AML); log-loss {ll:.3f}, ROC AUC {auc:.3f}")
    else:
        counts = pd.Series(errors).value_counts().sort_index()
        spread = ", ".join(f"{e} in {c}" for e, c in counts.items())
        print(f"{name}: errors per seed: {spread} (of {len(P)} seeds); "
              f"mean log-loss {ll:.3f}, mean ROC AUC {auc:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--random-genes", action="store_true",
                       help="consider a random 84 genes at each split (the model drawn in the figures)")
    which.add_argument("--tuned", action="store_true",
                       help="50-gene prefilter, then 300 stumps on a random 7 genes each (default 20 seeds)")
    ap.add_argument("--seeds", type=int, help="fit with this many seeds (default 20 with --tuned, else 1)")
    ap.add_argument("--top", type=int, default=10, help="number of genes to list (default 10)")
    args = ap.parse_args()

    params, n_genes = MODEL, None
    if args.random_genes:
        params = RANDOM_GENES_MODEL
    elif args.tuned:
        params, n_genes = TUNED_MODEL, TUNED_GENES
    n_seeds = args.seeds or (20 if args.tuned else 1)
    seed0 = params["random_state"]

    if n_genes:
        print(f"SignalToNoise(k={n_genes}), then")
    print("GradientBoostingClassifier(" + ", ".join(f"{k}={v!r}" for k, v in params.items()) + ")")
    if n_seeds > 1:
        print(f"refitted with random_state = {seed0} to {seed0 + n_seeds - 1}")
    print()

    Xtr, ytr = load("train")
    Xte, yte = load("test")
    Ptr, Pte, imps = [], [], []
    for s in range(n_seeds):
        model = make_model(params | dict(random_state=seed0 + s), n_genes).fit(Xtr, ytr)
        Ptr.append(model.predict_proba(Xtr)[:, 1])
        Pte.append(model.predict_proba(Xte)[:, 1])
        imps.append(importances(model, Xtr.columns))
    Ptr, Pte = np.array(Ptr), np.array(Pte)

    report("Training", Ptr, ytr.values)
    report("Test    ", Pte, yte.values)

    # ---- Per test sample, next to the paper's 50-gene predictor --------------------------------------
    labels, paper = load_labels(), paper_predictions()
    cls = np.array(["ALL", "AML"])
    wrong = ((Pte >= 0.5) != yte.values).mean(0)
    p = Pte.mean(0)
    t = pd.DataFrame({
        "actual": labels.loc[Xte.index, "class"],
        "source": labels.loc[Xte.index, "source"],
        "P(AML)": p.round(3),
        "GBT": cls[(p >= 0.5).astype(int)],
        "GBT ok": ["" if w == 0 else "WRONG" if n_seeds == 1 else f"wrong in {w:.0%}" for w in wrong],
        "paper": paper.loc[Xte.index, "predicted"],
        "strength": paper.loc[Xte.index, "strength"],
        "paper call": np.where(~paper.loc[Xte.index, "call"], "uncertain",
                               np.where(paper.loc[Xte.index, "predicted"] == labels.loc[Xte.index, "class"],
                                        "", "WRONG")),
    })
    where = f", mean over {n_seeds} seeds" if n_seeds > 1 else ""
    print(f"\nTest samples (P(AML){where}; paper = Golub et al.'s 50-gene predictor, "
          "which makes no call below strength 0.3):")
    print(t.to_string())
    called = paper.loc[Xte.index, "call"]
    n_right = int((t["paper call"] == "").sum())
    print(f"\nPaper: {int(called.sum())}/{len(t)} called, {n_right} of them correct; "
          f"{int((~called).sum())} left uncertain.")

    # ---- Genes the model uses ---------------------------------------------------------------------------
    imp = pd.concat(imps, axis=1).mean(axis=1)
    imp = imp[imp > 1e-6].sort_values(ascending=False)  # drop float-noise importances
    top = imp.head(args.top)
    desc = gene_descriptions()
    print(f"\nGenes that affect the output: {len(imp)}. Top {len(top)} by importance "
          f"(share of total loss reduction{where}):")
    for probe, v in top.items():
        print(f"  {v:6.3f}  {probe:<14} {desc.get(probe, '')[:60]}")


if __name__ == "__main__":
    main()
