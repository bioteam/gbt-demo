"""Load the Golub et al. (1999) ALL/AML data from data/raw/ as sample x probe matrices."""

import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
FILES = {"train": "data_set_ALL_AML_train.txt", "test": "data_set_ALL_AML_independent.txt"}


def load_labels() -> pd.DataFrame:
    """Per-sample labels (sample number 1-72) parsed from table_ALL_AML_samples.txt."""
    rows = []
    for line in (RAW / "table_ALL_AML_samples.txt").read_text().splitlines():
        m = re.match(r"^\s*(\d+)\s+(ALL|AML)\s+(BM|PB)\b(.*)$", line)
        if not m:
            continue
        n, cls, tissue, rest = int(m[1]), m[2], m[3], m[4]
        cell = re.search(r"\b([BT])-cell\b", rest)
        source = rest.split()[-1]
        rows.append({"sample": n, "class": cls, "tissue": tissue,
                     "cell": cell[1] if cell else None, "source": source})
    labels = pd.DataFrame(rows).set_index("sample").sort_index()
    labels["split"] = ["train" if n <= 38 else "test" for n in labels.index]
    return labels


def load_expression(split: str, drop_controls: bool = True) -> pd.DataFrame:
    """Expression matrix for 'train' or 'test': rows = samples (sorted), columns = probe IDs."""
    # Data rows end with a trailing tab; index_col=False stops pandas shifting the columns.
    df = pd.read_csv(RAW / FILES[split], sep="\t", index_col=False)
    df = df.set_index("Gene Accession Number")
    keep = [c for c in df.columns if c.strip().isdigit()]
    X = df[keep].T
    X.index = X.index.astype(int)
    X.index.name = "sample"
    X.columns.name = "probe"
    if drop_controls:
        X = X.loc[:, ~X.columns.str.startswith("AFFX")]
    return X.sort_index()


def gene_descriptions() -> pd.Series:
    """Probe ID -> gene description, from the training file."""
    df = pd.read_csv(RAW / FILES["train"], sep="\t", index_col=False,
                     usecols=["Gene Description", "Gene Accession Number"])
    return df.set_index("Gene Accession Number")["Gene Description"]


def load(split: str):
    """(X, y) for a split, with y = 1 for AML and 0 for ALL."""
    X = load_expression(split)
    y = (load_labels().loc[X.index, "class"] == "AML").astype(int)
    return X, y
