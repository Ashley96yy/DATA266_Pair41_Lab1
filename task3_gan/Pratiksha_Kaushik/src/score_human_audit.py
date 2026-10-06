"""Scores the blinded human audit once both raters have filled their sheets.

Inputs (outputs/human_audit/):
    rater1.csv, rater2.csv   sample_id, style, content, artifacts (1-5), notes
    _key.csv                 sample_id -> direction (hidden from raters)

Outputs:
    outputs/human_audit/audit_scores_per_sample.csv
    outputs/human_audit/audit_summary.csv   mean per rater, exact / within-1 agreement,
                                            Cohen's kappa (unweighted + quadratic)

Same calculation as section 16 of the v3 notebook.

Usage:
    python src/score_human_audit.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT = Path(__file__).resolve().parent.parent / "outputs" / "human_audit"
CRIT = ["style", "content", "artifacts"]


def cohen_kappa(a, b, weights=None, labels=(1, 2, 3, 4, 5)):
    k = len(labels)
    pos = {lab: i for i, lab in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[pos[x], pos[y]] += 1
    O /= O.sum()
    E = np.outer(O.sum(axis=1), O.sum(axis=0))
    i, j = np.indices((k, k))
    if weights == "quadratic":
        W = (i - j) ** 2 / (k - 1) ** 2
    else:
        W = (i != j).astype(float)
    denom = (W * E).sum()
    return float("nan") if denom == 0 else float(1 - (W * O).sum() / denom)


def main():
    r1 = pd.read_csv(AUDIT / "rater1.csv")
    r2 = pd.read_csv(AUDIT / "rater2.csv")
    key = pd.read_csv(AUDIT / "_key.csv")

    for name, r in (("rater1", r1), ("rater2", r2)):
        missing = [c for c in CRIT if pd.to_numeric(r[c], errors="coerce").isna().any()]
        if missing:
            raise SystemExit(f"{name}.csv is not complete yet (empty or non-numeric: {missing})")

    rows = []
    for c in CRIT:
        a, b = r1[c].astype(int).values, r2[c].astype(int).values
        assert set(a) | set(b) <= {1, 2, 3, 4, 5}, f"{c}: scores must be 1-5"
        rows.append(dict(
            criterion=c,
            mean_rater1=a.mean(), mean_rater2=b.mean(), mean_both=(a.mean() + b.mean()) / 2,
            exact_agreement=(a == b).mean(), within_1_agreement=(np.abs(a - b) <= 1).mean(),
            cohen_kappa=cohen_kappa(a, b), cohen_kappa_quadratic=cohen_kappa(a, b, "quadratic"),
        ))
    summary = pd.DataFrame(rows)

    per = r1[["sample_id"]].copy()
    for c in CRIT:
        per[f"{c}_r1"] = r1[c].astype(int)
        per[f"{c}_r2"] = r2[c].astype(int)
        per[f"{c}_mean"] = (per[f"{c}_r1"] + per[f"{c}_r2"]) / 2
    per = per.merge(key, on="sample_id")

    by_dir = per.groupby("direction")[[f"{c}_mean" for c in CRIT]].mean().round(2)
    overall = per[[f"{c}_mean" for c in CRIT]].values.mean()

    summary.to_csv(AUDIT / "audit_summary.csv", index=False)
    per.to_csv(AUDIT / "audit_scores_per_sample.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print("\nby direction:\n" + by_dir.to_string())
    print(f"\noverall human-audit score (mean of all criteria, both raters): {overall:.2f} / 5")


if __name__ == "__main__":
    main()
