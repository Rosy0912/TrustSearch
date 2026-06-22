#!/usr/bin/env python3
"""
Matched read-out evolution table: full (probe-guided) vs baseline (EM-only).

For each checkpoint's scalars dump, compute the among-correct (EM=1) AUC of
separating grounded-correct (flip=1) from parametric-correct (flip=0) for every
read-out:
    entropy / answer-logprob / max-logprob   (competitor scalars, NOT trained)
    logit-probe / hidden-probe L21,24,27      (learned read-outs)

The clean, non-circular claim lives in the *scalar* rows: they are never a
training target, so if they stay ~0.5-0.65 across both arms and all steps,
competitor read-outs are structurally colour-blind. The probe rows for `full`
are confounded (full is trained toward the probe); only full-minus-baseline at
matched steps attributes a probe-AUC change to probe-guidance.
"""
import glob, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

LAYERS = [21, 24, 27]
STEPS = [25, 50, 75, 100]


def auc_scalar(s, y):
    s = np.asarray(s, np.float64); y = np.asarray(y, np.int8)
    if len(np.unique(y)) < 2:
        return float("nan")
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def auc_vec(X, y, seed=0):
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int8)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return float("nan")
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


def row(npz):
    d = np.load(npz, allow_pickle=True)
    ye, yf = d["y_em"], d["y_flip"]
    m = ye == 1
    yA = yf[m].astype(np.int8)
    nG, nP = int(yA.sum()), int((yA == 0).sum())
    out = {"n": int(m.sum()), "nG": nG, "nP": nP}
    out["entropy"] = auc_scalar(d["entropy"][m], yA) if "entropy" in d else float("nan")
    out["ans_lp"] = auc_scalar(d["ans_lp"][m], yA) if "ans_lp" in d else float("nan")
    out["max_lp"] = auc_scalar(d["max_lp"][m], yA) if "max_lp" in d else float("nan")
    out["logit"] = auc_vec(d["logit"][m], yA) if "logit" in d else float("nan")
    for L in LAYERS:
        k = f"Xc_{L}"
        out[f"probe_L{L}"] = auc_vec(d[k][m], yA) if k in d else float("nan")
    return out


def fmt(v):
    return "  nan " if v != v else f"{v:6.3f}"


def main():
    cols = ["n", "nG", "nP", "entropy", "ans_lp", "max_lp", "logit",
            "probe_L21", "probe_L24", "probe_L27"]
    for arm, tag in [("full", "full"), ("baseline", "baseem")]:
        print(f"\n{'='*92}\n[{arm}]  among-correct grounded-vs-parametric AUC by step\n{'='*92}")
        print(f"{'step':>5}" + "".join(f"{c:>10}" for c in cols))
        rows = {}
        for s in STEPS:
            f = f"dumps/scalars/{tag}_s{s}.npz"
            if not glob.glob(f):
                print(f"{s:>5}   (missing {f})"); continue
            r = row(f); rows[s] = r
            line = f"{s:>5}"
            for c in cols:
                v = r[c]
                line += f"{v:>10d}" if c in ("n", "nG", "nP") else f"{fmt(v):>10}"
            print(line)
        globals()[f"_{arm}"] = rows

    # full - baseline on the learned read-outs (attribution to probe-guidance)
    fr, br = globals().get("_full", {}), globals().get("_baseline", {})
    print(f"\n{'='*92}\n[full - baseline]  probe-AUC delta at matched steps "
          f"(>0 => probe-guidance raised separability beyond plain training)\n{'='*92}")
    keys = [f"probe_L{L}" for L in LAYERS] + ["logit"]
    print(f"{'step':>5}" + "".join(f"{k:>12}" for k in keys))
    for s in STEPS:
        if s in fr and s in br:
            line = f"{s:>5}"
            for k in keys:
                d = fr[s][k] - br[s][k]
                line += "        nan " if d != d else f"{d:>+12.3f}"
            print(line)


if __name__ == "__main__":
    main()
