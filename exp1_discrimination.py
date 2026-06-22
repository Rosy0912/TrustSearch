#!/usr/bin/env python3
"""
Exp1 — Discrimination: can we tell GROUNDED-correct from PARAMETRIC-correct?

Thesis under test (the load-bearing claim of the paper):
  Hallucination info is NOT absent from the output distribution, it is *discarded*
  by scalar read-outs (entropy / answer-logprob / max-logprob). A learned probe on
  the commit-token hidden state recovers the discarded "is the answer grounded in
  evidence?" dimension.

Task A (primary):  among CORRECT answers (EM=1), separate
    grounded-correct (flip=1: answer changes when retrieved evidence is removed)
    vs parametric-correct (flip=0: answer survives -> memorised / guessed).
  A reward that only sees EM is colour-blind here; we ask which *read-out* sees it.

Read-outs compared (all on the SAME rollouts):
  (1) entropy           — ARPO-style scalar          [needs --gen scalars]
  (2) answer-logprob    — IGPO-style scalar           [needs --gen scalars]
  (3) max-logprob       — confidence scalar           [needs --gen scalars]
  (4) logit-probe       — learned read-out on W.h     [needs --gen scalars]
  (5) hidden-probe @L   — learned read-out on h_L     [from existing dumps]

Modes:
  --from-dumps DIR...   pool existing ts_probe_dump .npz (Xc_{L}, y_em, y_flip) and
                        report hidden-probe AUC. Memory-aware (one layer at a time).
  --scalars FILE.npz    optional scalars dump (entropy/maxlp/anslp/logit + y_em/y_flip)
                        produced by ts_probe_scalars.py; adds read-outs (1)-(4).
"""
import argparse, glob, gc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def cv_auc_vec(X, y, C=1.0, seed=0):
    """5-fold out-of-fold ROC-AUC for a (possibly high-dim) feature matrix."""
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int8)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


def cv_auc_scalar(s, y):
    """AUC of a single scalar read-out (direction-agnostic: take max(auc, 1-auc))."""
    s = np.asarray(s, np.float64)
    y = np.asarray(y, np.int8)
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def load_labels(files):
    ye, yf, idx = [], [], []
    off = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "y_flip" not in d or "Xc_21" not in d:
            continue
        n = len(d["y_em"])
        if d["Xc_21"].shape[0] != n:
            continue
        ye.append(d["y_em"]); yf.append(d["y_flip"])
        idx.append((f, off, off + n)); off += n
    return np.concatenate(ye), np.concatenate(yf), idx, off


def load_layer(files, L, total):
    """Stream one layer's commit-token hidden states into a single float32 matrix."""
    X = None; pos = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "y_flip" not in d or f"Xc_{L}" not in d:
            continue
        n = len(d["y_em"])
        if d[f"Xc_{L}"].shape[0] != n:
            continue
        x = d[f"Xc_{L}"].astype(np.float32)
        if X is None:
            X = np.empty((total, x.shape[1]), np.float32)
        X[pos:pos + n] = x; pos += n
        del x, d; gc.collect()
    return X[:pos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dumps", nargs="+", default=["dumps/base250b"],
                    help="dirs of ts_probe_dump .npz files")
    ap.add_argument("--layers", default="21,24,27")
    ap.add_argument("--scalars", default="", help="optional scalars .npz from ts_probe_scalars.py")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    files = []
    for d in args.from_dumps:
        files += sorted(glob.glob(d.rstrip("/") + "/*.npz"))
    print(f"[exp1] pooling {len(files)} dumps from {args.from_dumps}")
    ye, yf, _, total = load_labels(files)
    print(f"[exp1] rollouts={total}  EM=1={int(ye.sum())}  flip=1={int(yf.sum())}")

    mA = ye == 1
    nG, nP = int(yf[mA].sum()), int((yf[mA] == 0).sum())
    print(f"\n{'='*64}\n[TASK A] grounded-correct vs parametric-correct (EM=1, n={int(mA.sum())})"
          f"\n         grounded(flip=1)={nG}  parametric(flip=0)={nP}\n{'='*64}")
    print(f"{'read-out':<26}{'AUC':>8}")

    # ============================================================
    # ALIGNED table: all read-outs on the SAME rollouts (scalars file)
    # ============================================================
    if args.scalars:
        s = np.load(args.scalars, allow_pickle=True)
        syf, sye = s["y_flip"], s["y_em"]
        sm = sye == 1; sA = syf[sm]
        print(f"\n[ALIGNED on {args.scalars}]  among-correct n={int(sm.sum())} "
              f"grounded={int(sA.sum())} parametric={int((sA==0).sum())}")
        def rep(name, key, kind):
            if key not in s:
                print(f"{name:<26}{'N/A':>8}"); return
            v = s[key][sm]
            auc = cv_auc_scalar(v, sA) if kind == "scalar" else cv_auc_vec(v, sA)
            print(f"{name:<26}{auc:>8.3f}")
        rep("(1) entropy",        "entropy",   "scalar")
        rep("(2) answer-logprob", "ans_lp",    "scalar")
        rep("(3) max-logprob",    "max_lp",    "scalar")
        rep("(4) logit-probe",    "logit",     "vec")
        for L in layers:
            if f"Xc_{L}" in s:
                rep(f"(5) hidden-probe L{L}", f"Xc_{L}", "vec")
        print()

    # ---- hidden-probe (5): large-sample corroboration from existing dumps ----
    if nG >= 10 and nP >= 10:
        print("[large-sample hidden-probe from dumps]")
        for L in layers:
            X = load_layer(files, L, total)
            auc = cv_auc_vec(X[mA], yf[mA])
            print(f"(5) hidden-probe L{L:<13}{auc:>8.3f}")
            del X; gc.collect()
    else:
        print("(5) hidden-probe: too few per class for Task A")

    # ---- reference: full-set grounded discrimination + EM-prediction ----
    print(f"\n[ref] full-set grounded(flip) discrimination (all n={total}):")
    for L in layers:
        X = load_layer(files, L, total)
        print(f"      hidden-probe L{L}: AUC={cv_auc_vec(X, yf):.3f}")
        del X; gc.collect()


if __name__ == "__main__":
    main()
