#!/usr/bin/env python3
"""
CKA (Centered Kernel Alignment) drift analysis + Frozen probe AUC evaluation.

This script produces TWO key figures for the SEEK paper:
1. CKA similarity curve: how much the representation geometry drifts from base model
   across RL training rounds (r25, r50, r75) for ts_full and ts_nogate.
2. Frozen probe AUC curve: the r0-trained probe's performance on each round's data,
   alongside co-evolved (self-trained) probe AUC.

Usage:
    python analysis_cka_drift.py [--output_dir figures/]
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
from pathlib import Path


# ============================================================
# CKA computation (linear kernel, unbiased HSIC estimator)
# ============================================================

def centering_matrix(n):
    """H = I - (1/n) * 11^T"""
    return np.eye(n) - np.ones((n, n)) / n


def linear_CKA(X, Y):
    """
    Compute linear CKA between two representations X (n, p) and Y (n, q).
    Uses the Kornblith et al. (2019) formulation:
        CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    n = X.shape[0]
    assert Y.shape[0] == n, f"Sample size mismatch: {X.shape[0]} vs {Y.shape[0]}"

    # Cast to float32 to avoid float16 overflow in Frobenius norm
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Compute CKA
    XtX = X.T @ X  # (p, p)
    YtY = Y.T @ Y  # (q, q)
    XtY = X.T @ Y  # (p, q)

    # Frobenius norms
    numerator = np.linalg.norm(XtY, 'fro') ** 2
    denominator = np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')

    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


def rbf_CKA(X, Y, sigma=None):
    """
    CKA with RBF kernel (more sensitive to nonlinear geometry changes).
    Falls back to median heuristic for sigma if not provided.
    """
    from scipy.spatial.distance import pdist, squareform

    n = X.shape[0]

    def rbf_kernel(Z, s):
        dists = squareform(pdist(Z, 'sqeuclidean'))
        return np.exp(-dists / (2 * s ** 2))

    if sigma is None:
        sigma_x = np.median(pdist(X, 'euclidean'))
        sigma_y = np.median(pdist(Y, 'euclidean'))
        sigma = (sigma_x + sigma_y) / 2

    Kx = rbf_kernel(X, sigma)
    Ky = rbf_kernel(Y, sigma)

    # Center kernels
    H = centering_matrix(n)
    Kx_c = H @ Kx @ H
    Ky_c = H @ Ky @ H

    hsic = np.trace(Kx_c @ Ky_c) / ((n - 1) ** 2)
    norm_x = np.sqrt(np.trace(Kx_c @ Kx_c)) / (n - 1)
    norm_y = np.sqrt(np.trace(Ky_c @ Ky_c)) / (n - 1)

    if norm_x * norm_y < 1e-10:
        return 0.0
    return hsic / (norm_x * norm_y)


# ============================================================
# Data loading
# ============================================================

def load_base_representations(base_dir='dumps/base250b', prefix='nq', max_samples=1500):
    """Load and concatenate base model representations."""
    files = sorted(glob.glob(f'{base_dir}/{prefix}_*.npz'))
    arrays = {k: [] for k in ['Xc_21', 'Xa_21', 'Xp_21', 'Xc_24', 'Xa_24', 'Xp_24', 'Xc_27', 'Xa_27', 'Xp_27']}
    qi_list = []

    for f in files:
        d = np.load(f)
        for k in arrays:
            if k in d:
                arrays[k].append(d[k])
        # Load qi from corresponding jsonl
        jsonl_f = f.replace('.npz', '.jsonl')
        if os.path.exists(jsonl_f):
            with open(jsonl_f) as fp:
                qi_list.extend([json.loads(l)['qi'] for l in fp])

    result = {k: np.concatenate(v, axis=0) for k, v in arrays.items() if v}
    result['qi'] = np.array(qi_list) if qi_list else None

    # Subsample if too large
    n = result['Xc_21'].shape[0]
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, max_samples, replace=False)
        result = {k: v[idx] if isinstance(v, np.ndarray) else v for k, v in result.items()}

    print(f"  Base: loaded {result['Xc_21'].shape[0]} samples from {len(files)} files")
    return result


def load_round_representations(round_dir, max_samples=1500):
    """Load representations from a co-evolution round dump."""
    npz_file = os.path.join(round_dir, 'nq_3000.npz')
    if not os.path.exists(npz_file):
        return None

    d = np.load(npz_file)
    result = {k: d[k] for k in d.files}

    n = result['Xc_21'].shape[0]
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, max_samples, replace=False)
        result = {k: v[idx] if isinstance(v, np.ndarray) and v.shape[0] == n else v
                  for k, v in result.items()}

    print(f"  {round_dir}: loaded {result['Xc_21'].shape[0]} samples")
    return result


# ============================================================
# Frozen probe evaluation
# ============================================================

def load_probe(probe_path):
    """Load a trained probe (sklearn pipeline from joblib-pickled dict)."""
    import joblib
    data = joblib.load(probe_path)
    # data is a dict with keys: clf, position, layer, label, model, auc
    return data


def evaluate_probe_on_data(probe_dict, data, label_key='y_flip'):
    """Evaluate a probe on given data using the probe's own position/layer config.
    
    probe_dict: {'clf': Pipeline, 'position': str, 'layer': int, 'label': str, ...}
    data: dict with keys like Xa_21, Xc_24, y_flip, y_flip2, etc.
    """
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    pos = probe_dict['position']   # e.g. 'Xa'
    layer = probe_dict['layer']    # e.g. 21
    repr_key = f"{pos}_{layer}"    # e.g. 'Xa_21'
    
    if repr_key not in data:
        return {'auc': float('nan'), 'balanced_acc': float('nan'), 'repr_key': repr_key}
    
    X = data[repr_key].astype(np.float32)  # avoid float16 precision issues
    
    # Determine label
    lbl = probe_dict.get('label', 'flip')
    if lbl == 'flip' and 'y_flip' in data:
        y = (data['y_flip'] >= 0.5).astype(int)
    elif lbl == 'flip2' and 'y_flip2' in data:
        y = (data['y_flip2'] >= 0.5).astype(int)
    elif lbl == 'closedbook' and 'q_cb' in data:
        y = data['q_cb'].astype(int)
    else:
        y = (data.get('y_flip', np.zeros(X.shape[0])) >= 0.5).astype(int)
    
    # Skip if single class
    if len(np.unique(y)) < 2:
        return {'auc': 0.5, 'balanced_acc': 0.5, 'repr_key': repr_key}
    
    clf = probe_dict['clf']
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, 'decision_function'):
        y_prob = clf.decision_function(X)
    else:
        y_prob = clf.predict(X).astype(float)

    y_pred = (y_prob >= 0.5).astype(int)

    results = {'repr_key': repr_key}
    try:
        results['auc'] = roc_auc_score(y, y_prob)
    except ValueError:
        results['auc'] = 0.5

    results['balanced_acc'] = balanced_accuracy_score(y, y_pred)
    return results


# ============================================================
# Main analysis
# ============================================================

def mean_repr_per_gid(data, repr_key, gids, target_gids):
    """Compute mean representation per gid for alignment."""
    X = data[repr_key].astype(np.float32)
    result = []
    for g in target_gids:
        mask = (gids == g)
        if mask.sum() > 0:
            result.append(X[mask].mean(axis=0))
        else:
            result.append(np.zeros(X.shape[1], dtype=np.float32))
    return np.array(result)


def compute_cka_drift(base_data, round_data_dict, repr_keys=None):
    """
    Compute CKA between rounds using gid-aligned mean representations.
    This aligns the same queries across rounds to measure genuine drift
    rather than input distribution differences.
    
    Returns dict: {round_name: {repr_key: cka_value}}
    """
    if repr_keys is None:
        repr_keys = ['Xc_24']

    results = {}
    round_names = sorted(round_data_dict.keys())
    
    if len(round_names) < 2:
        print("    WARNING: Need ≥2 rounds for inter-round CKA")
        return results
    
    # Find shared gids across all rounds
    gid_sets = []
    for rname in round_names:
        rdata = round_data_dict[rname]
        if rdata is not None and 'gid' in rdata:
            gid_sets.append(set(rdata['gid']))
    
    if not gid_sets:
        print("    WARNING: No gid data found")
        return results
    
    shared_gids = sorted(set.intersection(*gid_sets))
    print(f"    Shared gids across rounds: {len(shared_gids)}")
    
    # Use first round (r25) as anchor
    anchor_name = round_names[0]
    anchor_data = round_data_dict[anchor_name]
    anchor_gids = anchor_data['gid']
    
    # CKA of anchor with itself = 1.0
    results[anchor_name] = {}
    for key in repr_keys:
        results[anchor_name][key] = 1.0
    print(f"    {anchor_name} (anchor): CKA = 1.0 by definition")
    
    # CKA of anchor with later rounds (gid-aligned)
    for round_name in round_names[1:]:
        rdata = round_data_dict[round_name]
        if rdata is None:
            continue
        results[round_name] = {}
        round_gids = rdata['gid']
        
        for key in repr_keys:
            if key not in anchor_data or key not in rdata:
                continue
            X_anchor = mean_repr_per_gid(anchor_data, key, anchor_gids, shared_gids)
            X_round = mean_repr_per_gid(rdata, key, round_gids, shared_gids)
            cka = linear_CKA(X_anchor, X_round)
            results[round_name][key] = cka
            print(f"    CKA({key}) {anchor_name} vs {round_name}: {cka:.4f}")

    return results


def compute_frozen_probe_auc(frozen_probe_dict, round_data_dict):
    """
    Evaluate frozen (round0) probe on each round's data.
    The probe's own position/layer config determines which repr to use.
    Returns dict: {round_name: {auc, balanced_acc, repr_key}}
    """
    results = {}
    for round_name, rdata in sorted(round_data_dict.items()):
        if rdata is None:
            continue
        metrics = evaluate_probe_on_data(frozen_probe_dict, rdata)
        results[round_name] = metrics
        print(f"    Frozen probe on {round_name}: AUC={metrics['auc']:.4f}, "
              f"BalAcc={metrics['balanced_acc']:.4f} (using {metrics['repr_key']})")

    return results


def compute_self_probe_auc(round_data_dict, probes_dir):
    """
    Evaluate each round's co-evolved probe on its own data (self-AUC).
    """
    results = {}
    for round_name, rdata in sorted(round_data_dict.items()):
        if rdata is None:
            continue
        # Find corresponding probe
        probe_path = os.path.join(probes_dir, round_name, 'commit_probe.pkl')
        if not os.path.exists(probe_path):
            print(f"    {round_name}: no co-evolved probe found at {probe_path}")
            continue

        probe_dict = load_probe(probe_path)
        metrics = evaluate_probe_on_data(probe_dict, rdata)
        results[round_name] = metrics
        print(f"    Self probe on {round_name}: AUC={metrics['auc']:.4f}, "
              f"BalAcc={metrics['balanced_acc']:.4f} (using {metrics['repr_key']}, "
              f"pos={probe_dict['position']}, L={probe_dict['layer']})")

    return results


def plot_results(cka_results, frozen_results, self_results, output_dir, experiment='ts_full'):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Set style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.figsize': (10, 4),
    })

    rounds = sorted(cka_results.keys())
    round_labels = [r.split('_')[-1] for r in rounds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel A: CKA drift ----
    repr_keys = list(next(iter(cka_results.values())).keys()) if cka_results else ['Xc_24']
    colors = {'Xc_21': '#2196F3', 'Xc_24': '#4CAF50', 'Xc_27': '#FF9800',
              'Xa_24': '#9C27B0', 'Xp_24': '#F44336'}
    key_labels = {'Xc_21': 'Commit L21', 'Xc_24': 'Commit L24', 'Xc_27': 'Commit L27',
                  'Xa_24': 'Answer L24', 'Xp_24': 'Predict L24'}

    for key in repr_keys:
        cka_vals = [cka_results[rname].get(key, np.nan) for rname in rounds]
        ax1.plot(range(len(cka_vals)), cka_vals, 'o-',
                 color=colors.get(key, '#333'), label=key_labels.get(key, key),
                 linewidth=2, markersize=8)

    ax1.set_xticks(range(len(round_labels)))
    ax1.set_xticklabels(round_labels)
    ax1.set_ylabel('CKA Similarity (anchored to r25)')
    ax1.set_xlabel('Training Round')
    ax1.set_title(f'Representation Drift ({experiment})')
    ax1.set_ylim(0.5, 1.05)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # ---- Panel B: Frozen vs Self probe AUC ----
    frozen_rounds = sorted(frozen_results.keys())
    frozen_auc = [frozen_results[r]['auc'] for r in frozen_rounds]
    frozen_labels = [r.split('_')[-1] for r in frozen_rounds]
    frozen_x = list(range(len(frozen_rounds)))

    ax2.plot(frozen_x, frozen_auc, 's-',
             color='#F44336', label='Frozen (r0) Probe', linewidth=2, markersize=8)

    if self_results:
        self_rounds = sorted(self_results.keys())
        self_auc = [self_results[r]['auc'] for r in self_rounds]
        self_x = [frozen_rounds.index(r) for r in self_rounds if r in frozen_rounds]
        ax2.plot(self_x, self_auc[:len(self_x)], 'D-',
                 color='#4CAF50', label='Co-evolved Probe (self)', linewidth=2, markersize=8)

    ax2.set_xticks(frozen_x)
    ax2.set_xticklabels(frozen_labels)
    ax2.set_ylabel('AUC-ROC')
    ax2.set_xlabel('Training Round')
    ax2.set_title(f'Probe Performance ({experiment})')
    ax2.set_ylim(0.7, 1.02)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=frozen_probe['auc'] if 'frozen_probe' in dir() else 0.824,
                color='orange', linestyle='--', alpha=0.5, label='Base AUC')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'cka_drift_{experiment}.pdf')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='figures/', help='Output directory for figures')
    parser.add_argument('--base_dir', default='dumps/base250b', help='Base model dump directory')
    parser.add_argument('--coevo_dir', default='dumps/coevo', help='Co-evolution dumps directory')
    parser.add_argument('--probes_dir', default='probes/coevo', help='Co-evolved probes directory')
    parser.add_argument('--base_probe', default='probes/base250b/commit_probe.pkl',
                        help='Frozen (base/round0) probe path')
    parser.add_argument('--experiments', nargs='+', default=['ts_full', 'ts_nogate'],
                        help='Experiments to analyze')
    parser.add_argument('--repr_keys', nargs='+',
                        default=['Xc_21', 'Xc_24', 'Xc_27'],
                        help='Representation keys to compute CKA for')
    parser.add_argument('--max_samples', type=int, default=1200,
                        help='Max samples per round for CKA (memory)')
    args = parser.parse_args()

    print("=" * 60)
    print("CKA Drift Analysis + Frozen Probe Evaluation")
    print("=" * 60)

    # Load base model representations
    print("\n[1] Loading base model representations...")
    base_data = load_base_representations(args.base_dir, max_samples=args.max_samples)

    # Load frozen probe
    print(f"\n[2] Loading frozen probe from {args.base_probe}...")
    if os.path.exists(args.base_probe):
        frozen_probe = load_probe(args.base_probe)
        print(f"  Probe: position={frozen_probe['position']}, layer={frozen_probe['layer']}, "
              f"label={frozen_probe['label']}, base_auc={frozen_probe['auc']:.4f}")
    else:
        print(f"  WARNING: Frozen probe not found at {args.base_probe}")
        frozen_probe = None

    # Process each experiment
    for exp in args.experiments:
        print(f"\n{'=' * 60}")
        print(f"  Experiment: {exp}")
        print(f"{'=' * 60}")

        # Find all rounds for this experiment
        round_dirs = sorted(glob.glob(os.path.join(args.coevo_dir, f'{exp}_r*')))
        if not round_dirs:
            print(f"  No round data found for {exp}")
            continue

        print(f"  Found rounds: {[os.path.basename(d) for d in round_dirs]}")

        # Load round data
        round_data = {}
        for rd in round_dirs:
            rname = os.path.basename(rd)
            print(f"\n  Loading {rname}...")
            round_data[rname] = load_round_representations(rd, max_samples=args.max_samples)

        # Compute CKA
        print(f"\n[3] Computing CKA drift for {exp}...")
        cka_results = compute_cka_drift(base_data, round_data, repr_keys=args.repr_keys)

        # Evaluate frozen probe
        frozen_results = {}
        if frozen_probe is not None:
            print(f"\n[4] Evaluating frozen probe on each round...")
            frozen_results = compute_frozen_probe_auc(frozen_probe, round_data)

        # Evaluate self (co-evolved) probes
        print(f"\n[5] Evaluating co-evolved probes (self-AUC)...")
        self_results = compute_self_probe_auc(round_data, args.probes_dir)

        # Also cross-evaluate: each round's probe on OTHER rounds' data (drift detection)
        print(f"\n[5b] Cross-round evaluation (r25 probe → r50/r75 data)...")
        cross_results = {}
        round_names_sorted = sorted(round_data.keys())
        if len(round_names_sorted) >= 2:
            first_round = round_names_sorted[0]
            first_probe_path = os.path.join(args.probes_dir, first_round, 'commit_probe.pkl')
            if os.path.exists(first_probe_path):
                first_probe = load_probe(first_probe_path)
                for rname in round_names_sorted[1:]:
                    if round_data[rname] is not None:
                        metrics = evaluate_probe_on_data(first_probe, round_data[rname])
                        cross_results[rname] = metrics
                        print(f"    {first_round} probe → {rname} data: AUC={metrics['auc']:.4f}")

        # Plot
        print(f"\n[6] Generating figures...")
        plot_results(cka_results, frozen_results, self_results, args.output_dir, experiment=exp)

        # Print summary table
        print(f"\n{'=' * 60}")
        print(f"  SUMMARY: {exp}")
        print(f"{'=' * 60}")
        print(f"  {'Round':<20} {'CKA(Xc_24)':<12} {'Frozen AUC':<12} {'Self AUC':<12}")
        print(f"  {'-'*56}")
        print(f"  {'base (r0)':<20} {'1.0000':<12} {'—':<12} {'—':<12}")
        for rname in sorted(round_data.keys()):
            cka_val = cka_results.get(rname, {}).get('Xc_24', float('nan'))
            frozen_val = frozen_results.get(rname, {}).get('auc', float('nan'))
            self_val = self_results.get(rname, {}).get('auc', float('nan'))
            print(f"  {rname:<20} {cka_val:<12.4f} {frozen_val:<12.4f} {self_val:<12.4f}")


if __name__ == '__main__':
    main()
