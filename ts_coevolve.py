"""TrustSearch co-evolution controller.

Watches each probe-using experiment's checkpoints. When a new policy checkpoint
appears (every save_freq steps), it:
  1. dumps fresh rollouts USING THAT checkpoint (so hidden states are the CURRENT
     policy's own commit-token states) + recomputes flip / closed-book labels,
  2. refits the commit + boundary probes,
  3. atomically updates that experiment's probe manifest -> the probe server hot-reloads
     both the encoder (= current checkpoint) and the probe heads.
This is the策略<->探针 co-evolution loop. It also logs the closed-book correctness
(cb_rate) per round on a FIXED held-out question set = the knowledge-boundary migration
curve (the self-evolution evidence).

Fault-tolerant: if a dump/train fails, the manifest is left unchanged (server keeps the
previous round); training never blocks on the controller.
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, time

PY = "python"


def latest_ckpt(exp, after):
    """Return (step, path) of the newest complete actor checkpoint with step>after."""
    best = (after, None)
    for d in glob.glob(f"verl_checkpoints/{exp}/actor/global_step_*"):
        if not os.path.exists(os.path.join(d, "config.json")):
            continue
        try:
            step = int(d.rsplit("_", 1)[1])
        except ValueError:
            continue
        if step > best[0]:
            best = (step, d)
    return best


def run(cmd, env=None):
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


def refit(exp, step, ckpt, args, env):
    tag = f"{exp}_r{step}"
    dump_dir = f"dumps/coevo/{tag}"
    probe_dir = f"probes/coevo/{tag}"
    os.makedirs(dump_dir, exist_ok=True)
    # 1) dump fresh rollouts on a FIXED held-out slice (comparable cb across rounds)
    rc = run([PY, "ts_probe_dump.py", "--model", ckpt, "--data", "data/nq_search/train.parquet",
              "--tag", "nq", "--cap", str(args.nq), "--offset", str(args.holdout_offset),
              "--G", "5", "--layers", "21,24,27", "--gpu_mem", "0.55", "--out", dump_dir], env)
    if rc != 0:
        print(f"  [refit] dump failed for {tag}", flush=True); return False
    # 2) refit probes
    rc = run([PY, "ts_probe_train.py", "--dumps", dump_dir, "--layers", "21,24,27",
              "--save", probe_dir], env)
    if rc != 0:
        print(f"  [refit] train failed for {tag}", flush=True); return False
    # 3) atomically update manifest -> server hot-reloads
    manifest = f"probes/live_{exp}/manifest.json"
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    payload = {"round": step, "encoder": ckpt,
               "commit": os.path.abspath(os.path.join(probe_dir, "commit_probe.pkl")),
               "boundary": os.path.abspath(os.path.join(probe_dir, "boundary_probe.pkl"))}
    tmp = manifest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, manifest)
    # 4) log knowledge-boundary migration (cb_rate on the fixed slice)
    try:
        import numpy as np
        cbs = [float(np.load(p, allow_pickle=True)["q_cb"].mean())
               for p in glob.glob(os.path.join(dump_dir, "*.npz"))
               if len(np.load(p, allow_pickle=True)["q_cb"])]
        cb = float(np.mean(cbs)) if cbs else float("nan")
    except Exception:
        cb = float("nan")
    with open("logs/boundary_migration.csv", "a") as f:
        f.write(f"{int(time.time())},{exp},{step},{cb:.4f}\n")
    print(f"  [refit] {tag} done -> manifest updated, boundary cb_rate={cb:.3f}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default="full,nogate,nocost")
    ap.add_argument("--nq", type=int, default=300)
    ap.add_argument("--holdout_offset", type=int, default=3000,
                    help="use a slice NOT used to fit the round-0 probe")
    ap.add_argument("--retriever", default="http://192.168.102.21:8000/retrieve")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    exps = ["ts_" + e for e in args.exps.split(",")]

    env = dict(os.environ)
    env["RETRIEVER_URL"] = args.retriever
    env["NO_PROXY"] = "192.168.0.0/16,127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)

    last = {e: 0 for e in exps}
    print(f"[coevolve] watching {exps} (poll={args.poll}s, nq={args.nq})", flush=True)
    while True:
        for e in exps:
            step, ckpt = latest_ckpt(e, last[e])
            if ckpt is not None:
                print(f"[coevolve] new checkpoint {e} step={step} -> refit", flush=True)
                if refit(e, step, ckpt, args, env):
                    last[e] = step
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
