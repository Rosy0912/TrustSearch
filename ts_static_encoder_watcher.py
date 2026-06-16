"""Simple encoder-only manifest updater for ts_static experiment.

Watches verl_checkpoints/ts_static/actor/ for new global_step_* dirs.
When found, updates probes/live_ts_static/manifest.json with the new encoder path
but keeps probe paths unchanged (frozen at r0).

This ensures the probe server uses the CURRENT policy's hidden states
but the ORIGINAL (r0) probe heads for scoring.
"""
import argparse, glob, json, os, time

def latest_ckpt(ckpt_root):
    pattern = os.path.join(ckpt_root, "global_step_*")
    dirs = sorted(glob.glob(pattern), key=lambda d: int(d.split("_")[-1]))
    if not dirs:
        return None, None
    last = dirs[-1]
    step = int(last.split("_")[-1])
    return step, last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="verl_checkpoints/ts_static/actor")
    ap.add_argument("--manifest", default="probes/live_ts_static/manifest.json")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    last_step = -1
    print(f"[static-watcher] watching {args.ckpt_root} (poll={args.poll}s)", flush=True)

    while True:
        step, ckpt = latest_ckpt(args.ckpt_root)
        if step is not None and step > last_step:
            # Read current manifest to preserve probe paths
            if os.path.exists(args.manifest):
                with open(args.manifest) as f:
                    m = json.load(f)
            else:
                m = {}
            m["round"] = step
            m["encoder"] = ckpt
            # Keep frozen probes
            m.setdefault("commit", "probes/base250b/commit_probe.pkl")
            m.setdefault("boundary", "probes/base250b/boundary_probe.pkl")

            with open(args.manifest + ".tmp", "w") as f:
                json.dump(m, f)
            os.replace(args.manifest + ".tmp", args.manifest)
            print(f"[static-watcher] step {step} -> encoder={ckpt}", flush=True)
            last_step = step
        time.sleep(args.poll)

if __name__ == "__main__":
    main()
