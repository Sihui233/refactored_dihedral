#!/usr/bin/env python3
import itertools
import subprocess
import time
import os
import getpass
import random
from functools import partial

# -----------------------------
# sweep grid
# -----------------------------
LR_LIST = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
WD_LIST = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

# Match your current big P sweep
P_LIST = [2 ** e for e in range(2, 10)]   # p ∈ {512, 1024, 2048}
NUM_MLP_LAYERS_LIST = [1]              # transformer MLP layers inside each block
SEEDS = [2]                    # tweak as you like

# -----------------------------
# global variables
# -----------------------------
OPTIMIZER = "adam"
EPOCHS = 700
BATCH_EXPERIMENT = "random_random"

# Transformer specifics (constants or narrow sweeps)
NUM_HEADS = 4                 # <-- fixed by your rule
N_CTX = 2                     # keep as in your example; change if needed
ACT_TYPE = "ReLU"
ATTN_COEFF = 1.0
NN_MULTIPLIER = 8

# -----------------------------
# Slurm settings
# -----------------------------
SLURM_SCRIPT = "polynomials_momentum_trans.sh"  # sbatch wrapper
QUEUE_CAP = 1000
SLEEP_WHEN_BUSY = 60
MAX_SUBMIT_ATTEMPTS = 6
INITIAL_BACKOFF = 30
MAX_BACKOFF = 600

SHUFFLE_COMBOS = False

# -----------------------------
# helpers
# -----------------------------
def ceil_pow2(x: int) -> int:
    """Smallest power of two >= x."""
    if x <= 1:
        return 1
    p = 1
    while p < x:
        p <<= 1
    return p

def compute_group_size(p: int) -> int:
    return 2 * p

def compute_k(p: int) -> int:
    """
    Keep your MLP heuristic unless you want to change it for Transformers:
    k = 2*p when p < 16, else floor(0.9*2*p).
    """
    if p < (1 << 4):
        frac = 1.0
    else:
        frac = 0.9
    return int(frac * p * 2)

def batch_size_for_p(p: int) -> int:
    return 2 * p

def compute_transformer_dims(p: int) -> tuple[int, int, int]:
    """
    Enforce:
      - d_model = next power of two >= |G|, where |G| = 2*p
      - num_heads = 4 (fixed)
      - d_head = d_model // 4
    Ensure d_model >= 4 so it's divisible by 4.
    """
    group_size = compute_group_size(p)
    d_model = max(4, ceil_pow2(group_size))
    d_head = d_model // NUM_HEADS
    return d_model, d_head, NUM_HEADS

def jobs_in_queue(user: str) -> int:
    try:
        out = subprocess.check_output(
            ["squeue", "-u", user, "-h", "-o", "%i"], text=True
        ).strip()
        return 0 if not out else len(out.splitlines())
    except Exception:
        return 0

def submit_once(argv_args: list[str]) -> tuple[bool, str, str]:
    cmd = ["sbatch", SLURM_SCRIPT] + argv_args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    ok = (proc.returncode == 0)
    print(("Launched" if ok else "Submit failed") + ":", " ".join(cmd))
    if out.strip():
        print(out.strip())
    if err.strip():
        print(err.strip())
    return ok, out, err

def should_backoff_on_error(stderr_text: str) -> bool:
    s = stderr_text.lower()
    needles = [
        "too many jobs", "qosmaxsubmit", "limit", "account has too many jobs",
        "job submit", "queue full", "rate limit"
    ]
    return any(n in s for n in needles)

# -----------------------------
# main
# -----------------------------
def main():
    user = os.environ.get("USER") or getpass.getuser()

    # (lr, wd, p, num_mlp_layers)
    combos = list(itertools.product(LR_LIST, WD_LIST, P_LIST, NUM_MLP_LAYERS_LIST))
    if SHUFFLE_COMBOS:
        random.shuffle(combos)

    for lr, wd, p, num_mlp_layers in combos:
        # Derived params
        bs = batch_size_for_p(p)
        k_val = compute_k(p)  # keep your k semantics
        d_model, d_head, num_heads = compute_transformer_dims(p)

        seeds = [str(s) for s in SEEDS]

        # argv order must match controllers/config_Transformer.py::Config.from_argv
        argv_args = [
            f"{lr:.8g}",                  # learning_rate
            f"{wd:.8g}",                  # weight_decay
            str(p),                       # p
            str(bs),                      # batch_size
            OPTIMIZER,                    # optimizer
            str(EPOCHS),                  # epochs
            str(k_val),                   # k
            BATCH_EXPERIMENT,             # batch_experiment
            str(d_model),                 # d_model
            str(d_head),                  # d_head
            str(num_heads),               # num_heads (fixed 4)
            str(N_CTX),                   # n_ctx
            ACT_TYPE,                     # act_type
            f"{ATTN_COEFF:.8g}",          # attn_coeff
            str(NN_MULTIPLIER),           # nn_multiplier
            str(num_mlp_layers),          # num_mlp_layers
        ] + seeds                         # seeds...

        # Queue throttle
        while True:
            q = jobs_in_queue(user)
            if q >= QUEUE_CAP:
                print(f"[queue] jobs for {user} = {q} ≥ {QUEUE_CAP}; sleeping {SLEEP_WHEN_BUSY}s…")
                time.sleep(SLEEP_WHEN_BUSY)
            else:
                break

        # Submit with exponential backoff on rate/limit errors
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_SUBMIT_ATTEMPTS + 1):
            ok, _out, err = submit_once(argv_args)
            if ok:
                break
            if should_backoff_on_error(err):
                print(f"[backoff] attempt {attempt}/{MAX_SUBMIT_ATTEMPTS}; sleeping {backoff}s…")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                print(f"[retry] non-limit failure; sleeping {backoff}s…")
                time.sleep(backoff)
        else:
            print("❌ Giving up this combo after repeated failures.")

    print("\n✅ All Transformer sweep submissions attempted.")

if __name__ == "__main__":
    main()
