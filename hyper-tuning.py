#!/usr/bin/env python3
import itertools
import subprocess
import time
import os
import getpass
import random

# -----------------------------
# sweep grid
# -----------------------------
# LR_LIST = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
# WD_LIST = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
# P_LIST = [2 ** e for e in range(2, 13)]    # 2^2 ... 2^12
# NUM_LAYERS_LIST = [1, 2]
# SEEDS = list(range(10))     

# test
LR_LIST = [1e-5]
WD_LIST = [1e-3]
P_LIST = [2 ** e for e in range(4, 5)]
NUM_LAYERS_LIST = [2]
SEEDS = list(range(1))  

# -----------------------------
# global variables
# -----------------------------
OPTIMIZER = "adam"
EPOCHS = 500
BATCH_EXPERIMENT = "random_random"
MLP_CLASS = "two_embed"     
FEATURES = 128              

# -----------------------------
# Slurm/队列节流设置
# -----------------------------
SLURM_SCRIPT = "polynomials_momentum.sh"  # sbatch wrapper
QUEUE_CAP = 100            
SLEEP_WHEN_BUSY = 60       # seconds (too short)
MAX_SUBMIT_ATTEMPTS = 6    
INITIAL_BACKOFF = 30       
MAX_BACKOFF = 600          

SHUFFLE_COMBOS = True


def compute_num_neurons(p: int) -> int:
    """
    case 1) 8*(2*p)
    case 2) if 8*(2*p) > (2p)^2, take 2^x, where x is the smallest x where 2^(x+1) >= (2p)^2
           equiv to num_neurons = ceil_pow2((2p)^2) // 2
    """
    group_size = 2 * p
    full = group_size * group_size
    cand = 8 * group_size
    if cand <= full:
        return cand
    pow2 = 1
    while pow2 <= full:
        pow2 <<= 1
    return pow2 // 2

def compute_k(p: int) -> int:
    """
    k = floor(0.9*p*4)   if p < 2^4
        floor(0.8*p*4)   if 2^4 <= p < 2^8
        floor(0.6*p*4)   if p >= 2^8 
    """
    # if p < (1 << 4):
    #     frac = 0.9
    # elif p < (1 << 8):
    #     frac = 0.8
    # else:
    #     frac = 0.6
    frac = 0.9
    return int(frac * p * 4)

def batch_size_for_p(p: int) -> int:
    return p

def jobs_in_queue(user: str) -> int:
    """
    返回当前用户在 Slurm 队列中的作业数（RUNNING+PENDING）。
    """
    try:
        out = subprocess.check_output(
            ["squeue", "-u", user, "-h", "-o", "%i"], text=True
        ).strip()
        return 0 if not out else len(out.splitlines())
    except Exception:
        # 如果 squeue 不可用，就不做队列限流（返回 0）
        return 0

def submit_once(argv_args: list[str]) -> tuple[bool, str, str]:
    """
    调用 sbatch 提交一次；返回 (成功?, stdout, stderr)
    """
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
    """
    检测常见的“提交过多/限流”错误，触发退避睡眠。
    """
    s = stderr_text.lower()
    needles = [
        "too many jobs", "qosmaxsubmit", "limit", "account has too many jobs",
        "job submit", "queue full", "rate limit"
    ]
    return any(n in s for n in needles)

def main():
    user = os.environ.get("USER") or getpass.getuser()

    # 生成所有组合
    combos = list(itertools.product(LR_LIST, WD_LIST, P_LIST, NUM_LAYERS_LIST))
    if SHUFFLE_COMBOS:
        random.shuffle(combos)

    for lr, wd, p, num_layers in combos:
        # 依次计算该组合的派生参数
        bs = batch_size_for_p(p)
        k = compute_k(p)
        num_neurons = compute_num_neurons(p)
        seeds = [str(s) for s in SEEDS]

        # —— 准备 argv（严格按 Config.from_argv 的顺序）——
        argv_args = [
            f"{lr:.8g}",           # learning_rate
            f"{wd:.8g}",           # weight_decay
            str(p),                # p
            str(bs),               # batch_size
            OPTIMIZER,             # optimizer
            str(EPOCHS),           # epochs
            str(k),                # k
            BATCH_EXPERIMENT,      # batch_experiment
            str(num_neurons),      # num_neurons
            MLP_CLASS,             # MLP_class
            str(FEATURES),         # features
            str(num_layers),       # num_layers
        ] + seeds                  # random_seed_int_1 ...

        # —— 队列限流：如果队列太满，先睡 —— 
        while True:
            q = jobs_in_queue(user)
            if q >= QUEUE_CAP:
                print(f"[queue] jobs for {user} = {q} ≥ {QUEUE_CAP}; sleeping {SLEEP_WHEN_BUSY}s…")
                time.sleep(SLEEP_WHEN_BUSY)
            else:
                break

        # —— 提交 + 退避重试（遇到限流错误指数回退）——
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
                # 其它类型错误就不无限等，直接稍等后再试
                print(f"[retry] non-limit failure; sleeping {backoff}s…")
                time.sleep(backoff)
        else:
            print("❌ Giving up this combo after repeated failures.")

    print("\n✅ All sweep submissions attempted.")

if __name__ == "__main__":
    main()
