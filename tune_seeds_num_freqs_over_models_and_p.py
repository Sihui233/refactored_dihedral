#!/usr/bin/env python3
import subprocess
import time

# Fixed hyperparameters
learning_rate = "0.00075"
weight_decay = "0.00001"

# Fixed arguments
optimizer = "adam"
epochs = "2500"
k = "59"
batch_experiment = "random_random"
num_neurons = "1024"
features = "128"

# Models and p values
# models = ["no_embed", "one_embed", "two_embed"]
models = ["no_embed"]
num_layers = "4" 
p_values = [59, 60, 61, 62, 63, 64, 65, 66]

# Seed/job settings
seeds_per_job = 5
total_seeds = 10
jobs_per_setting = total_seeds // seeds_per_job
starting_seed = 1
max_concurrent = 50  # adjust if needed

# Process list
processes = []

for model in models:
    for p in p_values:
        batch_size = str(p)
        print(f"\n--- Launching jobs for model = {model}, p = {p} ---\n")
        
        for job in range(jobs_per_setting):
            seeds = [str(starting_seed + job * seeds_per_job + i) for i in range(seeds_per_job)]

            job_args = [
                learning_rate,
                weight_decay,
                str(p),
                batch_size,
                optimizer,
                epochs,
                k,
                batch_experiment,
                num_neurons,
                model,
                features,
                num_layers  # ✅ Inserted before the seeds
            ] + seeds

            command = ["sbatch", "polynomials_momentum.sh"] + job_args
            print("Launching:", " ".join(command))

            proc = subprocess.Popen(command)
            processes.append(proc)

            # Throttle if too many jobs launched
            if len(processes) >= max_concurrent:
                for proc in processes:
                    proc.wait()
                processes = []

# Wait for remaining jobs
for proc in processes:
    proc.wait()

print("\n✅ All SLURM jobs launched.")
