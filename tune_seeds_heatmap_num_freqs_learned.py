#!/usr/bin/env python3
import subprocess
import time

# learning_rates = [0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 0.0005]
# learning_rates += [lr / 10 for lr in learning_rates]
# Define hyperparameter grids to sweep
# learning_rates = [0.0005, 0.00025, 0.0001, 0.000075, 0.00005]
# weight_decays = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 
#                  0.0000005, 0.0000001, 0.00000005, 0.00000001]

learning_rates = [0.0025, 0.001, 0.00075, 0.0005, 0.00025, 0.0001, 0.000075, 0.00005, 0.000025, 0.00001]
weight_decays = [
    0.001, 0.00075, 0.0005, 0.00025, 0.0001, 0.000075, 0.00005, 0.000025, 0.00001, 0.0000075]#,
#     0.000005, 0.0000025, 0.000001, 0.00000075, 0.0000005, 0.00000025, 0.0000001, 0.000000075, 0.00000005, 0.00000001
# ]

# Fixed arguments (common to all runs)
p = "18"
batch_size = "18"
optimizer = "adam"
epochs = "5000"
k = "58"
batch_experiment = "random_random"
num_neurons = "1024"
features = "128"

# Define the new hyperparameters to loop over.
# Note: num_layers can be passed as string or converted later as int in your training script.
num_layers_list = ["1", "2", "3"]#, "4", "5"]
mlp_classes = ["two_embed"]

# Seed/job settings
num_jobs = 1          # Jobs per (lr, wd, mlp_class, num_layers) combination.
seeds_per_job = 5
max_concurrent = 50
starting_seed = 1

# Process list
processes = []

# Loop over the new hyperparameters first.
for mlp_class in mlp_classes:
    for num_layers in num_layers_list:
        print(f"\n=== Launching jobs for MLP_class = {mlp_class}, num_layers = {num_layers} ===")
        # Now loop over learning rate and weight decay pairs.
        for lr in learning_rates:
            lr_str = f"{lr:.8f}"
            for wd in weight_decays:
                wd_str = f"{wd:.8f}"
                print(f"\n--- Launching jobs for learning_rate = {lr_str}, weight_decay = {wd_str} ---\n")
                for job in range(num_jobs):
                    # Generate seeds for this job.
                    seeds = [str(starting_seed + job * seeds_per_job + i) for i in range(seeds_per_job)]
    
                    # Prepare args for training script in the expected order.
                    job_args = [
                        lr_str,
                        wd_str,
                        p,
                        batch_size,
                        optimizer,
                        epochs,
                        k,
                        batch_experiment,
                        num_neurons,
                        mlp_class,
                        features,
                        num_layers
                    ] + seeds
    
                    # SLURM command
                    command = ["sbatch", "polynomials_momentum.sh"] + job_args
                    print("Launching:", " ".join(command))
    
                    # Launch job
                    proc = subprocess.Popen(command)
                    processes.append(proc)
    
                    # Throttle submissions if needed.
                    if len(processes) >= max_concurrent:
                        for proc in processes:
                            proc.wait()
                        processes = []

# Wait for any remaining processes.
for proc in processes:
    proc.wait()

print("\n✅ All (learning_rate, weight_decay, mlp_class, num_layers) sweep jobs launched.")
