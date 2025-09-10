#!/usr/bin/env python
import subprocess
import time

# Fixed arguments (these will remain the same for every job)
fixed_args = [
    "0.0005",           # learning_rate
    "0.0001",           # weight_decay #was 0.0006 before I did one-hot mlp 1 layer
    "18",               # p
    "18",               # batch_size
    "adam",             # optimizer
    "3500",             # epochs
    "59",               # k
    "random_random",    # batch_experiment
    "512",              # num_neurons
    "two_embed",        # MLP_class
    "128",              # features
    "1"                 # num_layers
]

num_jobs = 50       # Launch 50 jobs (each with 10 seeds)
seeds_per_job = 20  # Each job gets 10 seeds
max_concurrent = 50  # Run 50 jobs concurrently

# Set the starting seed (change this value as needed)
starting_seed = 1

processes = []  # List to hold current batch processes

for job in range(num_jobs):
    # Create the list of seeds for this job.
    seeds = [str(starting_seed + job * seeds_per_job + i) for i in range(seeds_per_job)]
    
    # Combine fixed args with the seeds.
    job_args = fixed_args + seeds
    
    # Build the command. The SLURM script is assumed to be in the same directory.
    command = ["sbatch", "polynomials_momentum.sh"] + job_args
    
    print("Launching job:", " ".join(command))
    
    # Launch the job asynchronously.
    proc = subprocess.Popen(command)
    processes.append(proc)
    
    # When we have max_concurrent processes, wait for them all to finish.
    if len(processes) == max_concurrent:
        for p in processes:
            p.wait()
        processes = []  # Reset the list for the next batch

# Wait for any remaining processes.
for p in processes:
    p.wait()

print("All jobs launched.")
