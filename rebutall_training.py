#!/usr/bin/env python3
import subprocess

# Base command without seeds.
# base_cmd = (
#     "0.0001 0.00005 59 59 adam "
#     "5000 58 random_random 512 1 3072 0.0 0.0"
# )
# base_cmd = (
#     "0.00075 0.0001 66 66 adam "
#     "2001 50 random_random 8 1 3300 0.0 0.0 1"
# )
# attention 1.0 optimal hyperparameters are below
base_cmd = (
    "0.00075 0.000025 59 59 adam "
    "5000 58 random_random 8 1 3072 0.0 0.0 3"
)
# attention 0.0 optimal hyperparameters are below
# base_cmd = (
#     "0.00025 0.000001 59 59 adam "
#     "6000 45 random_random 8 1 3072 0.0 0.0 1"
# )

n_jobs = 25          # Total number of jobs to submit
seeds_per_job = 1   # Number of seeds per job
first_seed = 8
for job in range(n_jobs):
    # Calculate the starting seed for this job.
    start_seed = job * seeds_per_job + first_seed
    # Create a space separated list of seeds.
    seeds = " ".join(str(seed) for seed in range(start_seed, start_seed + seeds_per_job))
    
    # Build the full command to be submitted via sbatch.
    # Assuming training.sh accepts the command to run as an argument.
    # full_cmd = f"sbatch polynomials_momentum_2.sh {base_cmd} {seeds}"
    full_cmd = f"sbatch polynomials_momentum_paper_2.sh {base_cmd} {seeds}"

    print(f"Submitting job {job+1}/{n_jobs}: {full_cmd}")
    # Submit the job
    subprocess.run(full_cmd, shell=True)
