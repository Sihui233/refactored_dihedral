#!/bin/bash
#SBATCH --job-name=b
#SBATCH --output=/home/mila/m/moisescg/scratch/slurm_logs/b%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30Gb
#SBATCH --time=5:30:00
#SBATCH --gres=gpu:1 

# === Load common modules ===
module reset
module load openmpi/4.0.4
module load python/3.10

# === Use CUDA 12 for all nodes ===
module load cudatoolkit/12.2.2
export CUDA_HOME=/cvmfs/ai.mila.quebec/apps/arch/common/cuda/12.2.2
echo "🔧 Using CUDA 12 environment (.env_cuda12)"
source /home/mila/m/moisescg/curvature-optimizer/.env_cuda12/bin/activate

# === Set environment variables ===
export LD_LIBRARY_PATH=$HOME/.local/lib:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$HOME/.local/include:$CPATH

# === Change to working directory ===
cd /home/mila/m/moisescg/scratch/bias_vs_nobias

# === Run your training script ===
python /home/mila/m/moisescg/group-training-final/train_mlp_multilayer_dirichlet_bs_metrics_added.py "$@"
