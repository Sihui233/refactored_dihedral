#!/bin/bash
#SBATCH --job-name=scaling
#SBATCH --output=/home/mila/w/weis/scratch/DL/hypersearch/b%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=47Gb
#SBATCH --time=47:35:00
#SBATCH --gres=gpu:1
#SBATCH --constraint='turing|ampere|lovelace'

# polynomials_momentum.sh
# === Load common modules ===
module reset
# module load openmpi/4.0.4
# module load python/3.10
export CUDA_HOME=/cvmfs/ai.mila.quebec/apps/arch/common/cuda/11.8

module unload anaconda
module load openmpi/4.0.4
module load cudatoolkit/11.8
module load python/3.10
module load libffi
# === Use CUDA 12 for all nodes ===
# module load cudatoolkit/12.2.2
# export CUDA_HOME=/cvmfs/ai.mila.quebec/apps/arch/common/cuda/12.2.2
# echo "🔧 Using CUDA 12 environment (.env_cuda12)"
source /home/mila/w/weis/DL/.env/bin/activate

# === Set environment variables ===
# export LD_LIBRARY_PATH=$HOME/.local/lib:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# export CPATH=$CUDA_HOME/include:$HOME/.local/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$HOME/.local/include:$CPATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
# Don’t pre-grab all VRAM; leave space for autotuner / spikes
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# (optional) cap the client memory fraction if you still see fragmentation
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90

# Reduce autotuning memory (the log even suggested this)
export XLA_FLAGS="--xla_gpu_autotune_level=3"

# === Change to working directory ===
cd /home/mila/w/weis/DL/refactored_dihedral

# === Run your training script ===
python3 -m run.run_training_MLP "$@"
