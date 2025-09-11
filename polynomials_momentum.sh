#!/bin/bash
#SBATCH --job-name=b
#SBATCH --output=/home/mila/w/weis/scratch/DL/hypersearch/b%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30Gb
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1

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
# === Change to working directory ===
cd /home/mila/w/weis/DL/refactored_dihedral

# === Run your training script ===
python3 -m run.run_training_MLP "$@"
