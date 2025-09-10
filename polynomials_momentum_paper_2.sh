#!/bin/bash
#SBATCH --job-name=b
#SBATCH --output=/home/mila/m/moisescg/scratch/slurm_logs/b%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80Gb
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint='turing|ampere|lovelace|hopper'


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
python /home/mila/m/moisescg/rebuttals_iclr_2024/refactored_polynomials_transformer_make_r2_heatmap_attn=1_top-k_layer_all_quant_metrics_margins_dirichlet.py "$@"
#python /home/mila/m/moisescg/rebuttals_iclr_2024/refactored_polynomials_transformer_make_r2_heatmap_attn=0_top-k_layer_all_quant_metrics_margins_dirichlet.py "$@"

