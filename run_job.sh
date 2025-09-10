#!/bin/bash
#SBATCH --job-name=graphing
#SBATCH --output=logs/%x_%j.out  
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=40G
#SBATCH --time=24:00:00                 

# ============ 环境配置 ============
source /home/mila/w/weis/DL/init.sh 

# ============ 运行命令 ============
# python3 paper_plots.py
./input.sh