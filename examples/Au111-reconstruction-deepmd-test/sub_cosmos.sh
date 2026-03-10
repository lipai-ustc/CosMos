#!/bin/bash                 
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o %j.out                     
#SBATCH -e %j.err

# --- 1. 确保加载了 CUDA 库 (根据你集群的具体版本修改，如 cuda-11.x) ---
# 如果集群有 module 系统，取消下面注释并修改版本
# module load cuda/11.8

# --- 2. 检查当前环境下 GPU 是否可用 (调试用) ---
echo "Checking GPU status..."
nvidia-smi
python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"

# --- 3. 设置 DeepMD 并行参数 (防止 CPU 抢占) ---
export OMP_NUM_THREADS=4
export TF_INTRA_OP_PARALLELISM_THREADS=4
export TF_INTER_OP_PARALLELISM_THREADS=1

# --- 4. 运行程序 ---
python ../../cosmos_run.py
