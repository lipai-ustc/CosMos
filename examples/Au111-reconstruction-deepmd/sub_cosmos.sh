#!/bin/bash                 
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o %j.out                     
#SBATCH -e %j.err

#module purge
#module load gpumd/3.9.5
#module load gpumd/4.6.0

#nep
#gpumd >run.log 2>$1
python ../../cosmos_run.py
