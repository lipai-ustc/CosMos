#!/bin/bash                 
#SBATCH --partition=node2
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --output=%j.out                     

#mpirun -n $SLURM_NTASKS vasp_std
python ../../cosmos_run.py
