#!/bin/bash
#SBATCH -A m4474                  
#SBATCH --output=job_output_%j.txt
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

# echo "Loading PyTorch 2.1.0..."
# module purge
# module load pytorch/2.1.0-cu12

srun --mpi=pmi2 shifter --image=docker:vmikuni/tensorflow:ngc-23.12-tf2-v1 python scripts/test.py

echo "done!"
