#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=cg_openmm
#SBATCH --output=cg_openmm_%j.out

hostname
date

module load cuda/10.1

mpirun -np 1 python run_replica_exchange.py
srun python process_replica_exchange.py

