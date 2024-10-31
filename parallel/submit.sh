#!/bin/bash

#SBATCH -t 04:00:00
#SBATCH -J tuning
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --cpus-per-task 2
#SBATCH --mem 36G
#SBATCH --qos low

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# module load Mambaforge/23.3.1-1-hpc1
# mamba activate freja_stable
origdir=$PWD
cd ../

runscript=$1
echo "running" $runscript
time python $runscript
cd $origdir

echo "done"
