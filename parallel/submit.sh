#!/bin/bash

#SBATCH -t 04:00:00
#SBATCH -J tuning
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --cpus-per-task 4
#SBATCH --mem 36G
#SBATCH --qos low

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

module load buildtool-easybuild/4.9.0-hpc434151c3d \
       GCCcore/11.3.0 \
       FFmpeg/4.4.2

module load Mambaforge/23.3.1-1-hpc1
mamba activate freja_stable

origdir=$PWD
cd ../

runscript=$1
echo "running" $runscript
time python $runscript
cd $origdir

echo "done"
