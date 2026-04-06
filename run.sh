#!/bin/bash

#SBATCH --job-name=death
#SBATCH --output=/home/c01toja/CISPA-projects/training_dynamics-2024/SAFEgptPrune/safe-torch/logs/job-%j.out
#SBATCH --gres=gpu:A100:4
#SBATCH --partition=xe8545
#SBATCH --time=1000

JOBDATADIR=`ws create work --space "$SLURM_JOB_ID" --duration "7 00:00:00"`
JOBTMPDIR=/tmp/job-"$SLURM_JOB_ID"

srun mkdir "$JOBTMPDIR"
srun mkdir -p "$JOBDATADIR" "$JOBTMPDIR"

srun --container-image=projects.cispa.saarland:5005#c01adga/ffcv-imagenet:v6 --container-mounts="$JOBTMPDIR":/tmp bash /home/c01toja/CISPA-projects/training_dynamics-2024/SAFEgptPrune/safe-torch/"$1" "$2" "$3" "$4" "$5"

