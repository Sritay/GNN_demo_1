#!/bin/bash

#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --account=e760
#SBATCH --job-name=BubbleCNN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=03:00:00
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sritay.mistry@brunel.ac.uk
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt

# 1. Setup Environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores

module restore
module load PrgEnv-cray

# 2. Define Paths
WORK_DIR="$PWD"
SIF_IMAGE="/work/e760/e760/sritay/bubble_cnn/bubble_cnn.sif"

echo "Job started at $(date)"
echo "Running in: $WORK_DIR"
echo "Using container: $SIF_IMAGE"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: Singularity image not found!"
    exit 1
fi

# 3. Run the Container
# --pwd /app ensures the python script runs FROM inside the /app folder
# so it can find "dump_stable.lammpstrj" directly.
singularity exec \
    --bind "$WORK_DIR":/app \
    --pwd /app \
    "$SIF_IMAGE" \
    python3 predict.py

echo "Job finished at $(date)"
