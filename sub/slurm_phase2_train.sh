#!/bin/bash
#SBATCH --job-name=purrsight_phase2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/phase2_train_%j.out
#SBATCH --error=logs/slurm/phase2_train_%j.err

# Purr-Sight Phase 2 Training - SLURM Job Script
# 
# This script launches Phase 2 distributed training on a SLURM-managed GPU cluster.
#
# Usage:
#   sbatch sub/slurm_phase2_train.sh
#
# To customize resources:
#   sbatch --nodes=2 --gpus-per-node=8 sub/slurm_phase2_train.sh
#
# To monitor:
#   squeue -u $USER
#   tail -f logs/slurm/phase2_train_<jobid>.out

set -e
set -u

echo "=========================================="
echo "Purr-Sight Phase 2 Training - SLURM Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Create log directory
mkdir -p logs/slurm

# Load required modules (adjust for your cluster)
# Uncomment and modify as needed for your cluster
# module load cuda/11.8
# module load cudnn/8.6
# module load nccl/2.15
# module load python/3.10

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path as needed
conda activate purrsight

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# NCCL settings for multi-node training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_ASYNC_ERROR_HANDLING=1

# Get master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# Configuration file
CONFIG_FILE="config/phase2_cluster.yaml"

echo "=========================================="
echo "Starting distributed training..."
echo "=========================================="

# Launch training with torchrun
# torchrun will automatically set RANK, LOCAL_RANK, WORLD_SIZE
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train/train_llm/train_phase2.py \
    --config $CONFIG_FILE \
    --num-gpus $SLURM_GPUS_PER_NODE

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "=========================================="

# Print job statistics
if command -v sacct &> /dev/null; then
    echo "Job Statistics:"
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
fi

exit $EXIT_CODE
