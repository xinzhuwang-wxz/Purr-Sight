#!/bin/bash
# Purr-Sight Phase 2 Distributed Training Script
# 
# This script sets up and launches distributed training for Phase 2 multi-modal LLM training
# using PyTorch's torchrun launcher with Distributed Data Parallel (DDP).
#
# Requirements satisfied:
# - 6.1: Initialize DDP with correct world size and rank
# - 6.2: Wrap model with DistributedDataParallel  
# - 6.3: Configure data loader with DistributedSampler
# - 6.4: Synchronize gradients across all processes
# - 6.5: Use environment variables for distributed configuration
#
# Usage:
#   ./cluster_train.sh [config_path] [num_nodes] [node_rank] [master_addr] [master_port]
#
# Examples:
#   # Single node, 4 GPUs
#   ./cluster_train.sh config/train_config.yaml
#   
#   # Multi-node: Node 0 (master)
#   ./cluster_train.sh config/train_config.yaml 2 0 192.168.1.100 29500
#   
#   # Multi-node: Node 1 (worker)  
#   ./cluster_train.sh config/train_config.yaml 2 1 192.168.1.100 29500

set -e  # Exit on any error
set -u  # Exit on undefined variables

# =============================================================================
# Configuration and Default Values
# =============================================================================

# Script metadata
SCRIPT_NAME="cluster_train.sh"
SCRIPT_VERSION="1.0.0"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Default values
DEFAULT_CONFIG="config/train_config.yaml"
DEFAULT_NUM_NODES=1
DEFAULT_NODE_RANK=0
DEFAULT_MASTER_ADDR="localhost"
DEFAULT_MASTER_PORT=29500
DEFAULT_NPROC_PER_NODE="auto"  # Auto-detect GPUs per node
DEFAULT_BACKEND="nccl"
DEFAULT_LOG_LEVEL="INFO"

# Parse command line arguments
CONFIG_PATH=${1:-$DEFAULT_CONFIG}
NUM_NODES=${2:-$DEFAULT_NUM_NODES}
NODE_RANK=${3:-$DEFAULT_NODE_RANK}
MASTER_ADDR=${4:-$DEFAULT_MASTER_ADDR}
MASTER_PORT=${5:-$DEFAULT_MASTER_PORT}

# Additional configuration
NPROC_PER_NODE=${NPROC_PER_NODE:-$DEFAULT_NPROC_PER_NODE}
BACKEND=${BACKEND:-$DEFAULT_BACKEND}
LOG_LEVEL=${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}

# Derived values (will be calculated later after validation)
WORLD_SIZE=0

# Logging configuration
LOG_DIR="logs/distributed_training"
LOG_FILE="$LOG_DIR/cluster_train_${TIMESTAMP}_node${NODE_RANK}.log"

# =============================================================================
# Utility Functions
# =============================================================================

# Logging functions
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" | tee -a "$LOG_FILE" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE" >&2
}

log_debug() {
    if [ "$LOG_LEVEL" = "DEBUG" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEBUG] $*" | tee -a "$LOG_FILE"
    fi
}

# Print usage information
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [config_path] [num_nodes] [node_rank] [master_addr] [master_port]

Arguments:
  config_path   Path to training configuration YAML file (default: $DEFAULT_CONFIG)
  num_nodes     Total number of nodes in the cluster (default: $DEFAULT_NUM_NODES)
  node_rank     Rank of this node (0-based, default: $DEFAULT_NODE_RANK)
  master_addr   IP address of the master node (default: $DEFAULT_MASTER_ADDR)
  master_port   Port for distributed communication (default: $DEFAULT_MASTER_PORT)

Environment Variables:
  NPROC_PER_NODE    Number of processes (GPUs) per node (default: auto-detect)
  BACKEND           Distributed backend (default: $DEFAULT_BACKEND)
  LOG_LEVEL         Logging level: INFO, DEBUG (default: $DEFAULT_LOG_LEVEL)
  CONDA_ENV         Conda environment name (default: purrsight)

Examples:
  # Single node with 4 GPUs
  $SCRIPT_NAME config/train_config.yaml

  # Multi-node cluster (2 nodes, 4 GPUs each)
  # On master node (192.168.1.100):
  $SCRIPT_NAME config/train_config.yaml 2 0 192.168.1.100 29500
  
  # On worker node:
  $SCRIPT_NAME config/train_config.yaml 2 1 192.168.1.100 29500

  # Custom GPU count per node
  NPROC_PER_NODE=8 $SCRIPT_NAME config/train_config.yaml

  # Debug mode
  LOG_LEVEL=DEBUG $SCRIPT_NAME config/train_config.yaml

EOF
}

# Cleanup function for graceful shutdown
cleanup() {
    local exit_code=$?
    log_info "Cleaning up distributed training processes..."
    
    # Kill any remaining Python processes
    pkill -f "train_phase2.py" || true
    
    # Clean up distributed process group
    if [ -n "${MASTER_PID:-}" ]; then
        kill "$MASTER_PID" 2>/dev/null || true
    fi
    
    log_info "Cleanup completed with exit code: $exit_code"
    exit $exit_code
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# =============================================================================
# Validation Functions
# =============================================================================

validate_arguments() {
    log_info "Validating arguments and environment..."
    
    # Validate config file
    if [ ! -f "$CONFIG_PATH" ]; then
        log_error "Configuration file not found: $CONFIG_PATH"
        usage
        exit 1
    fi
    
    # Validate numeric arguments
    if ! [[ "$NUM_NODES" =~ ^[1-9][0-9]*$ ]]; then
        log_error "num_nodes must be a positive integer, got: $NUM_NODES"
        exit 1
    fi
    
    if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || [ "$NODE_RANK" -ge "$NUM_NODES" ]; then
        log_error "node_rank must be 0 <= rank < num_nodes, got: $NODE_RANK (num_nodes: $NUM_NODES)"
        exit 1
    fi
    
    if ! [[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || [ "$MASTER_PORT" -gt 65535 ]; then
        log_error "master_port must be a valid port number (1-65535), got: $MASTER_PORT"
        exit 1
    fi
    
    # Validate master address format (basic check)
    if [[ ! "$MASTER_ADDR" =~ ^[a-zA-Z0-9.-]+$ ]]; then
        log_error "Invalid master_addr format: $MASTER_ADDR"
        exit 1
    fi
    
    log_info "Argument validation passed"
}

validate_environment() {
    log_info "Validating environment setup..."
    
    # Check if we're in the correct conda environment
    CONDA_ENV=${CONDA_ENV:-purrsight}
    if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
        log_warn "Expected conda environment '$CONDA_ENV', but currently in '$CONDA_DEFAULT_ENV'"
        log_info "Attempting to activate correct environment..."
        
        # Try to activate the correct environment
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV" || {
                log_error "Failed to activate conda environment: $CONDA_ENV"
                exit 1
            }
            log_info "Successfully activated conda environment: $CONDA_ENV"
        else
            log_error "Conda not found in PATH. Please activate the '$CONDA_ENV' environment manually."
            exit 1
        fi
    fi
    
    # Check Python availability
    if ! command -v python >/dev/null 2>&1; then
        log_error "Python not found in PATH"
        exit 1
    fi
    
    # Check PyTorch availability
    if ! python -c "import torch" 2>/dev/null; then
        log_error "PyTorch not available in current environment"
        exit 1
    fi
    
    # Check CUDA availability for NCCL backend
    if [ "$BACKEND" = "nccl" ]; then
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            log_error "CUDA not available, but NCCL backend requires CUDA"
            exit 1
        fi
        
        # Auto-detect GPU count if needed
        if [ "$NPROC_PER_NODE" = "auto" ]; then
            if command -v nvidia-smi >/dev/null 2>&1; then
                NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
                log_info "Auto-detected $NPROC_PER_NODE GPUs per node"
            else
                log_error "Cannot auto-detect GPU count: nvidia-smi not found"
                exit 1
            fi
        fi
        
        # Validate GPU count
        if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
            log_error "Invalid NPROC_PER_NODE: $NPROC_PER_NODE"
            exit 1
        fi
        
        # Check if we have enough GPUs
        local available_gpus
        available_gpus=$(nvidia-smi -L | wc -l)
        if [ "$NPROC_PER_NODE" -gt "$available_gpus" ]; then
            log_error "Requested $NPROC_PER_NODE GPUs, but only $available_gpus available"
            exit 1
        fi
    fi
    
    # Check training script availability
    if [ ! -f "train_phase2.py" ]; then
        log_error "Training script not found: train_phase2.py"
        exit 1
    fi
    
    log_info "Environment validation passed"
}

# =============================================================================
# Environment Setup Functions
# =============================================================================

setup_distributed_environment() {
    log_info "Setting up distributed training environment variables..."
    
    # Calculate WORLD_SIZE (total number of processes across all nodes)
    WORLD_SIZE=$((NUM_NODES * NPROC_PER_NODE))
    
    # Core distributed training environment variables (Requirement 6.5)
    export MASTER_ADDR="$MASTER_ADDR"
    export MASTER_PORT="$MASTER_PORT"
    export WORLD_SIZE="$WORLD_SIZE"
    export NODE_RANK="$NODE_RANK"
    export NPROC_PER_NODE="$NPROC_PER_NODE"
    
    # NCCL configuration for optimal performance
    if [ "$BACKEND" = "nccl" ]; then
        log_info "Configuring NCCL backend..."
        
        # NCCL debugging and performance settings
        export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
        export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-ALL}
        
        # NCCL network settings
        export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^docker0,lo}
        export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
        export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
        
        # NCCL performance optimizations
        export NCCL_TREE_THRESHOLD=${NCCL_TREE_THRESHOLD:-0}
        export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
        export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
        
        # NCCL timeout settings
        export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
        export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
    fi
    
    # PyTorch distributed settings
    export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
    export PYTHONUNBUFFERED=1  # Ensure immediate output flushing
    
    # CUDA settings for multi-GPU training
    if [ "$BACKEND" = "nccl" ]; then
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE-1)))}
        export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
    fi
    
    # OMP settings for optimal CPU usage
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
    
    log_info "Distributed environment variables configured:"
    log_info "  MASTER_ADDR=$MASTER_ADDR"
    log_info "  MASTER_PORT=$MASTER_PORT"
    log_info "  WORLD_SIZE=$WORLD_SIZE"
    log_info "  NODE_RANK=$NODE_RANK"
    log_info "  NPROC_PER_NODE=$NPROC_PER_NODE"
    log_info "  BACKEND=$BACKEND"
    
    if [ "$LOG_LEVEL" = "DEBUG" ]; then
        log_debug "Additional environment variables:"
        env | grep -E "(NCCL|CUDA|TORCH|OMP|MKL)_" | sort | while read -r line; do
            log_debug "  $line"
        done
    fi
}

setup_logging() {
    log_info "Setting up logging infrastructure..."
    
    # Create log directory first
    mkdir -p "$LOG_DIR"
    
    # Create symlink to latest log
    local latest_log="$LOG_DIR/latest_node${NODE_RANK}.log"
    ln -sf "$(basename "$LOG_FILE")" "$latest_log"
    
    log_info "Logging configured:"
    log_info "  Log file: $LOG_FILE"
    log_info "  Latest log: $latest_log"
    log_info "  Log level: $LOG_LEVEL"
}

# =============================================================================
# Training Launch Functions
# =============================================================================

prepare_training_command() {
    log_info "Preparing distributed training command..."
    
    # Base torchrun command
    local torchrun_cmd="torchrun"
    
    # Add torchrun arguments
    torchrun_cmd="$torchrun_cmd --nnodes=$NUM_NODES"
    torchrun_cmd="$torchrun_cmd --node_rank=$NODE_RANK"
    torchrun_cmd="$torchrun_cmd --nproc_per_node=$NPROC_PER_NODE"
    torchrun_cmd="$torchrun_cmd --master_addr=$MASTER_ADDR"
    torchrun_cmd="$torchrun_cmd --master_port=$MASTER_PORT"
    
    # Add additional torchrun options
    torchrun_cmd="$torchrun_cmd --max_restarts=3"  # Allow restarts on failure
    torchrun_cmd="$torchrun_cmd --rdzv_backend=c10d"  # Use c10d rendezvous backend
    torchrun_cmd="$torchrun_cmd --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    
    # Add training script and arguments
    torchrun_cmd="$torchrun_cmd train_phase2.py"
    torchrun_cmd="$torchrun_cmd --config $CONFIG_PATH"
    
    # Override configuration for distributed training
    torchrun_cmd="$torchrun_cmd --num-gpus $NPROC_PER_NODE"
    
    echo "$torchrun_cmd"
}

launch_training() {
    log_info "Launching distributed training..."
    
    # Prepare the training command
    local training_cmd
    training_cmd=$(prepare_training_command)
    
    log_info "Training command: $training_cmd"
    
    # Create a process monitoring function
    monitor_training() {
        local pid=$1
        local start_time
        start_time=$(date +%s)
        
        while kill -0 "$pid" 2>/dev/null; do
            sleep 30
            local current_time
            current_time=$(date +%s)
            local elapsed=$((current_time - start_time))
            log_debug "Training process $pid still running (elapsed: ${elapsed}s)"
        done
        
        wait "$pid"
        local exit_code=$?
        local end_time
        end_time=$(date +%s)
        local total_time=$((end_time - start_time))
        
        log_info "Training process completed with exit code $exit_code (total time: ${total_time}s)"
        return $exit_code
    }
    
    # Launch training with proper error handling
    log_info "Starting training process..."
    log_info "Command: $training_cmd"
    
    # Execute the training command
    eval "$training_cmd" &
    local training_pid=$!
    MASTER_PID=$training_pid
    
    log_info "Training process started with PID: $training_pid"
    
    # Monitor the training process
    if monitor_training "$training_pid"; then
        log_info "Distributed training completed successfully!"
        return 0
    else
        local exit_code=$?
        log_error "Distributed training failed with exit code: $exit_code"
        return $exit_code
    fi
}

# =============================================================================
# Health Check Functions
# =============================================================================

check_distributed_setup() {
    log_info "Performing distributed setup health checks..."
    
    # Check network connectivity to master node
    if [ "$NODE_RANK" -ne 0 ]; then
        log_info "Testing connectivity to master node: $MASTER_ADDR:$MASTER_PORT"
        if ! timeout 10 bash -c "</dev/tcp/$MASTER_ADDR/$MASTER_PORT" 2>/dev/null; then
            log_warn "Cannot connect to master node at $MASTER_ADDR:$MASTER_PORT"
            log_warn "This may be normal if the master node hasn't started yet"
        else
            log_info "Successfully connected to master node"
        fi
    fi
    
    # Check GPU availability and memory
    if [ "$BACKEND" = "nccl" ]; then
        log_info "Checking GPU status..."
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | \
        while IFS=, read -r idx name total used free; do
            log_info "  GPU $idx: $name (Memory: ${used}MB/${total}MB used, ${free}MB free)"
        done
    fi
    
    # Test PyTorch distributed functionality
    log_info "Testing PyTorch distributed functionality..."
    python -c "
import torch
import torch.distributed as dist
import os

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)')

print(f'Distributed available: {dist.is_available()}')
if dist.is_available():
    print(f'NCCL available: {dist.is_nccl_available()}')
    print(f'MPI available: {dist.is_mpi_available()}')
" 2>&1 | while read -r line; do
        log_info "  $line"
    done
    
    log_info "Health checks completed"
}

# =============================================================================
# Main Execution
# =============================================================================

print_banner() {
    cat << 'EOF'
================================================================================
    ____                      ____  _       _     _     
   |  _ \ _   _ _ __ _ __      / ___|(_) __ _| |__ | |_   
   | |_) | | | | '__| '__|____\___ \| |/ _` | '_ \| __|  
   |  __/| |_| | |  | | |_____|___) | | (_| | | | | |_   
   |_|    \__,_|_|  |_|       |____/|_|\__, |_| |_|\__|  
                                       |___/             
                                                         
           Phase 2 Distributed Training Launcher         
================================================================================
EOF
}

main() {
    # Print banner
    print_banner
    
    # Handle help request
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
        usage
        exit 0
    fi
    
    # Setup logging first
    setup_logging
    
    log_info "Starting $SCRIPT_NAME v$SCRIPT_VERSION"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Node: $NODE_RANK/$NUM_NODES"
    
    # Validate inputs
    validate_arguments
    validate_environment
    
    # Setup distributed environment
    setup_distributed_environment
    
    # Perform health checks
    check_distributed_setup
    
    # Launch training
    log_info "All checks passed. Launching distributed training..."
    if launch_training; then
        log_info "Distributed training completed successfully!"
        log_info "Check logs at: $LOG_FILE"
        exit 0
    else
        log_error "Distributed training failed!"
        log_error "Check logs at: $LOG_FILE"
        exit 1
    fi
}

# Execute main function with all arguments
main "$@"