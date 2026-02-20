# Cluster Training Guide for Phase 2

This guide explains how to use the `cluster_train.sh` script for distributed training of the Phase 2 multi-modal LLM on GPU clusters.

## Overview

The `cluster_train.sh` script provides a comprehensive solution for launching distributed training using PyTorch's `torchrun` with Distributed Data Parallel (DDP). It handles:

- Environment variable setup for distributed training
- NCCL backend configuration for GPU communication
- Multi-node cluster coordination
- Error handling and logging
- Health checks and validation

## Requirements Satisfied

- **6.1**: Initialize DDP with correct world size and rank
- **6.2**: Wrap model with DistributedDataParallel (handled by PyTorch Lightning)
- **6.3**: Configure data loader with DistributedSampler (handled by PyTorch Lightning)
- **6.4**: Synchronize gradients across all processes (handled by DDP)
- **6.5**: Use environment variables for distributed training configuration

## Quick Start

### Single Node Training (4 GPUs)

```bash
# Basic single-node training
./cluster_train.sh config/train_config.yaml

# With custom GPU count
NPROC_PER_NODE=8 ./cluster_train.sh config/train_config.yaml
```

### Multi-Node Training

For a 2-node cluster with 4 GPUs each:

**On Master Node (192.168.1.100):**
```bash
./cluster_train.sh config/train_config.yaml 2 0 192.168.1.100 29500
```

**On Worker Node (192.168.1.101):**
```bash
./cluster_train.sh config/train_config.yaml 2 1 192.168.1.100 29500
```

## Script Arguments

```
./cluster_train.sh [config_path] [num_nodes] [node_rank] [master_addr] [master_port]
```

- `config_path`: Path to training configuration YAML file (default: `config/train_config.yaml`)
- `num_nodes`: Total number of nodes in the cluster (default: 1)
- `node_rank`: Rank of this node, 0-based (default: 0)
- `master_addr`: IP address of the master node (default: localhost)
- `master_port`: Port for distributed communication (default: 29500)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NPROC_PER_NODE` | Number of processes (GPUs) per node | auto-detect |
| `BACKEND` | Distributed backend (nccl/gloo) | nccl |
| `LOG_LEVEL` | Logging level (INFO/DEBUG) | INFO |
| `CONDA_ENV` | Conda environment name | purrsight |

## Advanced Usage

### Debug Mode
```bash
LOG_LEVEL=DEBUG ./cluster_train.sh config/train_config.yaml
```

### Custom Backend
```bash
BACKEND=gloo ./cluster_train.sh config/train_config.yaml
```

### Custom Environment
```bash
CONDA_ENV=my_env ./cluster_train.sh config/train_config.yaml
```

## Cluster Setup Requirements

### Network Configuration

1. **Firewall**: Ensure the master port (default 29500) is open between nodes
2. **SSH Access**: Set up passwordless SSH between nodes (optional, for easier management)
3. **Shared Storage**: Use shared filesystem or ensure model checkpoints are accessible

### Software Requirements

1. **Conda Environment**: The `purrsight` environment must be available on all nodes
2. **PyTorch**: Version 2.0+ with CUDA support
3. **NCCL**: For GPU communication (usually included with PyTorch)
4. **Training Script**: `train_phase2.py` must be present in the same directory

### Hardware Requirements

1. **GPUs**: CUDA-compatible GPUs on all nodes
2. **Memory**: Sufficient GPU memory for the model and batch size
3. **Network**: High-bandwidth network for multi-node training (InfiniBand recommended)

## Monitoring and Logging

### Log Files

Logs are stored in `logs/distributed_training/`:
- `cluster_train_TIMESTAMP_nodeRANK.log`: Full log for each node
- `latest_nodeRANK.log`: Symlink to the latest log file

### Monitoring Training

```bash
# Monitor training progress
tail -f logs/distributed_training/latest_node0.log

# Check all nodes
for i in {0..3}; do
    echo "=== Node $i ==="
    tail -n 5 logs/distributed_training/latest_node$i.log
done
```

### Health Checks

The script performs automatic health checks:
- Configuration file validation
- Environment setup verification
- GPU availability and memory
- Network connectivity (for worker nodes)
- PyTorch distributed functionality

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   ```
   Cannot connect to master node at IP:PORT
   ```
   - Check firewall settings
   - Verify master node is running
   - Ensure correct IP address and port

2. **CUDA Out of Memory**
   ```
   CUDA out of memory
   ```
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable mixed precision training

3. **Environment Issues**
   ```
   Failed to activate conda environment
   ```
   - Ensure conda is in PATH
   - Verify environment exists on all nodes
   - Use `CONDA_ENV` variable to specify correct environment

4. **NCCL Errors**
   ```
   NCCL initialization failed
   ```
   - Check GPU compatibility
   - Verify CUDA installation
   - Try `BACKEND=gloo` for debugging

### Debug Commands

```bash
# Test environment setup
CONDA_ENV=base LOG_LEVEL=DEBUG ./cluster_train.sh --help

# Validate configuration
python -c "from train.train_llm.train_llm_conf import load_config; print(load_config('config/train_config.yaml'))"

# Check GPU status
nvidia-smi

# Test PyTorch distributed
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

## Performance Optimization

### NCCL Tuning

The script automatically sets optimal NCCL parameters:
- `NCCL_DEBUG=INFO`: Enable debugging
- `NCCL_TREE_THRESHOLD=0`: Use tree algorithms
- `NCCL_P2P_DISABLE=0`: Enable peer-to-peer communication
- `NCCL_SHM_DISABLE=0`: Enable shared memory

### Network Optimization

For InfiniBand networks:
```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
```

For Ethernet networks:
```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
```

### Memory Optimization

1. **Mixed Precision**: Enabled automatically in PyTorch Lightning
2. **Gradient Checkpointing**: Configure in training config
3. **Batch Size**: Scale with number of GPUs

## Example Cluster Configurations

### 2-Node Cluster (8 GPUs total)

**Node 0 (Master):**
```bash
./cluster_train.sh config/train_config.yaml 2 0 192.168.1.100 29500
```

**Node 1 (Worker):**
```bash
./cluster_train.sh config/train_config.yaml 2 1 192.168.1.100 29500
```

### 4-Node Cluster (32 GPUs total)

**Node 0 (Master):**
```bash
NPROC_PER_NODE=8 ./cluster_train.sh config/train_config.yaml 4 0 192.168.1.100 29500
```

**Nodes 1-3 (Workers):**
```bash
NPROC_PER_NODE=8 ./cluster_train.sh config/train_config.yaml 4 1 192.168.1.100 29500
NPROC_PER_NODE=8 ./cluster_train.sh config/train_config.yaml 4 2 192.168.1.100 29500
NPROC_PER_NODE=8 ./cluster_train.sh config/train_config.yaml 4 3 192.168.1.100 29500
```

## Integration with Job Schedulers

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=purrsight-phase2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# Get node information
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Launch training on each node
srun --ntasks=1 --nodes=1 --ntasks-per-node=1 \
    ./cluster_train.sh config/train_config.yaml $SLURM_NNODES $SLURM_NODEID $MASTER_ADDR $MASTER_PORT
```

### PBS/Torque Example

```bash
#!/bin/bash
#PBS -N purrsight-phase2
#PBS -l nodes=2:ppn=4:gpus=4
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR

# Extract node information
MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
NODE_RANK=$(grep -n $(hostname) $PBS_NODEFILE | head -n 1 | cut -d: -f1)
NODE_RANK=$((NODE_RANK - 1))  # Convert to 0-based

./cluster_train.sh config/train_config.yaml 2 $NODE_RANK $MASTER_ADDR 29500
```

## Best Practices

1. **Start Small**: Test with single node before scaling to multiple nodes
2. **Monitor Resources**: Watch GPU memory and network utilization
3. **Use Checkpoints**: Enable frequent checkpointing for long training runs
4. **Log Everything**: Use DEBUG mode for initial setup and troubleshooting
5. **Test Connectivity**: Verify network connectivity between nodes before training
6. **Backup Configs**: Keep training configurations in version control
7. **Resource Planning**: Calculate memory requirements and network bandwidth needs

## Support

For issues with the cluster training script:
1. Check the log files in `logs/distributed_training/`
2. Run with `LOG_LEVEL=DEBUG` for detailed information
3. Verify all requirements are met on all nodes
4. Test with single node first to isolate issues