# Phase 2 Cluster Training Guide

本文档说明如何在 Linux GPU 集群上运行 Phase 2 分布式训练。

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [训练方式](#训练方式)
- [监控和调试](#监控和调试)
- [常见问题](#常见问题)

## 环境要求

### 硬件要求

- **GPU**: NVIDIA GPU with CUDA support (推荐 V100, A100, or RTX 3090/4090)
- **内存**: 至少 32GB RAM per node
- **存储**: 至少 100GB 可用空间
- **网络**: 高速网络连接（推荐 InfiniBand 或 10GbE+）

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+, CentOS 7+, or similar)
- **CUDA**: 11.8 or later
- **cuDNN**: 8.6 or later
- **NCCL**: 2.15 or later (for multi-GPU training)
- **Python**: 3.10+
- **PyTorch**: 2.0+ with CUDA support
- **Conda**: Anaconda or Miniconda

### Python 依赖

所有依赖已在 `pyproject.toml` 中定义，通过以下命令安装：

```bash
conda activate purrsight
pip install -e .
```

## 快速开始

### 1. 准备数据

确保 Phase 2 训练数据已准备好：

```bash
# 检查数据目录
ls data/phase2/train.jsonl

# 验证数据格式
python scripts/validate_phase2_data.py data/phase2/train.jsonl
```

### 2. 准备 Phase 1 Checkpoint

确保 Phase 1 的 aligner checkpoint 可用：

```bash
# 检查 checkpoint
ls checkpoints/alignment/*/aligner.pt

# 更新配置文件中的路径
# 编辑 config/phase2_cluster.yaml
# phase1_checkpoint_path: "checkpoints/alignment/<run_id>/aligner.pt"
```

### 3. 单节点多 GPU 训练

最简单的方式，在单个节点上使用多个 GPU：

```bash
# 使用 cluster_train.sh 脚本
bash sub/cluster_train.sh config/phase2_cluster.yaml

# 或者直接使用 torchrun
torchrun --nproc_per_node=4 \
    train/train_llm/train_phase2.py \
    --config config/phase2_cluster.yaml \
    --num-gpus 4
```

### 4. 多节点训练

在多个节点上分布式训练：

**在主节点 (Node 0):**
```bash
bash sub/cluster_train.sh \
    config/phase2_cluster.yaml \
    2 \
    0 \
    192.168.1.100 \
    29500
```

**在工作节点 (Node 1):**
```bash
bash sub/cluster_train.sh \
    config/phase2_cluster.yaml \
    2 \
    1 \
    192.168.1.100 \
    29500
```

### 5. SLURM 集群训练

如果集群使用 SLURM 作业调度系统：

```bash
# 提交作业
sbatch sub/slurm_phase2_train.sh

# 查看作业状态
squeue -u $USER

# 查看日志
tail -f logs/slurm/phase2_train_<jobid>.out

# 取消作业
scancel <jobid>
```

## 配置说明

### 集群配置文件

`config/phase2_cluster.yaml` 包含所有训练参数：

```yaml
# 关键参数说明

# 批次大小（每个 GPU）
batch_size: 16  # 根据 GPU 内存调整
                # V100 (16GB): 8-16
                # A100 (40GB): 32-64
                # RTX 3090 (24GB): 16-32

# GPU 数量
num_gpus: 4  # 每个节点的 GPU 数量

# 学习率
learning_rate: 2.0e-4  # 基础学习率
                       # 实际学习率会根据 GPU 数量自动调整

# 混合精度训练
mixed_precision: true  # 使用 fp16 减少内存使用

# 梯度检查点
gradient_checkpointing: true  # 减少内存使用，略微降低速度

# 分布式后端
distributed_backend: "nccl"  # NVIDIA GPU 使用 NCCL
```

### 性能优化参数

根据你的硬件调整这些参数：

```yaml
# 数据加载
num_workers: 4  # CPU 核心数 / GPU 数量
pin_memory: true  # 加速 CPU-GPU 数据传输
persistent_workers: true  # 保持 workers 活跃

# 梯度累积（如果 OOM）
gradient_accumulation_steps: 2  # 有效 batch size = batch_size * num_gpus * accumulation_steps

# 内存优化
gradient_checkpointing: true  # 牺牲 20-30% 速度换取 50% 内存节省
```

## 训练方式

### 方式 1: cluster_train.sh 脚本（推荐）

功能最完整的启动脚本，包含：
- 自动环境验证
- GPU 检测
- 健康检查
- 详细日志
- 错误处理

```bash
# 基本用法
bash sub/cluster_train.sh <config> [num_nodes] [node_rank] [master_addr] [master_port]

# 示例
bash sub/cluster_train.sh config/phase2_cluster.yaml

# 自定义 GPU 数量
NPROC_PER_NODE=8 bash sub/cluster_train.sh config/phase2_cluster.yaml

# 调试模式
LOG_LEVEL=DEBUG bash sub/cluster_train.sh config/phase2_cluster.yaml
```

### 方式 2: 直接使用 torchrun

更直接的方式，适合熟悉 PyTorch 分布式训练的用户：

```bash
# 单节点
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train/train_llm/train_phase2.py \
    --config config/phase2_cluster.yaml

# 多节点 - 主节点
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train/train_llm/train_phase2.py \
    --config config/phase2_cluster.yaml

# 多节点 - 工作节点
torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=4 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train/train_llm/train_phase2.py \
    --config config/phase2_cluster.yaml
```

### 方式 3: SLURM 作业提交

适合使用 SLURM 管理的集群：

```bash
# 基本提交
sbatch sub/slurm_phase2_train.sh

# 自定义资源
sbatch --nodes=2 --gpus-per-node=8 --time=48:00:00 sub/slurm_phase2_train.sh

# 指定分区
sbatch --partition=gpu-a100 sub/slurm_phase2_train.sh

# 交互式调试
srun --nodes=1 --gpus-per-node=4 --pty bash
conda activate purrsight
bash sub/cluster_train.sh config/phase2_cluster.yaml
```

## 监控和调试

### 实时监控

#### 1. GPU 使用情况

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 查看 GPU 利用率
nvidia-smi dmon -s u

# 查看 GPU 内存
nvidia-smi dmon -s m
```

#### 2. 训练日志

```bash
# 查看最新日志
tail -f logs/distributed_training/latest_node0.log

# 查看特定节点日志
tail -f logs/distributed_training/cluster_train_<timestamp>_node0.log

# 搜索错误
grep -i error logs/distributed_training/*.log
```

#### 3. MLflow UI

```bash
# 启动 MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# 在浏览器中访问
# http://<your-server-ip>:5000
```

#### 4. TensorBoard（如果启用）

```bash
# 启动 TensorBoard
tensorboard --logdir=outputs/phase2 --host 0.0.0.0 --port 6006
```

### 性能分析

#### 1. 训练速度

```bash
# 查看每个 epoch 的时间
grep "Epoch.*completed" logs/distributed_training/*.log

# 查看每个 batch 的时间
grep "batch.*time" logs/distributed_training/*.log
```

#### 2. GPU 利用率

```bash
# 记录 GPU 使用情况
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 1 > gpu_usage.csv

# 分析平均利用率
awk -F',' 'NR>1 {sum+=$3; count++} END {print "Average GPU Utilization:", sum/count "%"}' gpu_usage.csv
```

#### 3. 网络带宽（多节点）

```bash
# 使用 iftop 监控网络
sudo iftop -i eth0

# 或使用 nload
nload eth0
```

### 调试技巧

#### 1. 启用详细日志

```bash
# 设置环境变量
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

# 运行训练
bash sub/cluster_train.sh config/phase2_cluster.yaml
```

#### 2. 单 GPU 测试

先在单个 GPU 上测试，确保代码正确：

```bash
# 修改配置
# num_gpus: 1

# 运行
python train/train_llm/train_phase2.py --config config/phase2_cluster.yaml --num-gpus 1
```

#### 3. 小数据集测试

使用小数据集快速验证：

```bash
# 创建小数据集
head -n 10 data/phase2/train.jsonl > data/phase2/train_small.jsonl

# 修改配置
# data_dir: "data/phase2"
# num_epochs: 1

# 运行
bash sub/cluster_train.sh config/phase2_cluster.yaml
```

#### 4. 检查 checkpoint

```bash
# 验证 checkpoint 可以加载
python -c "
import torch
ckpt = torch.load('checkpoints/phase2/<run_id>/model.pt', map_location='cpu')
print('Checkpoint keys:', ckpt.keys())
print('Model state dict keys:', len(ckpt['model_state_dict'].keys()))
"
```

## 常见问题

### 1. CUDA Out of Memory (OOM)

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```yaml
# 减小 batch size
batch_size: 8  # 从 16 减到 8

# 启用梯度累积
gradient_accumulation_steps: 2

# 启用梯度检查点
gradient_checkpointing: true

# 减小序列长度
max_text_length: 256  # 从 512 减到 256
```

### 2. NCCL 超时

**症状**: `NCCL timeout` 或 `NCCL error`

**解决方案**:
```bash
# 增加超时时间
export NCCL_TIMEOUT=1800  # 30 分钟

# 检查网络连接
ping <master_node_ip>

# 检查防火墙
sudo iptables -L

# 使用正确的网络接口
export NCCL_SOCKET_IFNAME=eth0  # 或 ib0 for InfiniBand
```

### 3. 进程挂起

**症状**: 训练开始后没有输出

**解决方案**:
```bash
# 启用详细日志
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 检查所有节点是否启动
# 确保 master_addr 和 master_port 正确

# 检查进程
ps aux | grep python
```

### 4. 数据加载慢

**症状**: GPU 利用率低，大部分时间在等待数据

**解决方案**:
```yaml
# 增加 data workers
num_workers: 8  # 从 4 增加到 8

# 启用 pin memory
pin_memory: true

# 启用 persistent workers
persistent_workers: true

# 预处理数据
# 将数据预处理并缓存到本地 SSD
```

### 5. 梯度爆炸/消失

**症状**: Loss 变成 NaN 或 Inf

**解决方案**:
```yaml
# 减小学习率
learning_rate: 1.0e-4  # 从 2.0e-4 减到 1.0e-4

# 启用梯度裁剪
gradient_clip_val: 1.0

# 增加 warmup steps
warmup_steps: 1000  # 从 500 增加到 1000

# 检查数据质量
# 确保没有异常值
```

### 6. Checkpoint 保存失败

**症状**: 无法保存 checkpoint

**解决方案**:
```bash
# 检查磁盘空间
df -h

# 检查权限
ls -la checkpoints/phase2/

# 创建目录
mkdir -p checkpoints/phase2

# 检查 NFS 挂载（如果使用共享存储）
mount | grep nfs
```

## 性能基准

### 预期性能（参考）

| 配置 | GPU | Batch Size | 吞吐量 (samples/sec) | 内存使用 |
|------|-----|------------|---------------------|---------|
| 单 GPU | V100 16GB | 8 | ~5 | 14GB |
| 单 GPU | A100 40GB | 32 | ~20 | 35GB |
| 4x GPU | V100 16GB | 8 per GPU | ~18 | 14GB per GPU |
| 4x GPU | A100 40GB | 32 per GPU | ~75 | 35GB per GPU |
| 8x GPU | A100 40GB | 32 per GPU | ~140 | 35GB per GPU |

*注意: 实际性能取决于数据复杂度、网络速度等因素*

### 优化建议

1. **使用混合精度训练**: 可以提升 2-3x 速度
2. **优化数据加载**: 使用足够的 workers 和 pin_memory
3. **使用梯度累积**: 在内存受限时保持大 batch size
4. **使用 InfiniBand**: 多节点训练时网络是瓶颈
5. **本地 SSD**: 将数据放在本地 SSD 而不是 NFS

## 下一步

训练完成后：

1. **验证 checkpoint**:
   ```bash
   python train/train_llm/validate_phase2.py \
       --checkpoint checkpoints/phase2/<run_id>/model.pt
   ```

2. **查看 MLflow 结果**:
   ```bash
   mlflow ui
   # 访问 http://localhost:5000
   ```

3. **运行推理**:
   ```bash
   bash sub/run_pred.sh --checkpoint checkpoints/phase2/<run_id>/model.pt
   ```

4. **导出模型**:
   ```bash
   python scripts/export_model.py \
       --checkpoint checkpoints/phase2/<run_id>/model.pt \
       --output models/phase2_final
   ```

## 支持

如有问题，请查看：
- 日志文件: `logs/distributed_training/`
- MLflow UI: `mlflow ui`
- GitHub Issues: [项目 Issues 页面]

## 参考资料

- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
