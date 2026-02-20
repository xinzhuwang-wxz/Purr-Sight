# Phase 2 Checkpoint 和 MLflow 对应关系说明

## 问题描述

之前的实现中，checkpoint 目录名和 MLflow run ID 是两个不同的 ID：
- Checkpoint: `{uuid}_{timestamp}` (自己生成的 UUID)
- MLflow: `{mlflow_run_id}` (MLflow 生成的 UUID)

这导致两者完全不对应，很难找到关联。

## 解决方案：使用 Phase 1 的方式

### Phase 1 的做法

Phase 1 使用 **MLflow 生成的 run_id** 作为 checkpoint 目录名的一部分：

```python
# Phase 1 代码
active_run = mlflow.active_run()
run_id = active_run.info.run_id  # MLflow 生成的 run_id
checkpoint_dir = f"{run_id}_{timestamp}"
```

结果：
- Checkpoint: `checkpoints/alignment/9caa59d265f14e8eb4d8c704a827d775_20260201_025845/`
- MLflow: `mlruns/{experiment_id}/9caa59d265f14e8eb4d8c704a827d775/`

**ID 完全对应！** ✅

### Phase 2 的新实现

现在 Phase 2 也采用同样的方式：

```python
# Phase 2 新代码
# 1. 先启动 MLflow run，让 MLflow 生成 run_id
mlflow.start_run(run_name=f"phase2_{timestamp}")
active_run = mlflow.active_run()
run_id = active_run.info.run_id  # 使用 MLflow 的 run_id

# 2. 用这个 run_id 创建 checkpoint 目录
checkpoint_dir = Path(checkpoint_dir) / f"{run_id}_{timestamp}"
```

结果：
- Checkpoint: `checkpoints/phase2/94525c6650a3407985928d7c2f83f9eb_20260201_044652/`
- MLflow: `mlruns/463312655126284597/94525c6650a3407985928d7c2f83f9eb/`

**ID 完全对应！** ✅

## 命名规则

### 统一的命名规则

```
{mlflow_run_id}_{timestamp}
```

其中：
- `mlflow_run_id`: MLflow 自动生成的 32 位十六进制 UUID
- `timestamp`: `YYYYMMDD_HHMMSS` 格式的时间戳

### 示例

**Phase 1:**
```
9caa59d265f14e8eb4d8c704a827d775_20260201_025845
└─────────────┬─────────────┘ └──────┬──────┘
         MLflow run_id              timestamp
```

**Phase 2:**
```
94525c6650a3407985928d7c2f83f9eb_20260201_044652
└─────────────┬─────────────┘ └──────┬──────┘
         MLflow run_id              timestamp
```

## 对应关系验证

### 从 Checkpoint 找到 MLflow

```bash
# Checkpoint 目录名的前 32 位就是 MLflow run_id
CHECKPOINT_DIR="94525c6650a3407985928d7c2f83f9eb_20260201_044652"
RUN_ID="${CHECKPOINT_DIR:0:32}"  # 提取前 32 位

# 在 MLflow 中查找
find mlruns -name "$RUN_ID" -type d
# 输出: mlruns/463312655126284597/94525c6650a3407985928d7c2f83f9eb
```

### 从 MLflow 找到 Checkpoint

```bash
# MLflow run_id
RUN_ID="94525c6650a3407985928d7c2f83f9eb"

# 在 checkpoint 目录中查找
ls checkpoints/phase2/ | grep "^${RUN_ID}_"
# 输出: 94525c6650a3407985928d7c2f83f9eb_20260201_044652
```

## 文件说明

### MLFLOW_RUN_ID.txt

每个 checkpoint 目录中都有这个文件，明确说明对应关系：

```
MLflow Run ID: 94525c6650a3407985928d7c2f83f9eb
Checkpoint Dir: checkpoints/phase2/94525c6650a3407985928d7c2f83f9eb_20260201_044652
Experiment: phase2_training_with_pretrained_aligner
Timestamp: 20260201_044652

Note: The checkpoint directory name includes the MLflow run_id:
  94525c6650a3407985928d7c2f83f9eb_20260201_044652 = 94525c6650a3407985928d7c2f83f9eb_20260201_044652
```

### README.md

完整的运行信息文档，包含：

```markdown
## Directory Naming Convention

The checkpoint directory name follows Phase 1 convention:
```
94525c6650a3407985928d7c2f83f9eb_20260201_044652 = {mlflow_run_id}_{timestamp}
                     = 94525c6650a3407985928d7c2f83f9eb_20260201_044652
```

This ensures the checkpoint directory and MLflow run are easily matched!
```

## 优势

### 1. 直观对应
- 目录名的前 32 位 = MLflow run_id
- 一眼就能看出对应关系

### 2. 易于查找
- 从 checkpoint 目录名直接提取 run_id
- 从 MLflow run_id 直接搜索 checkpoint 目录

### 3. 与 Phase 1 一致
- 两个阶段使用相同的命名规则
- 统一的项目结构

### 4. 时间戳保留
- 仍然包含时间戳信息
- 方便按时间排序和查找

## 总结

现在 Phase 2 的 checkpoint 和 MLflow 完全对应：

| 项目 | Phase 1 | Phase 2 |
|------|---------|---------|
| Checkpoint 目录 | `{mlflow_run_id}_{timestamp}` | `{mlflow_run_id}_{timestamp}` |
| MLflow 目录 | `mlruns/{exp_id}/{mlflow_run_id}/` | `mlruns/{exp_id}/{mlflow_run_id}/` |
| 对应关系 | ✅ 完全对应 | ✅ 完全对应 |

**问题已解决！** 🎉
