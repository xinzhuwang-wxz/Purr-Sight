# Purr-Sight 运行脚本

这个目录包含用于训练和推理的统一运行脚本。

## 📁 文件说明

- `run_train.sh` - 训练脚本（Phase 1 和 Phase 2）
- `run_pred.sh` - 推理脚本（支持视频/图片/文字输入）
- `cluster_train.sh` - 集群分布式训练脚本

## 🚀 快速开始

### Phase 1: 对齐训练

```bash
# 使用默认配置训练
./sub/run_train.sh 1

# 指定epoch数量
./sub/run_train.sh 1 --epochs 20

# 使用自定义配置
./sub/run_train.sh 1 --config config/my_config.yaml
```

**输出：**
- Checkpoints: `checkpoints/alignment/`
- MLflow logs: `mlruns/`

### Phase 2: LLM微调

```bash
# 自动查找Phase 1 checkpoint
./sub/run_train.sh 2

# 指定Phase 1 checkpoint
./sub/run_train.sh 2 --checkpoint checkpoints/alignment/xxx/aligner.pt

# 指定epoch数量
./sub/run_train.sh 2 --epochs 10
```

**输出：**
- Checkpoints: `checkpoints/phase2/`
- MLflow logs: `mlruns/`

### 推理

```bash
# 图片推理
./sub/run_pred.sh \
    --checkpoint checkpoints/phase2/best.pt \
    --image data/cat.png

# 视频推理
./sub/run_pred.sh \
    --checkpoint checkpoints/phase2/best.pt \
    --video data/test1.mov

# 文字推理
./sub/run_pred.sh \
    --checkpoint checkpoints/phase2/best.pt \
    --text "A cat is sitting on a windowsill"

# 指定输出文件
./sub/run_pred.sh \
    --checkpoint checkpoints/phase2/best.pt \
    --image data/cat.png \
    --output results/my_inference.json
```

**输出：**
- JSON结果: `results/inference_*.json`

## 📊 工作流程

### 完整训练流程

```bash
# 1. Phase 1训练（对齐训练）
./sub/run_train.sh 1 --epochs 20

# 2. Phase 2训练（LLM微调）
./sub/run_train.sh 2 --epochs 10

# 3. 推理测试
./sub/run_pred.sh \
    --checkpoint checkpoints/phase2/best.pt \
    --image data/cat.png
```

### 查看结果

```bash
# 查看checkpoints
ls -lh checkpoints/alignment/
ls -lh checkpoints/phase2/

# 查看MLflow UI
mlflow ui --backend-store-uri file://./mlruns

# 查看推理结果
cat results/inference_*.json | python -m json.tool
```

## 🔧 高级用法

### 自定义训练参数

```bash
# Phase 1 with custom parameters
./sub/run_train.sh 1 \
    --epochs 30 \
    --batch-size 32 \
    --config config/train_config.yaml

# Phase 2 with custom parameters
./sub/run_train.sh 2 \
    --epochs 15 \
    --batch-size 16 \
    --learning-rate 2e-4 \
    --checkpoint checkpoints/alignment/xxx/aligner.pt
```

### 指定设备

```bash
# 使用CPU推理
./sub/run_pred.sh \
    --checkpoint checkpoints/phase2/best.pt \
    --image data/cat.png \
    --device cpu

# 使用GPU推理
./sub/run_pred.sh \
    --checkpoint checkpoints/phase2/best.pt \
    --image data/cat.png \
    --device cuda
```

## 📝 注意事项

1. **Phase 2依赖Phase 1**
   - 必须先完成Phase 1训练
   - Phase 2会自动查找最新的Phase 1 checkpoint
   - 也可以手动指定checkpoint路径

2. **Checkpoint位置**
   - Phase 1: `checkpoints/alignment/<run_id>/aligner.pt`
   - Phase 2: `checkpoints/phase2/`

3. **MLflow日志**
   - 所有训练运行都会记录到`mlruns/`
   - 使用`mlflow ui`查看详细信息

4. **推理输出**
   - 默认保存到`results/`目录
   - JSON格式，包含行为分析结果

## 🐛 故障排除

### 找不到Phase 1 checkpoint

```bash
# 手动查找checkpoint
find checkpoints/alignment -name "aligner.pt"

# 指定checkpoint路径
./sub/run_train.sh 2 --checkpoint <path_to_aligner.pt>
```

### 推理失败

```bash
# 检查checkpoint是否存在
ls -lh checkpoints/phase2/

# 使用CPU模式
./sub/run_pred.sh --checkpoint <path> --image <path> --device cpu
```

### 查看日志

```bash
# 训练日志
tail -f logs/info.log

# 错误日志
tail -f logs/error.log
```

## 📞 获取帮助

```bash
# 查看训练帮助
./sub/run_train.sh

# 查看推理帮助
./sub/run_pred.sh --help
```

---

**更新日期：** 2026-02-01  
**版本：** 1.0
