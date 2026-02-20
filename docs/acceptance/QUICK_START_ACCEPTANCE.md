# Purr-Sight 快速验收指南

## 🚀 5分钟快速验收

本指南帮助您快速完成Purr-Sight项目的核心验收测试。

---

## 前置条件

```bash
# 1. 激活环境
conda activate purrsight

# 2. 确认在项目根目录
pwd  # 应该显示 .../Purr-Sight

# 3. 检查数据目录
ls data/test_alignment/  # Phase 1 测试数据
ls data/cat.png          # 推理测试图片
```

---

## 验收流程

### 步骤1：Phase 1 训练验收 (约10-15分钟)

```bash
# 运行Phase 1离线模式训练（3 epochs）
python acceptance_test_phase1.py --mode offline --epochs 3
```

**预期结果：**
```
✅ All acceptance tests PASSED
📄 Report saved to: acceptance_report_phase1_offline_YYYYMMDD_HHMMSS.json
```

**检查点：**
```bash
# 查看生成的checkpoints
ls -lh checkpoints/alignment/

# 应该看到3个checkpoint文件，每个约40-50MB
```

---

### 步骤2：Phase 2 训练验收 (约15-20分钟)

```bash
# 找到Phase 1最佳checkpoint
PHASE1_CKPT=$(ls -t checkpoints/alignment/*.pt | head -1)
echo "Using checkpoint: $PHASE1_CKPT"

# 运行Phase 2训练（3 epochs，含LoRA）
python acceptance_test_phase2.py \
    --phase1_checkpoint "$PHASE1_CKPT" \
    --epochs 3
```

**预期结果：**
```
✅ All acceptance tests PASSED
📄 Report saved to: acceptance_report_phase2_YYYYMMDD_HHMMSS.json
```

**检查点：**
```bash
# 查看Phase 2 checkpoints
ls -lh checkpoints/phase2/

# 检查LoRA参数
python -c "
import torch
ckpt = torch.load('checkpoints/phase2/$(ls -t checkpoints/phase2/*.pt | head -1 | xargs basename)', map_location='cpu')
lora_keys = [k for k in ckpt['model_state_dict'].keys() if 'lora' in k.lower()]
print(f'LoRA parameters: {len(lora_keys)} keys')
"
```

---

### 步骤3：推理模块验收 (约2-3分钟)

```bash
# 找到Phase 2最佳checkpoint
PHASE2_CKPT=$(ls -t checkpoints/phase2/*.pt | head -1)
echo "Using checkpoint: $PHASE2_CKPT"

# 创建结果目录
mkdir -p results

# 测试图片推理
python inference_module.py \
    --checkpoint "$PHASE2_CKPT" \
    --image data/cat.png \
    --output results/inference_image.json

# 测试文字推理
python inference_module.py \
    --checkpoint "$PHASE2_CKPT" \
    --text "A cat is sitting on a windowsill, looking outside peacefully" \
    --output results/inference_text.json
```

**预期结果：**
```
✅ Inference completed successfully
✅ Result saved to: results/inference_image.json
```

**检查点：**
```bash
# 查看推理结果
cat results/inference_image.json | python -m json.tool | head -30

# 验证JSON格式
python -c "
import json
with open('results/inference_image.json') as f:
    result = json.load(f)
    print('✅ JSON格式正确')
    print(f'行为: {result[\"analysis\"][\"behavior\"]}')
    print(f'置信度: {result[\"analysis\"][\"confidence\"]}')
"
```

---

## 验收检查清单

完成上述步骤后，使用此清单验证：

### ✅ Phase 1 验收

- [ ] 训练完成无错误
- [ ] 生成3个checkpoint文件
- [ ] Checkpoint文件大小合理（40-60MB）
- [ ] MLflow实验记录存在
- [ ] 验收报告JSON生成

**验证命令：**
```bash
# 检查checkpoint数量
ls checkpoints/alignment/*.pt | wc -l  # 应该 >= 3

# 检查MLflow
ls mlruns/  # 应该有实验目录

# 检查报告
ls acceptance_report_phase1_*.json
```

### ✅ Phase 2 验收

- [ ] 成功加载Phase 1 checkpoint
- [ ] 训练完成无错误
- [ ] 生成3个checkpoint文件（含LoRA）
- [ ] LoRA参数存在（约20-30个keys）
- [ ] 可训练参数比例合理（3-10%）
- [ ] 验收报告JSON生成

**验证命令：**
```bash
# 检查checkpoint数量
ls checkpoints/phase2/*.pt | wc -l  # 应该 >= 3

# 检查LoRA参数
python -c "
import torch
ckpt = torch.load('$(ls -t checkpoints/phase2/*.pt | head -1)', map_location='cpu')
lora_keys = [k for k in ckpt['model_state_dict'].keys() if 'lora' in k.lower()]
print(f'✅ LoRA参数: {len(lora_keys)} keys')
metadata = ckpt.get('metadata', {})
trainable = metadata.get('trainable_params', 0)
total = metadata.get('total_params', 1)
print(f'✅ 可训练比例: {trainable/total*100:.2f}%')
"

# 检查报告
ls acceptance_report_phase2_*.json
```

### ✅ 推理模块验收

- [ ] 图片推理成功
- [ ] 文字推理成功
- [ ] JSON输出格式正确
- [ ] 包含必需字段（behavior, confidence等）
- [ ] 置信度在0-1范围
- [ ] 结果文件保存成功

**验证命令：**
```bash
# 检查结果文件
ls results/*.json

# 验证JSON格式和内容
python -c "
import json
for file in ['results/inference_image.json', 'results/inference_text.json']:
    try:
        with open(file) as f:
            result = json.load(f)
        required = ['timestamp', 'input_type', 'analysis', 'metadata']
        missing = [k for k in required if k not in result]
        if missing:
            print(f'❌ {file}: 缺少字段 {missing}')
        else:
            print(f'✅ {file}: 格式正确')
            conf = result['analysis'].get('confidence', 0)
            if 0 <= conf <= 1:
                print(f'   置信度: {conf:.2f} ✓')
            else:
                print(f'   ⚠️  置信度异常: {conf}')
    except Exception as e:
        print(f'❌ {file}: {e}')
"
```

---

## 查看结果

### MLflow UI

```bash
# 启动MLflow UI
mlflow ui --backend-store-uri file://./mlruns

# 在浏览器打开
open http://localhost:5000
```

### 验收报告

```bash
# 查看Phase 1报告
cat acceptance_report_phase1_*.json | python -m json.tool

# 查看Phase 2报告
cat acceptance_report_phase2_*.json | python -m json.tool

# 查看推理结果
cat results/inference_image.json | python -m json.tool
```

---

## 常见问题

### Q1: 训练时间太长怎么办？

**A:** 可以减少epochs数量进行快速测试：
```bash
python acceptance_test_phase1.py --mode offline --epochs 1
python acceptance_test_phase2.py --phase1_checkpoint <path> --epochs 1
```

### Q2: 找不到Phase 1 checkpoint？

**A:** 检查checkpoint目录：
```bash
ls -la checkpoints/alignment/
# 如果为空，需要先运行Phase 1训练
```

### Q3: 推理模块报错找不到模型？

**A:** 确保使用正确的checkpoint路径：
```bash
# 列出所有可用checkpoints
find checkpoints -name "*.pt" -type f

# 使用最新的checkpoint
LATEST=$(find checkpoints/phase2 -name "*.pt" -type f | sort -r | head -1)
python inference_module.py --checkpoint "$LATEST" --image data/cat.png
```

### Q4: MLflow UI无法访问？

**A:** 检查MLflow tracking URI：
```bash
# 设置本地tracking URI
export MLFLOW_TRACKING_URI=file://./mlruns

# 重新启动UI
mlflow ui --backend-store-uri file://./mlruns
```

---

## 成功标准

所有验收测试通过的标志：

1. ✅ **Phase 1测试输出：** `✅ All acceptance tests PASSED`
2. ✅ **Phase 2测试输出：** `✅ All acceptance tests PASSED`
3. ✅ **推理测试输出：** `✅ Inference completed successfully`
4. ✅ **Checkpoint文件：** 两个阶段各有3+个checkpoint文件
5. ✅ **MLflow记录：** 实验和运行记录存在
6. ✅ **推理结果：** JSON格式正确，内容合理

---

## 下一步

验收测试全部通过后：

1. **查看详细文档**
   - `ACCEPTANCE_TEST_GUIDE.md` - 完整验收指南
   - `CORE_VALIDATION_SUMMARY.md` - 核心验证总结

2. **生产部署准备**
   - 优化模型配置
   - 设置监控和日志
   - 准备部署环境

3. **功能扩展**
   - 添加更多测试数据
   - 改进JSON输出格式
   - 优化推理性能

---

## 获取帮助

如遇到问题：

1. 查看日志：`tail -100 logs/info.log`
2. 查看错误日志：`tail -100 logs/error.log`
3. 参考完整指南：`ACCEPTANCE_TEST_GUIDE.md`
4. 查看测试文档：`tests/README.md`

---

**预计总时间：** 30-40分钟  
**难度：** 简单  
**前置要求：** purrsight环境已配置，测试数据已准备

祝验收顺利！🎉
