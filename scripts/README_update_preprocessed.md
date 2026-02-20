# 增量预处理脚本使用说明

## 概述

当预处理目录中已有部分.pt文件和index.jsonl时，可以使用这两个脚本继续预处理剩余样本：

1. **`update_preprocessed.py`** - 继续预处理，跳过已存在的.pt文件
2. **`update_index.py`** - 将新的.pt文件补充到现有的index.jsonl中

## 使用场景

### 场景1：继续预处理剩余样本

**步骤1：运行 `update_preprocessed.py`**

```bash
python scripts/update_preprocessed.py \
    --input_file data_formal_alin/align_v0.jsonl \
    --preprocessed_dir data_formal_alin/preprocessed
```

**功能**：
- 读取 `align_v0.jsonl` 中的所有样本
- 检查每个样本是否已有对应的.pt文件（通过sample_idx和hash匹配）
- 如果已存在，跳过；如果不存在，预处理并保存
- 可以随时停止（Ctrl+C），下次运行会继续处理剩余样本

**输出**：
- 在 `preprocessed_dir` 中生成新的.pt文件
- **不会**更新 `index.jsonl`（需要运行步骤2）

**示例输出**：
```
================================================================================
继续预处理数据
================================================================================
输入文件: data_formal_alin/align_v0.jsonl
输出目录: data_formal_alin/preprocessed
总样本数: 10,280

开始预处理（跳过已存在文件: True）...
预处理中: 100%|████████████| 10280/10280 [XX:XX<XX, X.XXit/s]

================================================================================
预处理完成
================================================================================
总样本数: 10,280
跳过（已存在）: 2,543
新处理: 7,737
失败: 0

提示: 运行 update_index.py 来更新索引文件
```

### 场景2：更新索引文件

**步骤2：运行 `update_index.py`**

```bash
python scripts/update_index.py \
    --preprocessed_dir data_formal_alin/preprocessed \
    --data_file data_formal_alin/align_v0.jsonl \
    --index_file data_formal_alin/preprocessed/index.jsonl
```

**功能**：
- 扫描预处理目录，找出所有.pt文件
- 读取现有的 `index.jsonl`，获取已有的sample_idx集合
- 找出新的.pt文件（不在现有index中的）
- 匹配原始数据文件，生成新的索引条目
- 追加到现有的 `index.jsonl` 中

**输出**：
- 更新 `index.jsonl`，添加新的索引条目

**示例输出**：
```
================================================================================
更新索引文件
================================================================================
预处理目录: data_formal_alin/preprocessed
数据文件: data_formal_alin/align_v0.jsonl
索引文件: data_formal_alin/preprocessed/index.jsonl

扫描预处理文件...
找到 10,280 个样本的预处理文件

加载现有索引文件...
现有索引包含 2,543 个样本

查找新的样本...
发现 7,737 个新样本

匹配新样本和原始数据...
匹配到 7,737 个新索引条目

追加 7,737 个新条目到索引文件...

验证更新后的索引文件...
更新后索引包含 10,280 个样本
新增: 7,737 个样本

================================================================================
✓ 索引文件更新完成！
================================================================================
```

## 完整工作流程

### 第一次预处理（完整流程）

```bash
# 1. 完整预处理
python -m purrsight.preprocess.prepre \
    --input_file data_formal_alin/align_v0.jsonl \
    --output_dir data_formal_alin/preprocessed \
    --index_file data_formal_alin/preprocessed/index.jsonl
```

### 增量预处理（继续处理剩余样本）

```bash
# 1. 继续预处理（可以随时停止）
python scripts/update_preprocessed.py \
    --input_file data_formal_alin/align_v0.jsonl \
    --preprocessed_dir data_formal_alin/preprocessed

# 2. 更新索引文件（在预处理完成后运行）
python scripts/update_index.py \
    --preprocessed_dir data_formal_alin/preprocessed \
    --data_file data_formal_alin/align_v0.jsonl \
    --index_file data_formal_alin/preprocessed/index.jsonl
```

## 参数说明

### `update_preprocessed.py`

- `--input_file`: 原始数据文件路径（align_v0.jsonl）
- `--preprocessed_dir`: 预处理文件输出目录
- `--force_reprocess`: （可选）强制重新预处理所有样本（即使文件已存在）

### `update_index.py`

- `--preprocessed_dir`: 预处理文件目录路径
- `--data_file`: 原始数据文件路径（align_v0.jsonl）
- `--index_file`: （可选）索引文件路径（默认: preprocessed_dir/index.jsonl）

## 注意事项

1. **顺序执行**：先运行 `update_preprocessed.py`，再运行 `update_index.py`
2. **可以中断**：`update_preprocessed.py` 可以随时停止，下次运行会继续处理剩余样本
3. **索引更新**：每次预处理完成后，记得运行 `update_index.py` 更新索引文件
4. **哈希匹配**：脚本通过sample_idx和hash匹配文件，确保数据一致性

## 故障排除

### 问题1：预处理失败

如果某些样本预处理失败，`update_preprocessed.py` 会跳过它们并继续处理其他样本。失败的样本不会生成.pt文件，也不会被添加到索引中。

### 问题2：索引不匹配

如果发现索引文件中的条目与实际.pt文件不匹配，可以：
1. 删除 `index.jsonl`
2. 运行 `scripts/create_index_from_preprocessed.py` 重新生成完整索引

### 问题3：哈希不匹配

如果 `update_index.py` 报告哈希不匹配，说明：
- 原始数据文件可能被修改
- 预处理文件可能损坏
- 需要重新预处理该样本
