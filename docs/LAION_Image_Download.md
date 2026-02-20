# LAION 图像下载指南

LAION-1M-VIT-H-14 数据集只包含元数据（metadata）和预计算的嵌入（embeddings），不包含原始图像文件。图像需要通过 URL 下载。

## 快速开始

### 1. 下载图像

```bash
# 下载前 1000 张图像（用于测试）
python download_laion_images.py --max-samples 1000

# 下载所有图像（可能需要很长时间）
python download_laion_images.py

# 指定自定义输出目录
python download_laion_images.py --output-dir ./laion_images --max-samples 5000
```

### 2. 转换数据集

下载完成后，运行 `makeindex.py` 转换数据集：

```bash
# 只处理 LAION 数据集
python makeindex.py --laion --no-esc50

# 处理两个数据集
python makeindex.py --esc50 --laion
```

脚本会自动检测已下载的图像，优先使用本地路径而不是 URL。

## 下载脚本参数

```bash
python download_laion_images.py --help
```

主要参数：

- `--metadata`: 元数据 parquet 文件路径（默认: `data_formal_alin/laion-1m-vit-h-14/metadate/metadata_000.parquet`）
- `--output-dir`: 输出目录（默认: `data_formal_alin/laion-1m-vit-h-14`）
- `--max-samples`: 最大下载样本数（默认: 全部）
- `--timeout`: 下载超时时间，秒（默认: 10）
- `--retry`: 重试次数（默认: 3）

## 输出结构

下载完成后，目录结构如下：

```
laion-1m-vit-h-14/
├── metadate/
│   └── metadata_000.parquet
├── img_emb/
│   └── img_emb_000.npy
├── text_emb/
│   └── text_emb_000.npy
├── images/                    # 新创建的图像目录
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── image_paths.jsonl          # 图像路径映射文件
└── download_errors.json       # 下载错误记录（如果有）
```

## 注意事项

1. **下载时间**: 下载大量图像可能需要很长时间（取决于网络速度和样本数量）
2. **存储空间**: 确保有足够的磁盘空间（每张图像约 100KB-500KB）
3. **网络稳定性**: 建议在网络稳定的环境下下载
4. **失败重试**: 脚本会自动重试失败的下载（默认 3 次）
5. **图像过滤**: 脚本会自动过滤掉：
   - 尺寸过小的图像（< 64x64）
   - 尺寸过大的图像（> 10000x10000）
   - 非图像文件

## 使用已下载的图像

下载完成后，`makeindex.py` 会自动检测 `image_paths.jsonl` 文件，并在转换 JSONL 时优先使用本地图像路径。

如果图像已下载，转换后的 JSONL 文件将包含：

```json
{"text": "a cat sitting on a mat", "image": "/path/to/images/123.jpg"}
```

如果图像未下载，将使用 URL：

```json
{"text": "a cat sitting on a mat", "image_url": "https://example.com/image.jpg"}
```

## 故障排除

### 下载失败

如果很多图像下载失败，可以：

1. 检查网络连接
2. 增加超时时间：`--timeout 30`
3. 增加重试次数：`--retry 5`
4. 查看错误日志：`download_errors.json`

### 内存不足

如果遇到内存问题，可以：

1. 分批下载：使用 `--max-samples` 参数
2. 下载后立即处理，然后删除图像文件

### URL 失效

某些 URL 可能已经失效，这是正常现象。脚本会跳过这些 URL 并记录到错误日志中。

## 示例工作流

```bash
# 1. 下载前 1000 张图像（测试）
python download_laion_images.py --max-samples 1000

# 2. 检查下载结果
ls -lh data_formal_alin/laion-1m-vit-h-14/images/ | head -20

# 3. 转换数据集（会自动使用本地图像）
python makeindex.py --laion --no-esc50

# 4. 检查转换结果
head -5 data_formal_alin/laion_train.jsonl

# 5. 如果测试成功，下载更多图像
python download_laion_images.py --max-samples 10000
```
