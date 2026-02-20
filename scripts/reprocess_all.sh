#!/bin/bash
# 重新生成所有预处理文件

set -e

echo "============================================================"
echo "重新生成预处理文件"
echo "============================================================"

# 配置
INPUT_FILE="data_formal_alin/align_v0.jsonl"
OUTPUT_DIR="data_formal_alin/preprocessed"
INDEX_FILE="data_formal_alin/preprocessed/index.jsonl"

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    echo "请先运行: python3 merge_datasets.py"
    exit 1
fi

# 清理旧的预处理文件（可选）
echo ""
echo "清理旧的预处理文件..."
rm -f "$OUTPUT_DIR"/*.pt "$INDEX_FILE" 2>/dev/null || true
echo "✓ 已清理"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行预处理
echo ""
echo "开始预处理..."
echo "输入文件: $INPUT_FILE"
echo "输出目录: $OUTPUT_DIR"
echo ""

python3 -m purrsight.preprocess.prepre \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --index_file "$INDEX_FILE" \
    --cleanup_corrupted

echo ""
echo "============================================================"
echo "✓ 预处理完成！"
echo "============================================================"
echo "索引文件: $INDEX_FILE"
echo "预处理文件数: $(find "$OUTPUT_DIR" -name "*.pt" -type f | wc -l | tr -d ' ')"
echo "索引条目数: $(wc -l < "$INDEX_FILE" | tr -d ' ')"
