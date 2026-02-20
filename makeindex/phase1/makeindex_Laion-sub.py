"""
LAION Subset 数据集探索脚本，并转换为 image-text pairs 格式

参考:
- LAION Subset: data_formal_alin/laion-subset

功能:
1. 探索 LAION Subset 数据集（image-text pairs）
2. 生成统计信息和可视化
3. 提取图像并保存为图片文件
4. 转换为 train.jsonl 格式用于对齐训练

数据集结构:
- image: PIL Image object (512×512)
- caption: Text description of the image
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from PIL import Image
import io

# 设置中文字体支持（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LaionSubsetExplorer:
    """LAION Subset 数据集探索器
    
    参考 explore_laion_data.py 和 explore_images_detailed.py 的实现
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        初始化数据集探索器
        
        Args:
            data_dir: 数据集根目录
            output_dir: 输出目录（用于保存统计信息和转换后的数据）
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir_path = self.data_dir / "data"
        self.images_output_dir = self.data_dir / "extracted_images"
        self.images_output_dir.mkdir(parents=True, exist_ok=True)
    
    def _extract_image_from_dict(self, image_val: dict) -> bytes:
        """从字典中提取图像字节数据
        
        Args:
            image_val: 包含图像数据的字典
            
        Returns:
            图像字节数据
            
        Raises:
            ValueError: 当无法提取图像数据时
        """
        img_bytes = image_val.get('bytes') or image_val.get('data')
        
        if img_bytes and isinstance(img_bytes, bytes):
            return img_bytes
        elif img_bytes and isinstance(img_bytes, str):
            # 字符串格式，尝试解析
            bytes_part = img_bytes
            
            # 检查是否包含 path 信息
            if "', 'path': '" in bytes_part:
                path_start = bytes_part.find("', 'path': '")
                if path_start > 0:
                    bytes_part = bytes_part[:path_start]
            
            # 去掉首尾的引号
            if bytes_part.startswith("'") and bytes_part.endswith("'"):
                bytes_part = bytes_part[1:-1]
            elif bytes_part.startswith('"') and bytes_part.endswith('"'):
                bytes_part = bytes_part[1:-1]
            
            # 使用 codecs.decode 处理转义序列
            import codecs
            single_escape = bytes_part.replace('\\\\', '\\')
            decoded = codecs.decode(single_escape, 'unicode_escape').encode('latin-1')
            
            if len(decoded) > 10:
                # 验证是否是有效的图像数据
                if decoded.startswith(b'\xff\xd8\xff') or decoded.startswith(b'\x89PNG') or decoded.startswith(b'RIFF'):
                    return decoded
            
            raise ValueError(f"无法解析图像字符串数据: 长度={len(bytes_part)}, 解码后长度={len(decoded)}")
        
        raise ValueError(f"无法从字典中提取图像数据: keys={list(image_val.keys())}")
    
    def _extract_image_bytes(self, image_data, idx: int) -> Path:
        """从图像数据中提取图像并保存到文件
        
        Args:
            image_data: 图像数据（可能是字典、字节或字符串）
            idx: 样本索引，用于生成文件名
            
        Returns:
            保存的图像文件路径
            
        Raises:
            ValueError: 当无法提取或保存图像时
        """
        # 提取图像字节数据
        if isinstance(image_data, dict):
            image_bytes = self._extract_image_from_dict(image_data)
            suggested_path = image_data.get('path', None)
        elif isinstance(image_data, bytes):
            image_bytes = image_data
            suggested_path = None
        elif isinstance(image_data, str):
            image_bytes = self._extract_image_from_dict({'bytes': image_data})
            suggested_path = None
        else:
            raise ValueError(f"不支持的图像数据类型: {type(image_data)}")
        
        # 验证图像数据
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_format = img.format
            img.close()
            
            # 根据格式确定扩展名
            if img_format == 'JPEG':
                ext = '.jpg'
            elif img_format == 'PNG':
                ext = '.png'
            elif img_format == 'WEBP':
                ext = '.webp'
            else:
                ext = '.jpg'
        except Exception:
            # 通过文件头判断
            if len(image_bytes) >= 3 and image_bytes[:3] == b'\xff\xd8\xff':
                ext = '.jpg'
            elif len(image_bytes) >= 4 and image_bytes[:4] == b'\x89PNG':
                ext = '.png'
            elif len(image_bytes) >= 12 and image_bytes[:4] == b'RIFF' and b'WEBP' in image_bytes[8:12]:
                ext = '.webp'
            else:
                raise ValueError(f"无法识别图像格式: 文件头={image_bytes[:12]}")
        
        # 生成文件名
        if suggested_path and isinstance(suggested_path, str):
            safe_filename = Path(suggested_path).name
            safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c in '.-_')
            if not safe_filename or '.' not in safe_filename:
                safe_filename = f"{idx:08d}{ext}"
            image_filename = safe_filename
        else:
            image_filename = f"{idx:08d}{ext}"
        
        image_path = self.images_output_dir / image_filename
        
        # 如果文件已存在，直接返回
        if image_path.exists():
            return image_path
        
        # 保存图像
        with open(image_path, 'wb') as img_file:
            img_file.write(image_bytes)
        
        # 验证文件完整性
        try:
            verify_img = Image.open(image_path)
            verify_img.verify()
            verify_img.close()
        except Exception as e:
            image_path.unlink()
            raise ValueError(f"保存的图像文件验证失败: {e}") from e
        
        return image_path
        
    def explore(self) -> Dict:
        """探索 LAION Subset 数据集"""
        print("=" * 80)
        print("Exploring LAION Subset Dataset")
        print("=" * 80)
        
        stats = {}
        
        # 查找所有 parquet 文件
        parquet_files = sorted(list(self.data_dir_path.glob("*.parquet")))
        
        if not parquet_files:
            print(f"Error: No parquet files found in {self.data_dir_path}")
            stats['error'] = "No parquet files found"
            return stats
        
        print(f"\nFound {len(parquet_files)} parquet files")
        
        # 读取所有 parquet 文件
        dfs = []
        try:
            for parquet_file in parquet_files:
                print(f"  Reading {parquet_file.name}...")
                try:
                    df = pd.read_parquet(parquet_file, engine='pyarrow')
                except Exception as e1:
                    try:
                        df = pd.read_parquet(parquet_file, engine='fastparquet')
                    except Exception as e2:
                        print(f"    Warning: Could not read {parquet_file.name}")
                        print(f"      PyArrow error: {e1}")
                        print(f"      FastParquet error: {e2}")
                        continue
                dfs.append(df)
            
            if not dfs:
                print("Error: Could not read any parquet files")
                stats['error'] = "Could not read parquet files"
                return stats
            
            # 合并所有数据框
            df = pd.concat(dfs, ignore_index=True)
            
            stats['total_samples'] = len(df)
            stats['num_files'] = len(dfs)
            stats['columns'] = df.columns.tolist()
            stats['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # 基本统计信息
            print(f"\nDataset Overview:")
            print(f"  Total samples: {stats['total_samples']:,}")
            print(f"  Number of files: {stats['num_files']}")
            print(f"  Columns: {len(stats['columns'])}")
            print(f"\nColumn names:")
            for col in stats['columns']:
                print(f"  - {col}")
            
            # 检查关键字段
            if 'caption' in df.columns:
                caption_lengths = df['caption'].str.len()
                stats['caption_stats'] = {
                    'mean_length': float(caption_lengths.mean()),
                    'median_length': float(caption_lengths.median()),
                    'min_length': int(caption_lengths.min()),
                    'max_length': int(caption_lengths.max()),
                    'std_length': float(caption_lengths.std())
                }
                print(f"\nCaption Statistics:")
                print(f"  Mean length: {stats['caption_stats']['mean_length']:.1f} chars")
                print(f"  Median length: {stats['caption_stats']['median_length']:.1f} chars")
                print(f"  Min length: {stats['caption_stats']['min_length']} chars")
                print(f"  Max length: {stats['caption_stats']['max_length']} chars")
            
            # 检查图像列的类型（参考 explore_laion_data.py）
            if 'image' in df.columns:
                first_image = df['image'].iloc[0] if len(df) > 0 else None
                if first_image is not None and pd.notna(first_image):
                    if isinstance(first_image, dict):
                        # 参考 explore_laion_data.py: 检查字典格式
                        stats['image_format'] = 'dict'
                        print(f"\nImage Format: Dictionary (embedded in parquet)")
                        print(f"  Keys: {list(first_image.keys())[:10]}")
                        print(f"  Images will be extracted to: {self.images_output_dir}")
                        
                        # 尝试提取第一个图像验证
                        img_bytes = self._extract_image_from_dict(first_image)
                        if img_bytes:
                            try:
                                img = Image.open(io.BytesIO(img_bytes))
                                print(f"  Sample image: {img.size[0]}×{img.size[1]}, format: {img.format}")
                                img.close()
                            except Exception as e:
                                print(f"  Warning: Could not verify sample image: {e}")
                    elif isinstance(first_image, bytes):
                        stats['image_format'] = 'bytes'
                        print(f"\nImage Format: Raw bytes data")
                        print(f"  Images will be extracted to: {self.images_output_dir}")
                    elif isinstance(first_image, str):
                        if len(first_image) > 1000:
                            stats['image_format'] = 'string'
                            print(f"\nImage Format: Encoded string ({len(first_image)} chars)")
                            print(f"  Images will be extracted to: {self.images_output_dir}")
                        else:
                            stats['image_format'] = 'path'
                            image_exists = df['image'].apply(lambda x: Path(str(x)).exists() if pd.notna(x) else False)
                            stats['images_exist'] = int(image_exists.sum())
                            stats['images_missing'] = int((~image_exists).sum())
                            print(f"\nImage Format: File paths")
                            print(f"  Existing: {stats['images_exist']:,}")
                            print(f"  Missing: {stats['images_missing']:,}")
                    else:
                        stats['image_format'] = 'unknown'
                        print(f"\nImage Format: Unknown type ({type(first_image)})")
                else:
                    stats['image_format'] = 'unknown'
            
            # 显示前几个样本
            print(f"\nSample entries (first 3):")
            for idx in range(min(3, len(df))):
                row = df.iloc[idx]
                print(f"\n  Sample {idx + 1}:")
                if 'caption' in df.columns:
                    caption_preview = str(row['caption'])[:100] + "..." if len(str(row['caption'])) > 100 else str(row['caption'])
                    print(f"    Caption: {caption_preview}")
                if 'image' in df.columns:
                    image_val = row.get('image')
                    if pd.notna(image_val):
                        if isinstance(image_val, (dict, bytes)) or (isinstance(image_val, str) and len(str(image_val)) > 1000):
                            print(f"    Image: [Bytes data, will be extracted]")
                        else:
                            image_path = str(image_val)
                            print(f"    Image: {image_path[:80]}...")
                    else:
                        print(f"    Image: N/A")
            
            stats['dataframe'] = df
            
        except Exception as e:
            print(f"Error reading data: {e}")
            import traceback
            traceback.print_exc()
            stats['error'] = str(e)
        
        return stats
    
    def visualize(self, stats: Dict) -> None:
        """可视化 LAION Subset 数据集统计信息"""
        if 'error' in stats or 'dataframe' not in stats:
            print("Cannot visualize: missing data")
            return
        
        df = stats['dataframe']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LAION Subset Dataset Statistics', fontsize=16, fontweight='bold')
        
        # 1. Caption length distribution
        if 'caption' in df.columns:
            ax1 = axes[0, 0]
            caption_lengths = df['caption'].str.len()
            ax1.hist(caption_lengths, bins=50, edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Caption Length (characters)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Caption Length Distribution')
            ax1.grid(True, alpha=0.3)
        
        # 2. Image format info
        if 'image' in df.columns:
            ax2 = axes[0, 1]
            image_format = stats.get('image_format', 'unknown')
            if image_format in ['dict', 'bytes', 'string']:
                ax2.text(0.5, 0.5, f"Image Format: {image_format}\n(Embedded in parquet)", 
                        ha='center', va='center', fontsize=12, transform=ax2.transAxes)
                ax2.set_title('Image Format')
                ax2.axis('off')
            else:
                image_exists = df['image'].apply(lambda x: Path(str(x)).exists() if pd.notna(x) else False)
                exists_counts = image_exists.value_counts()
                ax2.bar(['Missing', 'Exists'], [exists_counts.get(False, 0), exists_counts.get(True, 0)], 
                       alpha=0.7, edgecolor='black')
                ax2.set_ylabel('Count')
                ax2.set_title('Image File Existence')
                ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Caption length by existence (if applicable)
        if 'caption' in df.columns and 'image' in df.columns:
            ax3 = axes[1, 0]
            image_format = stats.get('image_format', 'unknown')
            if image_format not in ['dict', 'bytes', 'string']:
                image_exists = df['image'].apply(lambda x: Path(str(x)).exists() if pd.notna(x) else False)
                caption_lengths_exist = df[image_exists]['caption'].str.len()
                caption_lengths_missing = df[~image_exists]['caption'].str.len()
                ax3.hist([caption_lengths_exist, caption_lengths_missing], bins=30, 
                        label=['Image Exists', 'Image Missing'], alpha=0.7, edgecolor='black')
                ax3.set_xlabel('Caption Length (characters)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Caption Length by Image Existence')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.axis('off')
        else:
            axes[1, 0].axis('off')
        
        # 4. File distribution
        if 'num_files' in stats:
            ax4 = axes[1, 1]
            ax4.text(0.5, 0.5, f"Total Files: {stats['num_files']}\nTotal Samples: {stats['total_samples']:,}", 
                    ha='center', va='center', fontsize=14, transform=ax4.transAxes)
            ax4.set_title('Dataset Summary')
            ax4.axis('off')
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / "laion_subset_statistics.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        plt.close()
    
    def convert_to_jsonl(self, stats: Dict, output_path: str, max_samples: Optional[int] = None) -> None:
        """转换为 train.jsonl 格式"""
        if 'dataframe' not in stats:
            print("Error: No data to convert")
            return
        
        df = stats['dataframe']
        image_format = stats.get('image_format', 'unknown')
        
        print(f"\nConverting LAION Subset dataset to JSONL format...")
        print(f"Output path: {output_path}")
        if image_format in ['dict', 'bytes', 'string']:
            print(f"Image format: {image_format} (will extract to {self.images_output_dir})")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        skipped = 0
        image_exists_count = 0
        image_missing_count = 0
        image_extracted_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                if max_samples and count >= max_samples:
                    break
                
                # 获取文本
                caption = str(row.get('caption', '')).strip()
                if not caption:
                    skipped += 1
                    continue
                
                # 构建 JSON 记录
                record = {"text": caption}
                
                # 处理图像
                if 'image' in df.columns and pd.notna(row.get('image')):
                    image_val = row.get('image')
                    
                    if image_format in ['dict', 'bytes', 'string']:
                        # 从字典/字节/字符串数据中提取图像（失败会抛异常，跳过样本）
                        try:
                            image_path = self._extract_image_bytes(image_val, idx)
                            record["image"] = str(image_path.absolute())
                            image_exists_count += 1
                            image_extracted_count += 1
                        except Exception as e:
                            skipped += 1
                            image_missing_count += 1
                            continue
                    else:
                        # 文件路径格式
                        image_path = Path(str(image_val))
                        if image_path.exists():
                            record["image"] = str(image_path.absolute())
                            image_exists_count += 1
                        else:
                            skipped += 1
                            image_missing_count += 1
                            continue
                else:
                    skipped += 1
                    continue
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
                
                if count % 1000 == 0:
                    print(f"  Processed {count:,} samples...")
        
        print(f"\nConversion complete!")
        print(f"  Converted: {count:,} samples")
        print(f"    - With existing images: {image_exists_count:,}")
        if image_format in ['dict', 'bytes', 'string'] and image_extracted_count > 0:
            print(f"    - Images extracted: {image_extracted_count:,}")
        if skipped > 0:
            print(f"  Skipped: {skipped:,} samples")
            print(f"    - Missing images: {image_missing_count:,}")
            print(f"    - Missing captions: {skipped - image_missing_count:,}")
        print(f"  Output: {output_path}")
        if image_format in ['dict', 'bytes', 'string']:
            print(f"  Extracted images saved to: {self.images_output_dir}")
        
        # 检查是否有错误日志
        log_file = self.output_dir / "logs" / "image_extraction_errors.log"
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    error_count = content.count("ERROR - Sample")
                if error_count > 0:
                    print(f"\n  Note: {error_count} extraction errors logged to: {log_file}")
                    print(f"        Check the log file for detailed error information")
            except Exception as e:
                print(f"\n  Note: Error log exists but could not read: {e}")
                print(f"        Check log file manually: {log_file}")


def main():
    """主函数：处理 LAION Subset 数据集"""
    parser = argparse.ArgumentParser(
        description="LAION Subset 数据集探索脚本，并转换为 image-text pairs 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python makeindex_Laion-sub.py --data_dir data_formal_alin/laion-subset --output_dir data_formal_alin
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="LAION Subset 数据集目录路径（包含data/*.parquet文件）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录路径（用于保存统计信息和转换后的数据）"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出JSONL文件路径（默认: output_dir/laion_subset_train.jsonl）"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数（默认: 全部）"
    )
    
    parser.add_argument(
        "--skip_explore",
        action="store_true",
        help="跳过探索步骤，直接转换"
    )
    
    parser.add_argument(
        "--skip_visualize",
        action="store_true",
        help="跳过可视化步骤"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_file = Path(args.output_file) if args.output_file else output_dir / "laion_subset_train.jsonl"
    
    # 验证输入目录
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    data_dir_path = data_dir / "data"
    if not data_dir_path.exists():
        raise FileNotFoundError(f"数据子目录不存在: {data_dir_path}")
    
    parquet_files = list(data_dir_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到parquet文件: {data_dir_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("LAION Subset Dataset Processing")
    print("=" * 80)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    laion_subset_explorer = LaionSubsetExplorer(str(data_dir), str(output_dir))
    
    if not args.skip_explore:
        print("\n" + "=" * 80)
        print("STEP 1: Exploring LAION Subset Dataset")
        print("=" * 80)
        laion_subset_stats = laion_subset_explorer.explore()
        
        if not args.skip_visualize:
            laion_subset_explorer.visualize(laion_subset_stats)
        
        # 保存统计信息
        stats_file = output_dir / "laion_subset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({k: v for k, v in laion_subset_stats.items() if k != 'dataframe'}, 
                     f, indent=2, default=str)
        print(f"\nStatistics saved to: {stats_file}")
    else:
        # 如果跳过探索，需要先运行一次获取stats
        laion_subset_stats = laion_subset_explorer.explore()
    
    # 转换为 JSONL 格式
    print("\n" + "=" * 80)
    print("STEP 2: Converting to JSONL format")
    print("=" * 80)
    laion_subset_explorer.convert_to_jsonl(laion_subset_stats, str(output_file), max_samples=args.max_samples)
    
    print("\n" + "=" * 80)
    print("All tasks completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
