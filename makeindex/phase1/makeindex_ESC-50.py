"""
ESC-50 数据集探索脚本，并转换为 audio-text pairs 格式

参考:
- ESC-50: https://github.com/karoldvl/ESC-50

功能:
1. 探索 ESC-50 数据集（audio-text pairs）
2. 生成统计信息和可视化
3. 转换为 train.jsonl 格式用于对齐训练
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体支持（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DatasetExplorer:
    """数据集探索器基类"""
    
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
        
    def explore(self) -> Dict:
        """探索数据集，返回统计信息"""
        raise NotImplementedError
        
    def visualize(self, stats: Dict) -> None:
        """可视化统计信息"""
        raise NotImplementedError
        
    def convert_to_jsonl(self, stats: Dict, output_path: str, max_samples: Optional[int] = None) -> None:
        """转换为 train.jsonl 格式"""
        raise NotImplementedError


class ESC50Explorer(DatasetExplorer):
    """ESC-50 数据集探索器"""
    
    def __init__(self, data_dir: str, output_dir: str):
        super().__init__(data_dir, output_dir)
        self.csv_path = self.data_dir / "meta" / "esc50.csv"
        self.audio_dir = self.data_dir / "audio"
        
    def explore(self) -> Dict:
        """探索 ESC-50 数据集"""
        print("\n" + "=" * 80)
        print("Exploring ESC-50 Dataset")
        print("=" * 80)
        
        stats = {}
        
        # 读取 CSV 元数据
        try:
            print(f"\nReading metadata from: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            stats['total_samples'] = len(df)
            stats['columns'] = df.columns.tolist()
            
            print(f"\nDataset Overview:")
            print(f"  Total samples: {stats['total_samples']:,}")
            print(f"  Columns: {stats['columns']}")
            
            # 类别统计
            if 'category' in df.columns:
                category_counts = df['category'].value_counts()
                stats['category_distribution'] = category_counts.to_dict()
                stats['num_categories'] = len(category_counts)
                
                print(f"\nCategory Statistics:")
                print(f"  Number of categories: {stats['num_categories']}")
                print(f"  Samples per category:")
                for cat, count in category_counts.items():
                    print(f"    {cat}: {count}")
            
            # Fold 分布
            if 'fold' in df.columns:
                fold_counts = df['fold'].value_counts().sort_index()
                stats['fold_distribution'] = fold_counts.to_dict()
                print(f"\nFold Distribution:")
                for fold, count in fold_counts.items():
                    print(f"  Fold {fold}: {count} samples")
            
            # ESC-10 子集统计
            if 'esc10' in df.columns:
                esc10_counts = df['esc10'].value_counts()
                stats['esc10_distribution'] = esc10_counts.to_dict()
                print(f"\nESC-10 Subset:")
                print(f"  ESC-10 samples: {esc10_counts.get(True, 0)}")
                print(f"  Non-ESC-10 samples: {esc10_counts.get(False, 0)}")
            
            # Target 分布
            if 'target' in df.columns:
                target_counts = df['target'].value_counts().sort_index()
                stats['target_distribution'] = {int(k): int(v) for k, v in target_counts.items()}
                stats['num_targets'] = len(target_counts)
                print(f"\nTarget Classes: {stats['num_targets']} unique targets")
            
            # 显示前几个样本
            print(f"\nSample entries (first 5):")
            for idx in range(min(5, len(df))):
                row = df.iloc[idx]
                print(f"\n  Sample {idx + 1}:")
                print(f"    Filename: {row['filename']}")
                print(f"    Category: {row.get('category', 'N/A')}")
                print(f"    Target: {row.get('target', 'N/A')}")
                print(f"    Fold: {row.get('fold', 'N/A')}")
            
            # 检查音频文件是否存在
            audio_files = list(self.audio_dir.glob("*.wav"))
            stats['audio_files_found'] = len(audio_files)
            stats['expected_audio_files'] = len(df)
            
            print(f"\nAudio Files:")
            print(f"  Expected: {stats['expected_audio_files']}")
            print(f"  Found: {stats['audio_files_found']}")
            
            if stats['audio_files_found'] != stats['expected_audio_files']:
                print(f"  Warning: Mismatch between expected and found audio files!")
            
            stats['dataframe'] = df
            
        except Exception as e:
            print(f"Error reading metadata: {e}")
            stats['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        return stats
    
    def visualize(self, stats: Dict) -> None:
        """可视化 ESC-50 数据集统计信息"""
        if 'error' in stats or 'dataframe' not in stats:
            print("Cannot visualize: missing data")
            return
        
        df = stats['dataframe']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ESC-50 Dataset Statistics', fontsize=16, fontweight='bold')
        
        # 1. Category distribution
        if 'category' in df.columns:
            ax1 = axes[0, 0]
            category_counts = df['category'].value_counts()
            ax1.barh(range(len(category_counts)), category_counts.values)
            ax1.set_yticks(range(len(category_counts)))
            ax1.set_yticklabels(category_counts.index, fontsize=8)
            ax1.set_xlabel('Count')
            ax1.set_title('Category Distribution')
            ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Fold distribution
        if 'fold' in df.columns:
            ax2 = axes[0, 1]
            fold_counts = df['fold'].value_counts().sort_index()
            ax2.bar(fold_counts.index, fold_counts.values, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Fold')
            ax2.set_ylabel('Count')
            ax2.set_title('Cross-Validation Fold Distribution')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Target distribution
        if 'target' in df.columns:
            ax3 = axes[1, 0]
            target_counts = df['target'].value_counts().sort_index()
            ax3.bar(target_counts.index, target_counts.values, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Target Class')
            ax3.set_ylabel('Count')
            ax3.set_title('Target Class Distribution')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. ESC-10 vs non-ESC-10
        if 'esc10' in df.columns:
            ax4 = axes[1, 1]
            esc10_counts = df['esc10'].value_counts()
            ax4.pie(esc10_counts.values, labels=['Non-ESC-10', 'ESC-10'], 
                   autopct='%1.1f%%', startangle=90)
            ax4.set_title('ESC-10 Subset Distribution')
        
        plt.tight_layout()
        output_path = self.output_dir / "esc50_statistics.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        plt.close()
    
    def _generate_text_description(self, category: str) -> str:
        """生成多样化的文本描述
        
        Args:
            category: 音频类别名称（如 "dog", "thunderstorm"）
            
        Returns:
            多样化的文本描述
        """
        # 将下划线替换为空格，使文本更自然
        category_clean = category.replace('_', ' ')
        
        # 多样化的句式模板
        templates = [
            f"a sound of {category_clean}",
            f"the sound of {category_clean}",
            f"{category_clean} sound",
            f"{category_clean} audio",
            f"audio of {category_clean}",
            f"recording of {category_clean}",
            f"a {category_clean} recording",
            f"the {category_clean} audio",
            f"{category_clean} noise",
            f"sound effect of {category_clean}",
            f"a {category_clean} sound effect",
            f"hearing {category_clean}",
            f"listening to {category_clean}",
            f"{category_clean} audio clip",
            f"an audio clip of {category_clean}",
        ]
        
        # 随机选择一个模板
        return random.choice(templates)
    
    def convert_to_jsonl(self, stats: Dict, output_path: str, max_samples: Optional[int] = None) -> None:
        """转换为 train.jsonl 格式"""
        if 'dataframe' not in stats:
            print("Error: No data to convert. Please run explore() first.")
            return
        
        df = stats['dataframe']
        
        print(f"\nConverting ESC-50 dataset to JSONL format...")
        print(f"Output path: {output_path}")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        skipped = 0
        
        # 设置随机种子以确保可重现性（可选）
        # random.seed(42)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                if max_samples and count >= max_samples:
                    break
                
                filename = row['filename']
                audio_path = self.audio_dir / filename
                
                # 检查音频文件是否存在（不存在直接跳过）
                if not audio_path.exists():
                    skipped += 1
                    continue
                
                # 构建多样化的文本描述
                category = row.get('category', 'unknown sound')
                text = self._generate_text_description(category)
                
                # 构建 JSON 记录
                record = {
                    "text": text,
                    "audio": str(audio_path.absolute())
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
                
                if count % 100 == 0:
                    print(f"  Processed {count:,} samples...")
        
        print(f"\nConversion complete!")
        print(f"  Converted: {count:,} samples")
        if skipped > 0:
            print(f"  Skipped (missing files): {skipped:,} samples")
        print(f"  Output: {output_path}")


def main():
    """主函数：处理 ESC-50 数据集"""
    parser = argparse.ArgumentParser(
        description="ESC-50 数据集探索脚本，并转换为 audio-text pairs 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python makeindex_ESC-50.py --data_dir data_formal_alin/ESC-50-master --output_dir data_formal_alin
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="ESC-50 数据集目录路径（包含meta/esc50.csv和audio/目录）"
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
        help="输出JSONL文件路径（默认: output_dir/esc50_train.jsonl）"
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
    output_file = Path(args.output_file) if args.output_file else output_dir / "esc50_train.jsonl"
    
    # 验证输入目录
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    csv_path = data_dir / "meta" / "esc50.csv"
    audio_dir = data_dir / "audio"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"元数据文件不存在: {csv_path}")
    
    if not audio_dir.exists():
        raise FileNotFoundError(f"音频目录不存在: {audio_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("ESC-50 Dataset Processing")
    print("=" * 80)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    esc50_explorer = ESC50Explorer(str(data_dir), str(output_dir))
    
    if not args.skip_explore:
        print("\n" + "=" * 80)
        print("STEP 1: Exploring ESC-50 Dataset")
        print("=" * 80)
        esc50_stats = esc50_explorer.explore()
        
        if not args.skip_visualize:
            esc50_explorer.visualize(esc50_stats)
        
        # 保存统计信息
        stats_file = output_dir / "esc50_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({k: v for k, v in esc50_stats.items() if k != 'dataframe'}, 
                     f, indent=2, default=str)
        print(f"\nStatistics saved to: {stats_file}")
    else:
        # 如果跳过探索，需要先运行一次获取stats
        esc50_stats = esc50_explorer.explore()
    
    # 转换为 JSONL 格式
    print("\n" + "=" * 80)
    print("STEP 2: Converting to JSONL format")
    print("=" * 80)
    esc50_explorer.convert_to_jsonl(esc50_stats, str(output_file), max_samples=args.max_samples)
    
    print("\n" + "=" * 80)
    print("All tasks completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
