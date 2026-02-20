#!/usr/bin/env python3
"""
创建符合 SYSTEM_PROMPT_V3 格式的 Phase 2 示例数据

这个脚本生成符合 purrsight/LLM/prompts.py 中定义的 JSON Schema V3 的训练数据。
用于测试和演示，实际生产数据应该使用 auto_label_phase2_data.py 调用大模型API生成。

Usage:
    python scripts/create_phase2_sample_data.py --output data/phase2/train.jsonl
"""

import json
import argparse
from pathlib import Path

# 符合 JSON_SCHEMA_V3 的示例响应
SAMPLE_RESPONSES = [
    {
        "description": "平静坐姿的猫",
        "image": "../cat.png",
        "audio": None,
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": {
            "diagnostic": {
                "physical_markers": {
                    "ears": "forward",
                    "tail": "neutral",
                    "posture": "relaxed",
                    "vocalization": "silent"
                },
                "classification": {
                    "ethogram_group": "maintenance",
                    "affective_state": "content",
                    "arousal_level": "low",
                    "risk_rating": 1
                }
            },
            "behavioral_summary": "The cat displays a relaxed posture with ears forward and tail in neutral position. No signs of stress or aggression. The cat appears to be in a resting state, typical of maintenance behavior.",
            "human_actionable_insight": "您的猫咪目前处于放松和满足的状态。这是健康猫咪的正常表现，无需特别干预。继续提供安全舒适的环境即可。"
        }
    },
    {
        "description": "有人声的音频环境",
        "image": None,
        "audio": "../audio.m4a",
        "instruction": "Analyze the cat's behavior based on the auditory input. Consider the audio context and provide a structured JSON report.",
        "response": {
            "diagnostic": {
                "physical_markers": {
                    "ears": "alert",
                    "tail": "neutral",
                    "posture": "relaxed",
                    "vocalization": "silent"
                },
                "classification": {
                    "ethogram_group": "social_affiliative",
                    "affective_state": "content",
                    "arousal_level": "medium",
                    "risk_rating": 1
                }
            },
            "behavioral_summary": "Audio analysis indicates human voice presence in an indoor environment. The cat is likely in a familiar social setting with moderate arousal, showing attentiveness to human interaction without signs of distress.",
            "human_actionable_insight": "您的猫咪对人声保持警觉但不紧张，这表明它对家庭环境适应良好。继续保持温和的互动方式，有助于维持良好的人猫关系。"
        }
    },
    {
        "description": "多模态输入：图像+音频",
        "image": "../cat.png",
        "audio": "../audio.m4a",
        "instruction": "Analyze the cat's behavior based on the visual and auditory inputs. Provide a structured JSON report.",
        "response": {
            "diagnostic": {
                "physical_markers": {
                    "ears": "forward",
                    "tail": "neutral",
                    "posture": "relaxed",
                    "vocalization": "purr"
                },
                "classification": {
                    "ethogram_group": "social_affiliative",
                    "affective_state": "content",
                    "arousal_level": "low",
                    "risk_rating": 1
                }
            },
            "behavioral_summary": "The cat exhibits relaxed body language with forward-facing ears and neutral tail position. Combined with purring vocalization detected in audio, this indicates a positive social interaction. The cat is comfortable in the presence of humans and shows affiliative behavior.",
            "human_actionable_insight": "您的猫咪正在表达满足和亲近感。呼噜声是猫咪感到安全和快乐的明确信号。这是建立深厚人猫情感联系的好时机，可以轻柔地抚摸或陪伴它。"
        }
    },
    {
        "description": "警觉状态的猫（示例）",
        "image": "../cat.png",
        "audio": None,
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": {
            "diagnostic": {
                "physical_markers": {
                    "ears": "alert",
                    "tail": "upright",
                    "posture": "tense",
                    "vocalization": "silent"
                },
                "classification": {
                    "ethogram_group": "maintenance",
                    "affective_state": "anxious",
                    "arousal_level": "medium",
                    "risk_rating": 2
                }
            },
            "behavioral_summary": "The cat shows signs of heightened alertness with upright tail and tense posture. Ears are in alert position, scanning for potential threats. This suggests the cat is monitoring its environment for unfamiliar stimuli.",
            "human_actionable_insight": "您的猫咪目前处于警觉状态，可能察觉到了环境中的变化。建议观察是否有新的声音、气味或物体引起了它的注意。给予猫咪一些时间和空间来适应，避免突然的动作。"
        }
    },
    {
        "description": "潜在攻击性行为（示例）",
        "image": "../cat.png",
        "audio": None,
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": {
            "diagnostic": {
                "physical_markers": {
                    "ears": "flattened",
                    "tail": "lashing",
                    "posture": "crouched",
                    "vocalization": "hiss"
                },
                "classification": {
                    "ethogram_group": "agonistic",
                    "affective_state": "aggressive",
                    "arousal_level": "high",
                    "risk_rating": 4
                }
            },
            "behavioral_summary": "The cat displays clear defensive-aggressive signals: flattened ears, lashing tail, and crouched posture. Hissing vocalization indicates the cat feels threatened and is preparing to defend itself. This is a high-risk situation requiring immediate intervention.",
            "human_actionable_insight": "警告：您的猫咪正处于高度应激状态，表现出明显的防御性攻击信号。请立即停止任何可能引起威胁的行为，给猫咪足够的空间撤退。如果这种行为持续或频繁出现，建议咨询专业的动物行为学家或兽医。"
        }
    }
]


def create_sample_data(output_path: str, num_samples: int = 3):
    """
    创建符合 V3 Schema 的示例训练数据
    
    Args:
        output_path: 输出JSONL文件路径
        num_samples: 要生成的样本数量（从预定义样本中选择）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 选择前 num_samples 个样本
    selected_samples = SAMPLE_RESPONSES[:num_samples]
    
    print(f"Creating {len(selected_samples)} sample(s) for Phase 2 training...")
    print(f"Output: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(selected_samples, 1):
            # 构建JSONL行
            jsonl_entry = {
                "instruction": sample["instruction"],
                "response": json.dumps(sample["response"], ensure_ascii=False)
            }
            
            # 添加模态字段（如果存在）
            if sample["image"]:
                jsonl_entry["image"] = sample["image"]
            if sample["audio"]:
                jsonl_entry["audio"] = sample["audio"]
            
            # 写入文件
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
            
            print(f"  [{i}] {sample['description']}")
            if sample["image"]:
                print(f"      Image: {sample['image']}")
            if sample["audio"]:
                print(f"      Audio: {sample['audio']}")
    
    print(f"\n✓ Successfully created {len(selected_samples)} samples")
    print(f"\n示例响应格式:")
    print(json.dumps(selected_samples[0]["response"], ensure_ascii=False, indent=2))
    
    print(f"\n下一步:")
    print(f"1. 使用此数据测试训练: python train/train_llm/train_phase2.py --config config/phase2_example.yaml")
    print(f"2. 生产环境使用API标注: python scripts/auto_label_phase2_data.py --input_file <raw_data> --output_dir data/phase2")


def main():
    parser = argparse.ArgumentParser(
        description="创建符合 SYSTEM_PROMPT_V3 的 Phase 2 示例数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 创建3个示例（默认）
  python scripts/create_phase2_sample_data.py --output data/phase2/train.jsonl
  
  # 创建所有5个示例
  python scripts/create_phase2_sample_data.py --output data/phase2/train.jsonl --num_samples 5
  
  # 创建到不同目录
  python scripts/create_phase2_sample_data.py --output data/phase2_v3/train.jsonl
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/phase2/train.jsonl',
        help='输出JSONL文件路径 (默认: data/phase2/train.jsonl)'
    )
    
    parser.add_argument(
        '--num_samples', '-n',
        type=int,
        default=3,
        choices=range(1, 6),
        help='生成的样本数量 (1-5, 默认: 3)'
    )
    
    args = parser.parse_args()
    
    create_sample_data(args.output, args.num_samples)


if __name__ == "__main__":
    main()
