#!/usr/bin/env python3
"""
Generate more Phase 2 training data with diverse examples.

Creates synthetic training samples with proper V3 Schema JSON format.
"""

import json
from pathlib import Path

# Training samples with diverse scenarios
training_samples = [
    # Sample 1: Relaxed cat
    {
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": json.dumps({
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
        }, ensure_ascii=False),
        "image": "../cat.png"
    },
    
    # Sample 2: Alert cat with audio
    {
        "instruction": "Analyze the cat's behavior based on the auditory input. Consider the audio context and provide a structured JSON report.",
        "response": json.dumps({
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
        }, ensure_ascii=False),
        "audio": "../audio.m4a"
    },
    
    # Sample 3: Purring cat (multimodal)
    {
        "instruction": "Analyze the cat's behavior based on the visual and auditory inputs. Provide a structured JSON report.",
        "response": json.dumps({
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
        }, ensure_ascii=False),
        "image": "../cat.png",
        "audio": "../audio.m4a"
    },
    
    # Sample 4: Playful cat
    {
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": json.dumps({
            "diagnostic": {
                "physical_markers": {
                    "ears": "forward",
                    "tail": "upright",
                    "posture": "relaxed",
                    "vocalization": "chirp"
                },
                "classification": {
                    "ethogram_group": "social_affiliative",
                    "affective_state": "playful",
                    "arousal_level": "medium",
                    "risk_rating": 1
                }
            },
            "behavioral_summary": "The cat shows playful behavior with upright tail and forward ears. Chirping vocalization indicates excitement and readiness for interaction. Body posture is relaxed but alert, typical of play initiation.",
            "human_actionable_insight": "您的猫咪正处于玩耍的心情！这是与它互动的好时机。可以使用逗猫棒或玩具球与它玩耍，这有助于消耗精力并增进感情。"
        }, ensure_ascii=False),
        "image": "../cat.png"
    },
    
    # Sample 5: Anxious cat
    {
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": json.dumps({
            "diagnostic": {
                "physical_markers": {
                    "ears": "sideways",
                    "tail": "tucked",
                    "posture": "crouched",
                    "vocalization": "silent"
                },
                "classification": {
                    "ethogram_group": "agonistic",
                    "affective_state": "anxious",
                    "arousal_level": "high",
                    "risk_rating": 4
                }
            },
            "behavioral_summary": "The cat displays defensive body language with sideways ears and tucked tail. Crouched posture indicates fear or anxiety. The cat is in a heightened state of arousal and may be preparing to flee or defend itself.",
            "human_actionable_insight": "您的猫咪正处于焦虑状态。请给它一些空间和时间来平静下来。避免突然的动作或大声音。如果焦虑持续，建议咨询兽医排除健康问题。"
        }, ensure_ascii=False),
        "image": "../cat.png"
    },
    
    # Sample 6: Aggressive warning
    {
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": json.dumps({
            "diagnostic": {
                "physical_markers": {
                    "ears": "flattened",
                    "tail": "lashing",
                    "posture": "arched",
                    "vocalization": "hiss"
                },
                "classification": {
                    "ethogram_group": "agonistic",
                    "affective_state": "aggressive",
                    "arousal_level": "high",
                    "risk_rating": 5
                }
            },
            "behavioral_summary": "The cat exhibits clear aggressive signals with flattened ears, lashing tail, and arched posture. Hissing vocalization is a warning sign. The cat is in a highly aroused defensive state and may attack if provoked.",
            "human_actionable_insight": "警告：您的猫咪正在发出攻击警告！请立即停止接近，给它足够的空间。不要试图触摸或安抚它。如果攻击行为频繁出现，请咨询动物行为专家。"
        }, ensure_ascii=False),
        "image": "../cat.png"
    },
    
    # Sample 7: Grooming (maintenance)
    {
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": json.dumps({
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
            "behavioral_summary": "The cat is engaged in self-grooming behavior, a normal maintenance activity. Relaxed posture and forward ears indicate comfort and security. This is typical healthy cat behavior.",
            "human_actionable_insight": "您的猫咪正在进行日常理毛，这是健康和满足的标志。理毛行为有助于猫咪保持清洁和调节情绪。无需干预，让它自然完成即可。"
        }, ensure_ascii=False),
        "image": "../cat.png"
    },
    
    # Sample 8: Hunting mode (predatory)
    {
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": json.dumps({
            "diagnostic": {
                "physical_markers": {
                    "ears": "alert",
                    "tail": "neutral",
                    "posture": "crouched",
                    "vocalization": "silent"
                },
                "classification": {
                    "ethogram_group": "predatory",
                    "affective_state": "neutral",
                    "arousal_level": "high",
                    "risk_rating": 2
                }
            },
            "behavioral_summary": "The cat displays predatory behavior with alert ears and crouched posture. Silent and focused, indicating stalking or hunting mode. This is natural instinctive behavior, typically directed at toys or small moving objects.",
            "human_actionable_insight": "您的猫咪正在展现狩猎本能。这是正常的自然行为。可以提供互动玩具来满足它的狩猎需求，这有助于保持身心健康和活力。"
        }, ensure_ascii=False),
        "image": "../cat.png"
    },
    
    # Sample 9: Distressed cat
    {
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": json.dumps({
            "diagnostic": {
                "physical_markers": {
                    "ears": "flattened",
                    "tail": "tucked",
                    "posture": "tense",
                    "vocalization": "growl"
                },
                "classification": {
                    "ethogram_group": "agonistic",
                    "affective_state": "distressed",
                    "arousal_level": "high",
                    "risk_rating": 5
                }
            },
            "behavioral_summary": "The cat shows signs of severe distress with flattened ears, tucked tail, and tense posture. Growling indicates pain or extreme discomfort. This requires immediate attention.",
            "human_actionable_insight": "紧急：您的猫咪可能正在经历疼痛或极度不适！请立即联系兽医进行检查。在等待就医期间，保持环境安静，避免触碰可能引起疼痛的部位。"
        }, ensure_ascii=False),
        "image": "../cat.png"
    },
    
    # Sample 10: Friendly greeting
    {
        "instruction": "Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.",
        "response": json.dumps({
            "diagnostic": {
                "physical_markers": {
                    "ears": "forward",
                    "tail": "upright",
                    "posture": "relaxed",
                    "vocalization": "meow"
                },
                "classification": {
                    "ethogram_group": "social_affiliative",
                    "affective_state": "content",
                    "arousal_level": "medium",
                    "risk_rating": 1
                }
            },
            "behavioral_summary": "The cat displays friendly greeting behavior with upright tail and forward ears. Meowing is a social vocalization directed at humans. The cat is seeking attention or interaction.",
            "human_actionable_insight": "您的猫咪正在友好地向您打招呼！竖起的尾巴和叫声表示它想与您互动。这是增进感情的好时机，可以回应它的问候并给予关注。"
        }, ensure_ascii=False),
        "image": "../cat.png"
    }
]

def main():
    """Generate training data file."""
    output_path = Path("data/phase2/train_extended.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {len(training_samples)} training samples...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Generated {len(training_samples)} samples")
    print(f"📁 Saved to: {output_path}")
    
    # Validate each sample
    print("\n验证样本格式...")
    for i, sample in enumerate(training_samples, 1):
        try:
            # Parse the response JSON
            response_data = json.loads(sample['response'])
            
            # Check required fields
            assert 'diagnostic' in response_data
            assert 'physical_markers' in response_data['diagnostic']
            assert 'classification' in response_data['diagnostic']
            assert 'behavioral_summary' in response_data
            assert 'human_actionable_insight' in response_data
            
            print(f"  ✓ Sample {i}: Valid")
        except Exception as e:
            print(f"  ✗ Sample {i}: Invalid - {e}")
    
    print(f"\n完成！现在可以使用 train_extended.jsonl 重新训练模型")
    print(f"建议训练配置：")
    print(f"  - epochs: 10-20")
    print(f"  - batch_size: 2-4")
    print(f"  - learning_rate: 5e-5")

if __name__ == "__main__":
    main()
