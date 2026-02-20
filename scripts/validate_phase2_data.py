#!/usr/bin/env python3
"""
验证 Phase 2 数据格式是否符合 V3 Schema

Usage:
    python scripts/validate_phase2_data.py data/phase2/train.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any

# V3 Schema 验证规则
VALID_EARS = {"forward", "sideways", "flattened", "alert"}
VALID_TAIL = {"neutral", "tucked", "lashing", "upright", "puffed"}
VALID_POSTURE = {"relaxed", "crouched", "lateral_recumbent", "arched", "tense"}
VALID_VOCALIZATION = {"purr", "hiss", "growl", "chirp", "meow", "trill", "silent"}
VALID_ETHOGRAM = {"social_affiliative", "agonistic", "maintenance", "predatory"}
VALID_AFFECTIVE = {"content", "anxious", "aggressive", "playful", "distressed", "neutral"}
VALID_AROUSAL = {"low", "medium", "high"}


def validate_response_schema(response: Dict[str, Any], sample_id: int) -> list[str]:
    """
    验证response是否符合 JSON_SCHEMA_V3
    
    Returns:
        错误列表（空列表表示验证通过）
    """
    errors = []
    
    # 检查顶层结构
    if "diagnostic" not in response:
        errors.append(f"Sample {sample_id}: Missing 'diagnostic' field")
        return errors
    
    if "behavioral_summary" not in response:
        errors.append(f"Sample {sample_id}: Missing 'behavioral_summary' field")
    
    if "human_actionable_insight" not in response:
        errors.append(f"Sample {sample_id}: Missing 'human_actionable_insight' field")
    
    diagnostic = response.get("diagnostic", {})
    
    # 检查 physical_markers
    if "physical_markers" not in diagnostic:
        errors.append(f"Sample {sample_id}: Missing 'diagnostic.physical_markers'")
    else:
        markers = diagnostic["physical_markers"]
        
        if "ears" in markers and markers["ears"] not in VALID_EARS:
            errors.append(f"Sample {sample_id}: Invalid ears value: {markers['ears']}")
        
        if "tail" in markers and markers["tail"] not in VALID_TAIL:
            errors.append(f"Sample {sample_id}: Invalid tail value: {markers['tail']}")
        
        if "posture" in markers and markers["posture"] not in VALID_POSTURE:
            errors.append(f"Sample {sample_id}: Invalid posture value: {markers['posture']}")
        
        if "vocalization" in markers and markers["vocalization"] not in VALID_VOCALIZATION:
            errors.append(f"Sample {sample_id}: Invalid vocalization value: {markers['vocalization']}")
    
    # 检查 classification
    if "classification" not in diagnostic:
        errors.append(f"Sample {sample_id}: Missing 'diagnostic.classification'")
    else:
        classification = diagnostic["classification"]
        
        if "ethogram_group" in classification and classification["ethogram_group"] not in VALID_ETHOGRAM:
            errors.append(f"Sample {sample_id}: Invalid ethogram_group: {classification['ethogram_group']}")
        
        if "affective_state" in classification and classification["affective_state"] not in VALID_AFFECTIVE:
            errors.append(f"Sample {sample_id}: Invalid affective_state: {classification['affective_state']}")
        
        if "arousal_level" in classification and classification["arousal_level"] not in VALID_AROUSAL:
            errors.append(f"Sample {sample_id}: Invalid arousal_level: {classification['arousal_level']}")
        
        if "risk_rating" in classification:
            risk = classification["risk_rating"]
            if not isinstance(risk, int) or risk < 1 or risk > 5:
                errors.append(f"Sample {sample_id}: Invalid risk_rating: {risk} (must be 1-5)")
            
            # 验证 risk_rating 逻辑
            ethogram = classification.get("ethogram_group")
            affective = classification.get("affective_state")
            if ethogram == "agonistic" or affective in ["aggressive", "distressed"]:
                if risk < 4:
                    errors.append(f"Sample {sample_id}: risk_rating should be 4-5 for agonistic/distressed behavior (got {risk})")
    
    # 检查文本字段类型
    if "behavioral_summary" in response and not isinstance(response["behavioral_summary"], str):
        errors.append(f"Sample {sample_id}: behavioral_summary must be a string")
    
    if "human_actionable_insight" in response and not isinstance(response["human_actionable_insight"], str):
        errors.append(f"Sample {sample_id}: human_actionable_insight must be a string")
    
    return errors


def validate_jsonl_file(file_path: str) -> tuple[int, int, list[str]]:
    """
    验证JSONL文件
    
    Returns:
        (总样本数, 有效样本数, 错误列表)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return 0, 0, [f"File not found: {file_path}"]
    
    total_samples = 0
    valid_samples = 0
    all_errors = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_samples += 1
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue
            
            # 检查必需字段
            if "instruction" not in sample:
                all_errors.append(f"Line {line_num}: Missing 'instruction' field")
                continue
            
            if "response" not in sample:
                all_errors.append(f"Line {line_num}: Missing 'response' field")
                continue
            
            # 检查至少有一个模态
            if not any(k in sample for k in ["image", "audio", "video"]):
                all_errors.append(f"Line {line_num}: No modality field (image/audio/video)")
            
            # 解析和验证response
            try:
                response = json.loads(sample["response"])
            except json.JSONDecodeError as e:
                all_errors.append(f"Line {line_num}: Invalid response JSON - {e}")
                continue
            
            # 验证Schema
            schema_errors = validate_response_schema(response, line_num)
            all_errors.extend(schema_errors)
            
            if not schema_errors:
                valid_samples += 1
    
    return total_samples, valid_samples, all_errors


def main():
    parser = argparse.ArgumentParser(
        description="验证 Phase 2 数据格式",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'file',
        type=str,
        help='JSONL文件路径'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细错误信息'
    )
    
    args = parser.parse_args()
    
    print(f"Validating: {args.file}")
    print("=" * 80)
    
    total, valid, errors = validate_jsonl_file(args.file)
    
    if total == 0:
        print("❌ No samples found or file error")
        if errors:
            print(f"\nError: {errors[0]}")
        return 1
    
    print(f"Total samples: {total}")
    print(f"Valid samples: {valid}")
    print(f"Invalid samples: {total - valid}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} error(s):")
        if args.verbose:
            for error in errors:
                print(f"  - {error}")
        else:
            for error in errors[:5]:
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors (use --verbose to see all)")
        return 1
    else:
        print("\n✅ All samples are valid!")
        print("\nSchema compliance:")
        print("  ✓ All required fields present")
        print("  ✓ All enum values valid")
        print("  ✓ risk_rating logic correct")
        print("  ✓ JSON format valid")
        return 0


if __name__ == "__main__":
    exit(main())
