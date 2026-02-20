"""
Purr-Sight Unified Training Runner

Dispatches training tasks based on the selected phase.
"""

import argparse
import yaml
import sys
from dataclasses import fields
from typing import Dict, Any, Type

from purrsight.utils.logging import logger
from train.train_alignment.train_align_conf import AlignmentConfig
from train.train_alignment.train import train_model as train_phase1
from train.train_llm.train_llm_conf import LLMConfig
from train.train_llm.train import train_llm as train_phase2

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def dict_to_dataclass(cls: Type, data: Dict[str, Any]) -> Any:
    """
    Converts a dictionary to a dataclass instance, ignoring unknown keys.
    """
    valid_keys = {f.name for f in fields(cls)}
    filtered_data = {}
    for k, v in data.items():
        if k in valid_keys:
            filtered_data[k] = v
        else:
            # logger.debug(f"Ignoring config key '{k}' for {cls.__name__}")
            pass
    return cls(**filtered_data)

def main():
    parser = argparse.ArgumentParser(description="Purr-Sight Training Runner")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True, help="Training Phase (1: Alignment, 2: LLM)")
    parser.add_argument("--config", type=str, default="config/train_config.yaml", help="Path to configuration YAML")
    
    args = parser.parse_args()
    
    logger.info(f"Loading configuration from {args.config}")
    try:
        full_config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)
        
    common_cfg = full_config.get('common', {})
    
    if args.phase == 1:
        logger.info("==========================================")
        logger.info("   PHASE 1: Contrastive Alignment")
        logger.info("==========================================")
        
        phase_cfg = full_config.get('phase1', {})
        # Merge: Common settings are defaults, overridden by Phase settings
        merged_cfg = {**common_cfg, **phase_cfg}
        
        try:
            config_obj = dict_to_dataclass(AlignmentConfig, merged_cfg)
            train_phase1(config_obj)
        except Exception as e:
            logger.error(f"Phase 1 training failed: {e}", exc_info=True)
            sys.exit(1)
            
    elif args.phase == 2:
        logger.info("==========================================")
        logger.info("   PHASE 2: Multimodal Instruction Tuning")
        logger.info("==========================================")
        
        phase_cfg = full_config.get('phase2', {})
        merged_cfg = {**common_cfg, **phase_cfg}
        
        try:
            config_obj = dict_to_dataclass(LLMConfig, merged_cfg)
            train_phase2(config_obj)
        except Exception as e:
            logger.error(f"Phase 2 training failed: {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main()
