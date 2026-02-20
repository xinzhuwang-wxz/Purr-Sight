"""
Output Parser for Phase 2 Inference

Uses Pydantic to enforce structured JSON output conforming to V3 Schema.
Includes fallback parsing and validation.
"""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
import json
import re


# V3 Schema Pydantic Models

class PhysicalMarkers(BaseModel):
    """Physical markers of cat behavior."""
    ears: Literal["forward", "sideways", "flattened", "alert"] = Field(
        ..., 
        description="Ear position"
    )
    tail: Literal["neutral", "tucked", "lashing", "upright", "puffed"] = Field(
        ..., 
        description="Tail position"
    )
    posture: Literal["relaxed", "crouched", "lateral_recumbent", "arched", "tense"] = Field(
        ..., 
        description="Body posture"
    )
    vocalization: Literal["purr", "hiss", "growl", "chirp", "meow", "trill", "silent"] = Field(
        ..., 
        description="Vocalization type"
    )


class Classification(BaseModel):
    """Behavioral classification."""
    ethogram_group: Literal["social_affiliative", "agonistic", "maintenance", "predatory"] = Field(
        ..., 
        description="Ethogram behavioral group"
    )
    affective_state: Literal["content", "anxious", "aggressive", "playful", "distressed", "neutral"] = Field(
        ..., 
        description="Emotional/affective state"
    )
    arousal_level: Literal["low", "medium", "high"] = Field(
        ..., 
        description="Arousal level"
    )
    risk_rating: int = Field(
        ..., 
        ge=1, 
        le=5, 
        description="Risk rating from 1 (safe) to 5 (dangerous)"
    )
    
    @validator('risk_rating')
    def validate_risk_rating(cls, v, values):
        """Validate risk rating logic."""
        # If agonistic or distressed, risk should be 4-5
        if 'ethogram_group' in values and values['ethogram_group'] == 'agonistic':
            if v < 4:
                return 4  # Auto-correct to minimum safe value
        if 'affective_state' in values and values['affective_state'] == 'distressed':
            if v < 4:
                return 4  # Auto-correct to minimum safe value
        return v


class Diagnostic(BaseModel):
    """Diagnostic information including physical markers and classification."""
    physical_markers: PhysicalMarkers
    classification: Classification


class CatBehaviorAnalysis(BaseModel):
    """Complete cat behavior analysis conforming to V3 Schema."""
    diagnostic: Diagnostic
    behavioral_summary: str = Field(
        ..., 
        description="Objective description in English of visual/auditory cues"
    )
    human_actionable_insight: str = Field(
        ..., 
        description="Professional advice in Chinese for the owner"
    )
    raw_model_output: Optional[str] = Field(
        None,
        description="Original natural language output from the model"
    )


class OutputParser:
    """Parser for model output to structured JSON."""
    
    @staticmethod
    def parse_model_output(
        generated_text: str,
        strict: bool = False
    ) -> Dict[str, Any]:
        """Parse model output into structured JSON.
        
        Args:
            generated_text: Raw text output from the model
            strict: If True, raise exception on parsing failure. If False, return best-effort parse.
            
        Returns:
            Dictionary conforming to V3 Schema with additional 'raw_model_output' field
            
        Raises:
            ValueError: If strict=True and parsing fails
        """
        # Try to extract JSON from the text
        json_data = OutputParser._extract_json(generated_text)
        
        if json_data:
            try:
                # Validate with Pydantic
                analysis = CatBehaviorAnalysis(**json_data)
                # Add raw output
                result = analysis.dict()
                result['raw_model_output'] = generated_text
                return result
            except Exception as e:
                if strict:
                    raise ValueError(f"Failed to validate JSON against V3 Schema: {e}")
                # Fall through to fallback parsing
        
        # Fallback: Try to construct from text
        if not strict:
            return OutputParser._fallback_parse(generated_text)
        else:
            raise ValueError("No valid JSON found in model output")
    
    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from text.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Parsed JSON dictionary or None
        """
        # Try to find JSON pattern
        json_patterns = [
            r'\{[^{}]*"diagnostic"[^{}]*\{.*?\}.*?\}',  # Look for diagnostic key
            r'\{.*?\}',  # Any JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json_str = match.group()
                    # Try to parse
                    data = json.loads(json_str)
                    # Check if it has the required structure
                    if 'diagnostic' in data:
                        return data
                except json.JSONDecodeError:
                    continue
        
        return None
    
    @staticmethod
    def _fallback_parse(text: str) -> Dict[str, Any]:
        """Fallback parser when JSON extraction fails.
        
        Creates a best-effort structured output with default values.
        
        Args:
            text: Raw model output
            
        Returns:
            Dictionary with V3 Schema structure and default values
        """
        # Analyze text for keywords to infer behavior
        text_lower = text.lower()
        
        # Infer affective state
        if any(word in text_lower for word in ['calm', 'relaxed', 'peaceful', 'content']):
            affective_state = "content"
            arousal_level = "low"
            risk_rating = 1
        elif any(word in text_lower for word in ['anxious', 'nervous', 'worried', 'uncertain']):
            affective_state = "anxious"
            arousal_level = "medium"
            risk_rating = 2
        elif any(word in text_lower for word in ['aggressive', 'attack', 'bite', 'scratch']):
            affective_state = "aggressive"
            arousal_level = "high"
            risk_rating = 5
        elif any(word in text_lower for word in ['playful', 'play', 'energetic']):
            affective_state = "playful"
            arousal_level = "medium"
            risk_rating = 1
        elif any(word in text_lower for word in ['distressed', 'pain', 'injured', 'hurt']):
            affective_state = "distressed"
            arousal_level = "high"
            risk_rating = 5
        else:
            affective_state = "neutral"
            arousal_level = "low"
            risk_rating = 2
        
        # Infer ethogram group
        if any(word in text_lower for word in ['social', 'friendly', 'affection', 'rub']):
            ethogram_group = "social_affiliative"
        elif any(word in text_lower for word in ['aggressive', 'fight', 'defend', 'attack']):
            ethogram_group = "agonistic"
        elif any(word in text_lower for word in ['eat', 'groom', 'sleep', 'rest']):
            ethogram_group = "maintenance"
        elif any(word in text_lower for word in ['hunt', 'stalk', 'pounce', 'chase']):
            ethogram_group = "predatory"
        else:
            ethogram_group = "maintenance"
        
        # Create structured output with defaults
        result = {
            "diagnostic": {
                "physical_markers": {
                    "ears": "forward",
                    "tail": "neutral",
                    "posture": "relaxed",
                    "vocalization": "silent"
                },
                "classification": {
                    "ethogram_group": ethogram_group,
                    "affective_state": affective_state,
                    "arousal_level": arousal_level,
                    "risk_rating": risk_rating
                }
            },
            "behavioral_summary": f"Analysis based on text description. {text[:200]}...",
            "human_actionable_insight": "模型输出未包含结构化JSON，这是基于文本关键词的推断结果。建议重新训练模型以生成符合V3 Schema的输出。",
            "raw_model_output": text,
            "parsing_note": "Fallback parsing used - model did not generate structured JSON"
        }
        
        return result
    
    @staticmethod
    def format_output(
        analysis: Dict[str, Any],
        include_raw: bool = True,
        chinese_summary: bool = True
    ) -> str:
        """Format analysis for human-readable display.
        
        Args:
            analysis: Parsed analysis dictionary
            include_raw: Whether to include raw model output
            chinese_summary: Whether to include Chinese insights
            
        Returns:
            Formatted string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("猫咪行为分析报告 / Cat Behavior Analysis Report")
        lines.append("=" * 80)
        lines.append("")
        
        # Physical Markers
        lines.append("【物理标记 / Physical Markers】")
        pm = analysis['diagnostic']['physical_markers']
        lines.append(f"  耳朵 Ears:        {pm['ears']}")
        lines.append(f"  尾巴 Tail:        {pm['tail']}")
        lines.append(f"  姿势 Posture:     {pm['posture']}")
        lines.append(f"  发声 Vocalization: {pm['vocalization']}")
        lines.append("")
        
        # Classification
        lines.append("【行为分类 / Classification】")
        cls = analysis['diagnostic']['classification']
        lines.append(f"  行为组 Ethogram:    {cls['ethogram_group']}")
        lines.append(f"  情感状态 Affective:  {cls['affective_state']}")
        lines.append(f"  唤醒水平 Arousal:    {cls['arousal_level']}")
        lines.append(f"  风险评级 Risk:       {cls['risk_rating']}/5")
        lines.append("")
        
        # Behavioral Summary
        lines.append("【行为总结 / Behavioral Summary】")
        lines.append(f"  {analysis['behavioral_summary']}")
        lines.append("")
        
        # Human Actionable Insight
        if chinese_summary:
            lines.append("【专家建议 / Expert Advice】")
            lines.append(f"  {analysis['human_actionable_insight']}")
            lines.append("")
        
        # Raw output
        if include_raw and 'raw_model_output' in analysis and analysis['raw_model_output']:
            lines.append("【原始输出 / Raw Model Output】")
            lines.append(f"  {analysis['raw_model_output'][:500]}...")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Convenience functions

def parse_and_validate(generated_text: str, strict: bool = False) -> Dict[str, Any]:
    """Parse and validate model output.
    
    Args:
        generated_text: Raw model output
        strict: Whether to enforce strict validation
        
    Returns:
        Validated dictionary conforming to V3 Schema
    """
    return OutputParser.parse_model_output(generated_text, strict=strict)


def create_default_analysis(text_description: str = "") -> Dict[str, Any]:
    """Create a default analysis structure.
    
    Useful for testing or when model output is unavailable.
    
    Args:
        text_description: Optional text to include in summary
        
    Returns:
        Default analysis dictionary
    """
    return {
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
        "behavioral_summary": f"Default analysis. {text_description}",
        "human_actionable_insight": "这是一个默认的分析结果。请提供更多信息以获得准确的行为分析。",
        "raw_model_output": None
    }
