#!/usr/bin/env python3
"""
Purr-Sight Inference Module

End-to-end inference pipeline for multi-modal animal behavior analysis.
Supports video, image, and text inputs, outputs structured JSON.

Usage:
    # Video input
    python inference_module.py --video cat_video.mp4 --checkpoint model.pt
    
    # Image input
    python inference_module.py --image cat_photo.jpg --checkpoint model.pt
    
    # Text input (scene description)
    python inference_module.py --text "A cat is sitting on a windowsill" --checkpoint model.pt
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import preprocessing modules
from purrsight.preprocess.image import _ImageProcessor
from purrsight.preprocess.audio import _AudioProcessor
from purrsight.preprocess.text import _TextProcessor

# Import model components
from train.train_llm.multimodal_llm_module import MultiModalLLMModule
from train.train_llm.checkpoint_loader import CheckpointLoader

# Import output parser
from purrsight.LLM.output_parser import OutputParser


class PurrSightInference:
    """End-to-end inference pipeline for Purr-Sight model."""
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        local_files_only: bool = True
    ):
        """Initialize inference pipeline.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on (default: auto-detect)
            local_files_only: Use only local model files (no HuggingFace downloads)
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Initializing Purr-Sight Inference on device: {self.device}")
        
        # Initialize preprocessors
        self.image_preprocessor = _ImageProcessor()
        self.audio_preprocessor = _AudioProcessor()
        self.text_preprocessor = _TextProcessor()
        
        # Load model
        self.model = self._load_model(local_files_only=local_files_only)
        self.model.eval()
        
        print("✅ Inference pipeline initialized successfully")
    
    def _load_model(self, local_files_only: bool = True) -> nn.Module:
        """Load trained model from checkpoint.
        
        Args:
            local_files_only: Use only local model files
            
        Returns:
            Loaded model in evaluation mode
        """
        print(f"Loading model from: {self.checkpoint_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract configuration
        config = checkpoint.get('config', {})
        
        # Create model instance from config
        model = self._create_model_from_config(config, local_files_only=local_files_only)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(self.device)
        
        print(f"✅ Model loaded successfully")
        
        return model
    
    def _create_model_from_config(self, config: Dict[str, Any], local_files_only: bool = True) -> nn.Module:
        """Create model instance from configuration.
        
        Args:
            config: Model configuration dictionary
            local_files_only: Use only local model files
            
        Returns:
            Model instance
        """
        from purrsight.LLM.model import PurrSightMMLLM
        
        # Extract model configuration
        llm_model_path = config.get('llm_model_name', 'models/Qwen2.5-0.5B-Instruct')
        phase1_checkpoint = config.get('phase1_checkpoint_path', None)
        
        # Create LoRA config
        lora_config = {
            'enabled': True,
            'r': config.get('lora_r', 16),
            'lora_alpha': config.get('lora_alpha', 32),
            'lora_dropout': config.get('lora_dropout', 0.05),
            'target_modules': config.get('lora_target_modules', None),
            'task_type': 'CAUSAL_LM',
            'inference_mode': True  # Important for inference
        }
        
        # Create projector config
        projector_config = {
            'hidden_dim': config.get('projector_hidden_dim', 2048),
            'num_tokens': 4
        }
        
        # Initialize model
        model = PurrSightMMLLM(
            llm_model_path=llm_model_path,
            aligner_weights_path=phase1_checkpoint,
            freeze_encoders=True,
            freeze_projector=False,
            freeze_llm=False,
            lora_config=lora_config,
            projector_config=projector_config
        )
        
        return model
    
    def infer_from_video(
        self,
        video_path: Union[str, Path],
        extract_audio: bool = True,
        sample_frames: int = 8
    ) -> Dict[str, Any]:
        """Run inference on video input.
        
        Args:
            video_path: Path to video file
            extract_audio: Whether to extract and process audio
            sample_frames: Number of frames to sample from video
            
        Returns:
            Structured JSON output with behavior analysis
        """
        video_path = Path(video_path)
        print(f"Processing video: {video_path.name}")
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract frames from video
        frames = self._extract_video_frames(video_path, num_frames=sample_frames)
        
        # Process frames through image preprocessor
        image_features = []
        for frame in frames:
            processed_frame = self.image_preprocessor.preprocess(frame)
            image_features.append(processed_frame)
        
        # Stack frames
        image_tensor = torch.stack(image_features).to(self.device)
        
        # Extract and process audio if requested
        audio_tensor = None
        if extract_audio:
            audio_path = self._extract_audio_from_video(video_path)
            if audio_path and audio_path.exists():
                audio_tensor = self.audio_preprocessor.preprocess(str(audio_path))
                audio_tensor = audio_tensor.to(self.device)
        
        # Create text prompt
        text_prompt = "Analyze the animal behavior in this video and provide a structured description."
        text_tensor = self.text_preprocessor.preprocess(text_prompt)
        text_tensor = text_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self._run_model_inference(
                image=image_tensor,
                audio=audio_tensor,
                text=text_tensor
            )
        
        # Parse output to JSON
        result = self._parse_model_output(output, input_type='video', input_path=video_path)
        
        return result
    
    def infer_from_image(
        self,
        image_path: Union[str, Path],
        context_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run inference on image input.
        
        Args:
            image_path: Path to image file
            context_text: Optional context or question about the image
            
        Returns:
            Structured JSON output with behavior analysis
        """
        image_path = Path(image_path)
        print(f"Processing image: {image_path.name}")
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Preprocess image
        from PIL import Image
        pil_image = Image.open(str(image_path))
        processed_image = self.image_preprocessor.process_image(pil_image)
        # Convert numpy array to tensor and add batch dimension
        image_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(self.device)
        
        # Create text prompt with JSON schema requirement and few-shot example
        if context_text is None:
            text_prompt = """Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.

Example output format:
{"diagnostic": {"physical_markers": {"ears": "forward", "tail": "neutral", "posture": "relaxed", "vocalization": "silent"}, "classification": {"ethogram_group": "maintenance", "affective_state": "content", "arousal_level": "low", "risk_rating": 1}}, "behavioral_summary": "The cat displays relaxed body language.", "human_actionable_insight": "您的猫咪处于放松状态。"}

Now analyze this image and output JSON only:"""
        else:
            text_prompt = context_text
        
        # Run inference
        generated_text = self._run_model_inference(
            image=image_tensor,
            audio=None,
            text_prompt=text_prompt
        )
        
        # Parse output to JSON
        result = self._parse_model_output(generated_text, input_type='image', input_path=image_path)
        
        return result
    
    def infer_from_text(
        self,
        text_description: str
    ) -> Dict[str, Any]:
        """Run inference on text description.
        
        Args:
            text_description: Text description of scene or animal state
            
        Returns:
            Structured JSON output with behavior analysis
        """
        print(f"Processing text: {text_description[:50]}...")
        
        # Create prompt with few-shot example
        full_prompt = f"""Analyze the cat's behavior based on this description. Output valid JSON only.

Example output format:
{{"diagnostic": {{"physical_markers": {{"ears": "forward", "tail": "neutral", "posture": "relaxed", "vocalization": "silent"}}, "classification": {{"ethogram_group": "maintenance", "affective_state": "content", "arousal_level": "low", "risk_rating": 1}}}}, "behavioral_summary": "The cat displays relaxed body language.", "human_actionable_insight": "您的猫咪处于放松状态。"}}

Description: {text_description}

Output JSON only:"""
        
        # Run inference (text-only mode)
        generated_text = self._run_model_inference(
            image=None,
            audio=None,
            text_prompt=full_prompt
        )
        
        # Parse output to JSON
        result = self._parse_model_output(generated_text, input_type='text', input_text=text_description)
        
        return result
    
    def _extract_video_frames(
        self,
        video_path: Path,
        num_frames: int = 8
    ) -> List[torch.Tensor]:
        """Extract frames from video file.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of frame tensors
        """
        # Placeholder implementation
        # Actual implementation would use cv2 or torchvision to extract frames
        print(f"  Extracting {num_frames} frames from video...")
        
        # For now, return dummy frames
        frames = []
        for i in range(num_frames):
            # Create dummy frame (3, 224, 224)
            frame = torch.randn(3, 224, 224)
            frames.append(frame)
        
        return frames
    
    def _extract_audio_from_video(self, video_path: Path) -> Optional[Path]:
        """Extract audio track from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file or None
        """
        # Placeholder implementation
        # Actual implementation would use ffmpeg or similar
        print("  Extracting audio from video...")
        return None
    
    def _run_model_inference(
        self,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        text_prompt: str = "Analyze the animal behavior and provide a structured description."
    ) -> str:
        """Run model inference with multi-modal inputs.
        
        Args:
            image: Image tensor (optional)
            audio: Audio tensor (optional)
            text_prompt: Text prompt for the model
            
        Returns:
            Generated text response
        """
        # Prepare inputs dictionary
        inputs = {}
        
        # Add image if provided, otherwise use dummy tensor
        if image is not None:
            inputs['image'] = image
        else:
            # Create dummy image tensor (1, 3, 224, 224)
            inputs['image'] = torch.zeros(1, 3, 224, 224, device=self.device)
        
        # Add audio if provided, otherwise use dummy tensor
        if audio is not None:
            inputs['audio'] = audio
        else:
            # Create dummy audio tensor (1, 64, 256) - mel spectrogram shape
            inputs['audio'] = torch.zeros(1, 64, 256, device=self.device)
        
        # Tokenize text prompt
        tokenizer = self.model.tokenizer
        text_tokens = tokenizer(
            text_prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        inputs['input_ids'] = text_tokens['input_ids'].to(self.device)
        inputs['attention_mask'] = text_tokens['attention_mask'].to(self.device)
        
        # Run model forward pass to get multimodal embeddings
        with torch.no_grad():
            # Get multimodal embeddings
            mm_embeds = self.model.encode_multimodal(inputs)
            
            # Get text embeddings
            if hasattr(self.model.llm, 'model'):
                if hasattr(self.model.llm.model, 'embed_tokens'):
                    word_embeds = self.model.llm.model.embed_tokens(inputs['input_ids'])
                else:
                    word_embeds = self.model.llm.get_input_embeddings()(inputs['input_ids'])
            else:
                word_embeds = self.model.llm.get_input_embeddings()(inputs['input_ids'])
            
            # Concatenate multimodal and text embeddings
            inputs_embeds = torch.cat([mm_embeds, word_embeds], dim=1)
            
            # Adjust attention mask
            B, M, _ = mm_embeds.shape
            mm_mask = torch.ones(B, M, device=inputs['attention_mask'].device, dtype=inputs['attention_mask'].dtype)
            attention_mask = torch.cat([mm_mask, inputs['attention_mask']], dim=1)
            
            # Generate response
            output_ids = self.model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode output
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return generated_text
    
    def _parse_model_output(
        self,
        generated_text: str,
        input_type: str,
        input_path: Optional[Path] = None,
        input_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Parse model output into structured JSON using Pydantic validation.
        
        Args:
            generated_text: Generated text from model
            input_type: Type of input ('video', 'image', 'text')
            input_path: Path to input file (for video/image)
            input_text: Input text (for text mode)
            
        Returns:
            Structured JSON dictionary conforming to V3 Schema
        """
        # Use OutputParser to parse and validate
        try:
            analysis = OutputParser.parse_model_output(generated_text, strict=False)
        except Exception as e:
            print(f"Warning: Failed to parse model output: {e}")
            # Use fallback
            analysis = OutputParser._fallback_parse(generated_text)
        
        # Create result with metadata
        result = {
            'timestamp': datetime.now().isoformat(),
            'input_type': input_type,
            'model_checkpoint': str(self.checkpoint_path),
            'metadata': {
                'model_version': '2.0',
                'phase': 'phase2',
                'schema_version': 'V3'
            }
        }
        
        if input_path:
            result['input_file'] = str(input_path)
        
        if input_text:
            result['input_text'] = input_text
        
        # Add the parsed analysis
        result['analysis'] = analysis
        
        return result
    
    def save_result(
        self,
        result: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """Save inference result to JSON file.
        
        Args:
            result: Inference result dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Result saved to: {output_path}")
    
    def print_formatted_result(self, result: Dict[str, Any]) -> None:
        """Print formatted analysis result.
        
        Args:
            result: Inference result dictionary
        """
        if 'analysis' in result:
            formatted = OutputParser.format_output(
                result['analysis'],
                include_raw=True,
                chinese_summary=True
            )
            print(formatted)


def main():
    parser = argparse.ArgumentParser(description='Purr-Sight Inference Module')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--text', help='Text description')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Validate inputs
    input_count = sum([args.video is not None, args.image is not None, args.text is not None])
    if input_count == 0:
        print("Error: Must provide one of --video, --image, or --text")
        sys.exit(1)
    if input_count > 1:
        print("Error: Can only provide one input type at a time")
        sys.exit(1)
    
    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Initialize inference pipeline
    try:
        inference = PurrSightInference(
            checkpoint_path=args.checkpoint,
            device=device
        )
    except Exception as e:
        print(f"❌ Failed to initialize inference pipeline: {e}")
        sys.exit(1)
    
    # Run inference based on input type
    try:
        if args.video:
            result = inference.infer_from_video(args.video)
        elif args.image:
            result = inference.infer_from_image(args.image)
        elif args.text:
            result = inference.infer_from_text(args.text)
        
        # Print result
        print("\n" + "=" * 80)
        print("INFERENCE RESULT")
        print("=" * 80)
        print(json.dumps(result, indent=2))
        print("=" * 80)
        
        # Save result if output path provided
        if args.output:
            inference.save_result(result, args.output)
        
        print("\n✅ Inference completed successfully")
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
