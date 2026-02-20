"""PurrSight Multimodal LLM Model.

Combines:
1. Pre-trained Encoders (Image, Audio) from Phase 1
2. Pre-trained Aligner from Phase 1
3. Multimodal Projector (Phase 2)
4. LLM Backbone (e.g., Qwen, Llama)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from purrsight.encoder import _ImageEncoder, _AudioEncoder
from purrsight.alignment import ContrastiveAligner
from purrsight.LLM.projectors import MultimodalProjector
from purrsight.config import Modality, FeatureKey
from purrsight.utils.logging import logger

class PurrSightMMLLM(nn.Module):
    """PurrSight Multimodal LLM Architecture.
    
    Integrates pre-trained encoders and aligners with an LLM via a projector.
    Supports freezing various components and LoRA fine-tuning.

    Attributes:
        image_encoder: Pre-trained image encoder.
        audio_encoder: Pre-trained audio encoder.
        aligner: Contrastive aligner from Phase 1.
        llm: The causal language model.
        tokenizer: Tokenizer for the LLM.
        projector: Multimodal projector mapping aligned features to LLM tokens.
    """

    def __init__(
        self,
        llm_model_path: str,
        aligner_weights_path: Optional[str] = None,
        freeze_encoders: bool = True,
        freeze_projector: bool = False,
        freeze_llm: bool = False,
        lora_config: Optional[dict] = None,
        projector_config: Optional[dict] = None,
    ):
        """Initializes the PurrSightMMLLM.

        Args:
            llm_model_path: Path to the pre-trained LLM weights.
            aligner_weights_path: Path to the Phase 1 checkpoint (optional).
            freeze_encoders: Whether to freeze image/audio encoders and aligner. Defaults to True.
            freeze_projector: Whether to freeze the projector. Defaults to False.
            freeze_llm: Whether to freeze the LLM backbone. Defaults to False.
            lora_config: Configuration for LoRA fine-tuning.
            projector_config: Configuration for the projector (e.g., num_tokens).
        """
        super().__init__()
        
        # 1. Load Encoders (Image & Audio)
        # Text encoder from Phase 1 is not needed for generation, as LLM has its own embedding
        logger.info("Loading Encoders...")
        self.image_encoder = _ImageEncoder()
        self.audio_encoder = _AudioEncoder()
        
        # 2. Load Aligner (Phase 1)
        logger.info("Loading Aligner...")
        # Get dimensions from encoders
        image_dim = self.image_encoder.feature_dim
        audio_dim = 2048 # Cnn14 output
        text_dim = 384   # MiniLM output (needed for aligner init, though not used for gen)
        
        self.aligner = ContrastiveAligner(
            text_input_dim=text_dim,
            image_input_dim=image_dim,
            audio_input_dim=audio_dim,
            output_dim=512, # Aligned dim
            use_temperature_scaling=True
        )
        
        # Load Phase 1 weights if provided
        if aligner_weights_path:
            logger.info(f"Loading Phase 1 weights from {aligner_weights_path}")
            try:
                state_dict = torch.load(aligner_weights_path, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # CRITICAL FIX: Load weights directly into self.aligner.
                # Phase 1 saves "aligner.state_dict()", so keys are like "projection_heads...".
                # Phase 2 model has "self.aligner", so it expects keys like "aligner.projection_heads..."
                # if we use self.load_state_dict().
                # Instead, we load directly into the submodule to match keys automatically.
                msg = self.aligner.load_state_dict(state_dict, strict=False)
                logger.info(f"Aligner weights loaded successfully: {msg}")
                
            except Exception as e:
                logger.warning(f"Failed to load aligner weights: {e}")

        # 3. Load LLM
        logger.info(f"Loading LLM from {llm_model_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            torch_dtype=torch.float32,  # Use float32 for consistency
            trust_remote_code=True,
            # Don't use device_map="auto" - let PyTorch Lightning handle device placement
            # device_map="auto" can cause device mismatch issues with MPS
        )
        # Optimization: Enable Gradient Checkpointing.
        # Significantly reduces VRAM usage by trading compute for memory.
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
            logger.info("Gradient Checkpointing enabled for LLM")

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path,
            trust_remote_code=True,
            padding_side="right" # Usually right for training
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 4. Initialize Projector
        logger.info("Initializing Projector...")
        llm_hidden_dim = self.llm.config.hidden_size
        proj_cfg = projector_config or {}
        self.projector = MultimodalProjector(
            input_dim=512, # Aligned feature dim
            llm_dim=llm_hidden_dim,
            num_tokens=proj_cfg.get('num_tokens', 4),
            hidden_dim=proj_cfg.get('hidden_dim', 2048)
        )
        
        # 5. Apply Freezing / LoRA
        if freeze_encoders:
            self._freeze_module(self.image_encoder)
            self._freeze_module(self.audio_encoder)
            self._freeze_module(self.aligner)
            
        if freeze_llm and not lora_config:
            self._freeze_module(self.llm)
            
        # Apply LoRA if enabled
        if lora_config and lora_config.get('enabled', False):
            from peft import get_peft_model, LoraConfig, TaskType
            logger.info("Applying LoRA to LLM...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config.get('r', 16),
                lora_alpha=lora_config.get('lora_alpha', 32),
                lora_dropout=lora_config.get('lora_dropout', 0.05),
                target_modules=lora_config.get('target_modules', None)
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
            
        if freeze_projector:
            self._freeze_module(self.projector)
            
    def _freeze_module(self, module: nn.Module):
        """Freezes parameters of a module.
        
        Args:
            module: The module to freeze.
        """
        module.eval() # Critical: Set to eval mode for BatchNorm/Dropout.
        for param in module.parameters():
            param.requires_grad = False
            
    def encode_multimodal(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encodes Image/Audio -> Aligned Features -> Projected Soft Prompts.

        Args:
            inputs: Dictionary containing 'image' and/or 'audio' tensors.

        Returns:
            Concatenated multimodal tokens (B, Total_Tokens, LLM_Dim).
        """
        encoder_outputs = {}
        device = self.projector.mlp[0].weight.device # Get correct device
        
        # 1. Encode Images
        if FeatureKey.IMAGE in inputs and inputs[FeatureKey.IMAGE] is not None:
            # Assuming input is (B, 16, 3, H, W) or (B, C, H, W)
            # Reusing logic from Phase 1 LightningModule would be ideal, 
            # but for simplicity we implement basic forward here
            # For 16-frames:
            img = inputs[FeatureKey.IMAGE].to(device)
            if img.dim() == 5: # (B, T, C, H, W)
                B, T, C, H, W = img.shape
                img_flat = img.view(B*T, C, H, W)
                feats = self.image_encoder(img_flat)
                feats = feats.view(B, T, -1).mean(dim=1) # Simple mean pooling for now
            else:
                feats = self.image_encoder(img)
            encoder_outputs[Modality.IMAGE.value] = feats
            
        # 2. Encode Audio
        if FeatureKey.AUDIO in inputs and inputs[FeatureKey.AUDIO] is not None:
            aud = inputs[FeatureKey.AUDIO].to(device)
            # Redundancy Fix: Only run encoder if not empty/padding.
            if aud.sum().abs() > 1e-6:
                encoder_outputs[Modality.AUDIO.value] = self.audio_encoder(aud)
            
        # 3. Align
        # We need modality presence
        modality_presence = {
            k: v is not None for k, v in encoder_outputs.items()
        }
        
        # Handle missing modalities by adding zero tensors
        batch_size = list(encoder_outputs.values())[0].shape[0] if encoder_outputs else 1
        
        if Modality.IMAGE.value not in encoder_outputs:
            encoder_outputs[Modality.IMAGE.value] = torch.zeros(batch_size, self.image_encoder.feature_dim, device=device)
        if Modality.AUDIO.value not in encoder_outputs:
            encoder_outputs[Modality.AUDIO.value] = torch.zeros(batch_size, 2048, device=device)
        if Modality.TEXT.value not in encoder_outputs:
             encoder_outputs[Modality.TEXT.value] = torch.zeros(batch_size, 384, device=device)
             
        aligned_feats = self.aligner(encoder_outputs, modality_presence)
        
        # 4. Project
        # aligned_feats is Dict[str, Tensor], we want to concat or project individually?
        # The projector takes (B, 512).
        # We usually fuse them. For now, let's project Image and Audio separately and concat tokens.
        
        projected_tokens = []
        
        # Image Tokens
        img_aligned = aligned_feats[Modality.IMAGE.value]
        img_tokens = self.projector(img_aligned) # (B, N, D)
        projected_tokens.append(img_tokens)
        
        # Audio Tokens
        aud_aligned = aligned_feats[Modality.AUDIO.value]
        aud_tokens = self.projector(aud_aligned) # (B, N, D)
        projected_tokens.append(aud_tokens)
        
        # Concat: (B, 2*N, D)
        multimodal_embeds = torch.cat(projected_tokens, dim=1)
        
        return multimodal_embeds

    def forward(self, inputs: Dict[str, torch.Tensor], labels: torch.Tensor = None):
        """Forward pass for the multimodal LLM.
        
        Args:
            inputs: Dictionary containing 'image', 'audio', 'input_ids', 'attention_mask'.
            labels: Labels for causal language modeling (B, SeqLen).

        Returns:
            CausalLMOutputWithPast: Output from the LLM.
        """
        # 1. Get Multimodal Embeddings
        mm_embeds = self.encode_multimodal(inputs) # (B, M, D)
        
        # 2. Get Text Embeddings
        input_ids = inputs['input_ids']
        att_mask = inputs['attention_mask']
        
        # Get input embeddings from LLM
        # Handle different HF models (model.embed_tokens vs model.model.embed_tokens)
        if hasattr(self.llm, 'model'):
             if hasattr(self.llm.model, 'embed_tokens'):
                 word_embeds = self.llm.model.embed_tokens(input_ids)
             else:
                 word_embeds = self.llm.get_input_embeddings()(input_ids)
        else:
             word_embeds = self.llm.get_input_embeddings()(input_ids)
             
        # 3. Concatenate: [MM_Embeds, Text_Embeds]
        # We assume the input_ids already contain the instruction + response
        # We prepend MM embeds.
        
        inputs_embeds = torch.cat([mm_embeds, word_embeds], dim=1)
        
        # 4. Adjust Attention Mask
        # MM embeds also need attention (1s)
        B, M, _ = mm_embeds.shape
        mm_mask = torch.ones(B, M, device=att_mask.device, dtype=att_mask.dtype)
        attention_mask = torch.cat([mm_mask, att_mask], dim=1)
        
        # 5. Adjust Labels
        # Labels should correspond to input_ids.
        # We need to prepend ignore_index (-100) for the MM tokens
        if labels is not None:
            mm_labels = torch.full((B, M), -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([mm_labels, labels], dim=1)
        
        # 6. LLM Forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
