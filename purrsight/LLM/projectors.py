"""Multimodal Projectors for Phase 2 (LLM Adaptation).

Implements the bridge between Aligned Encoder features and LLM.

Key features:
- Linear-GELU-Linear architecture
- Soft Prompting (Maps 1 feature vector to N tokens)
- Zero-vector Fail-Safe (Maps missing modalities to <MISSING> token)
"""

import torch
import torch.nn as nn
from typing import Optional


class ModalityProjector(nn.Module):
    """Projector that transforms aligned features into LLM input embeddings.
    
    This projector implements a multi-layer MLP architecture with layer normalization,
    GELU activation, and dropout for robust feature transformation. It supports
    variable sequence lengths and is designed for Phase 2 training where aligned
    features from Phase 1 need to be projected into the LLM's input space.
    
    Architecture (for num_layers=2):
        Input (batch, seq_len, input_dim)
        -> Linear(input_dim, hidden_dim)
        -> LayerNorm(hidden_dim)
        -> GELU
        -> Dropout
        -> Linear(hidden_dim, output_dim)
        -> Output (batch, seq_len, output_dim)
    
    Attributes:
        input_dim: Dimension of aligned features from Phase 1
        output_dim: Dimension of LLM input embeddings
        hidden_dim: Hidden layer dimension for MLP
        num_layers: Number of projection layers (currently supports 2)
        dropout: Dropout probability for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize projector module.
        
        Args:
            input_dim: Dimension of aligned features (e.g., 512 from Phase 1)
            output_dim: Dimension of LLM input embeddings (e.g., 896 for Qwen2.5-0.5B)
            hidden_dim: Hidden layer dimension (default: 2048)
            num_layers: Number of projection layers (default: 2)
            dropout: Dropout probability (default: 0.1)
        
        Raises:
            ValueError: If num_layers is not 2 (currently only 2-layer architecture is supported)
        """
        super().__init__()
        
        if num_layers != 2:
            raise ValueError(f"Currently only num_layers=2 is supported, got {num_layers}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        
        # Build MLP: Linear → LayerNorm → GELU → Dropout → Linear
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights with Xavier uniform
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform initialization.
        
        Xavier uniform initialization helps maintain gradient flow through
        deep networks by keeping the variance of activations consistent
        across layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, aligned_features: torch.Tensor) -> torch.Tensor:
        """Project aligned features to LLM input space.
        
        This method transforms aligned features from Phase 1 into embeddings
        suitable for the LLM. It supports variable sequence lengths and
        validates input/output shapes.
        
        Args:
            aligned_features: Tensor of shape (batch, seq_len, input_dim)
                             or (batch, input_dim) for single-token inputs
        
        Returns:
            Projected embeddings of shape (batch, seq_len, output_dim)
            or (batch, output_dim) matching the input shape pattern
        
        Raises:
            ValueError: If input tensor has incorrect dimensions or shape
        """
        # Validate input dimensions
        if aligned_features.dim() not in [2, 3]:
            raise ValueError(
                f"Expected input tensor with 2 or 3 dimensions (batch, [seq_len,] input_dim), "
                f"got {aligned_features.dim()} dimensions with shape {aligned_features.shape}"
            )
        
        # Validate input feature dimension
        expected_dim = -1  # Last dimension should be input_dim
        if aligned_features.shape[expected_dim] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {aligned_features.shape[expected_dim]}"
            )
        
        # Store original shape for validation
        original_shape = aligned_features.shape
        
        # Apply projection
        output = self.layers(aligned_features)
        
        # Validate output shape
        expected_output_shape = original_shape[:-1] + (self.output_dim,)
        if output.shape != expected_output_shape:
            raise RuntimeError(
                f"Output shape mismatch. Expected {expected_output_shape}, "
                f"got {output.shape}"
            )
        
        return output


class MultimodalProjector(nn.Module):
    """Projector that maps aligned features to LLM embedding space.

    Architecture:
    Input (B, 512) -> Linear(512, hidden) -> GELU -> Linear(hidden, llm_dim * num_tokens) -> Reshape (B, num_tokens, llm_dim)

    Includes fail-safe for zero vectors (missing modalities).

    Attributes:
        input_dim: Dimension of input aligned features.
        llm_dim: Dimension of LLM embeddings.
        num_tokens: Number of soft prompt tokens per feature.
        mlp: The MLP network.
        missing_token: Learnable or fixed embedding for missing modalities.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        llm_dim: int = 1024,
        num_tokens: int = 4,
        hidden_dim: int = 2048,
        missing_token_learnable: bool = True
    ):
        """Initializes the MultimodalProjector.

        Args:
            input_dim: Dimension of input aligned features. Defaults to 512.
            llm_dim: Dimension of LLM embeddings (e.g., 1024 for MatFormer-OLMo-0.5B).
            num_tokens: Number of soft prompt tokens per feature. Defaults to 4.
            hidden_dim: Hidden dimension of the MLP. Defaults to 2048.
            missing_token_learnable: Whether the missing token embedding is learnable. Defaults to True.
        """
        super().__init__()
        self.input_dim = input_dim
        self.llm_dim = llm_dim
        self.num_tokens = num_tokens
        
        # MLP Adapter
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim * num_tokens)
        )
        
        # Learnable embedding for missing modalities (zero vectors)
        # If a modality is missing, we replace the projected output with this token
        # instead of feeding zeros to the LLM.
        # Initialize with small random values
        if missing_token_learnable:
            self.missing_token = nn.Parameter(torch.randn(1, num_tokens, llm_dim) * 0.02)
        else:
            self.register_buffer('missing_token', torch.zeros(1, num_tokens, llm_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects aligned features to LLM tokens.

        Args:
            x: Aligned features of shape (B, input_dim).

        Returns:
            Projected tokens of shape (B, num_tokens, llm_dim).
            If a feature vector is zero (missing modality), it is replaced by the missing_token.
        """
        B = x.shape[0]
        
        # Detect zero vectors (missing modalities)
        # Assuming x is (B, D)
        # We check if the sum of absolute values is 0 (or close to 0 due to float precision)
        # Since features are L2 normalized (norm=1), zero vectors have norm 0.
        is_zero = torch.sum(torch.abs(x), dim=1) < 1e-6  # (B,)
        
        # Project
        out = self.mlp(x)  # (B, num_tokens * llm_dim)
        out = out.view(B, self.num_tokens, self.llm_dim)  # (B, N, D)
        
        # Apply fail-safe for zero vectors
        if is_zero.any():
            # Expand missing token to batch size
            missing_embeds = self.missing_token.expand(B, -1, -1)
            
            # Create mask for broadcasting
            # is_zero is (B,), we need (B, N, D)
            mask = is_zero.view(B, 1, 1).expand(-1, self.num_tokens, self.llm_dim)
            
            # Replace where is_zero is True
            out = torch.where(mask, missing_embeds, out)
            
        return out
