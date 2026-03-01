"""
NxD Inference wrapper modules for Florence-2.

This module provides wrapper classes that adapt Florence-2 components to work
with neuronx-distributed-inference (NxD Inference) APIs. Each wrapper encapsulates
a specific component of the Florence-2 architecture and provides a clean interface
for compilation and inference on AWS Inferentia2.

The wrappers follow NxD Inference model builder patterns and handle:
- Stage-wise compilation for DaViT vision encoder
- Projection layer with position embeddings
- BART encoder for language encoding
- BART decoder with bucketing strategy
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .logging_config import get_logger


logger = get_logger(__name__)


class NxDVisionStage(nn.Module):
    """
    NxD Inference wrapper for a single DaViT vision encoder stage.
    
    This wrapper combines a convolutional downsampling layer and a transformer
    block into a single module that can be compiled with NxD Inference. The
    DaViT architecture has 4 stages, each processing the image at different
    spatial resolutions.
    
    Stage progression:
    - Stage 0: 768x768 RGB image → 192x192 feature map (128 channels)
    - Stage 1: 192x192 → 96x96 (256 channels)
    - Stage 2: 96x96 → 48x48 (512 channels)
    - Stage 3: 48x48 → 24x24 (1024 channels)
    
    Attributes:
        conv_layer: Convolutional downsampling layer
        transformer_block: Transformer block for feature processing
        input_size: Expected input spatial dimensions (H, W)
        output_size: Expected output spatial dimensions (H, W)
    
    Requirements:
        - 2.1: Preserve 4-stage DaViT architecture with stage-wise compilation
        - 2.3: Execute all 4 vision stages sequentially on NeuronCores
    """
    
    def __init__(
        self,
        conv_layer: nn.Module,
        transformer_block: nn.Module,
        input_size: Tuple[int, int],
        output_size: Tuple[int, int],
    ):
        """
        Initialize vision stage wrapper.
        
        Args:
            conv_layer: Convolutional downsampling layer from DaViT
            transformer_block: Transformer block from DaViT
            input_size: Input spatial dimensions (height, width)
            output_size: Output spatial dimensions (height, width)
        
        Example:
            >>> conv = model.vision_tower.encoder.stages[0].downsample
            >>> block = model.vision_tower.encoder.stages[0].blocks
            >>> stage = NxDVisionStage(conv, block, (768, 768), (192, 192))
        """
        super().__init__()
        self.conv_layer = conv_layer
        self.transformer_block = transformer_block
        self.input_size = input_size
        self.output_size = output_size
        
        logger.debug(
            f"Initialized NxDVisionStage: {input_size} → {output_size}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision stage.
        
        Applies convolutional downsampling followed by transformer processing.
        The input/output shapes depend on the stage number:
        
        - Stage 0: (B, 3, 768, 768) → (B, 36864, 128)
        - Stage 1: (B, 36864, 128) → (B, 9216, 256)
        - Stage 2: (B, 9216, 256) → (B, 2304, 512)
        - Stage 3: (B, 2304, 512) → (B, 576, 1024)
        
        Args:
            x: Input tensor with shape depending on stage
        
        Returns:
            Processed features with shape depending on stage
        
        Requirements:
            - 2.1: Preserve 4-stage DaViT architecture
            - 2.3: Execute stages sequentially on NeuronCores
        """
        # Apply convolutional downsampling with size parameter
        x, _ = self.conv_layer(x, self.input_size)
        
        # Apply transformer block with size parameter
        x, _ = self.transformer_block(x, self.output_size)
        
        return x


class NxDProjection(nn.Module):
    """
    NxD Inference wrapper for vision-to-language projection layer.
    
    This wrapper combines the projection layer, layer normalization, and position
    embeddings into a single module. It transforms vision encoder outputs from
    vision dimension (1024) to language dimension (768) and adds positional
    information.
    
    The projection process:
    1. Add position embeddings to vision features (24x24 grid)
    2. Add mean-pooled feature as first token (global image representation)
    3. Apply linear projection: 1024 → 768 dimensions
    4. Apply layer normalization
    
    Attributes:
        projection_layer: Linear projection matrix (1024 → 768)
        layer_norm: Layer normalization
        position_embed: Position embeddings for 24x24 feature grid
    
    Requirements:
        - 2.4: Compile projection layer to Neuron
        - 2.5: Perform position embedding addition on Neuron
    """
    
    def __init__(
        self,
        projection_layer: nn.Module,
        layer_norm: nn.Module,
        position_embed: torch.Tensor,
    ):
        """
        Initialize projection wrapper.
        
        Args:
            projection_layer: Linear projection matrix from Florence-2
            layer_norm: Layer normalization from Florence-2
            position_embed: Precomputed position embeddings (1, 576, 1024)
        
        Example:
            >>> projection = model.image_projection
            >>> norm = model.image_proj_norm
            >>> pos_embed = precomputed_position_embeddings
            >>> proj_wrapper = NxDProjection(projection, norm, pos_embed)
        """
        super().__init__()
        self.projection_layer = projection_layer
        self.layer_norm = layer_norm
        
        # Register position embeddings as buffer (not a parameter)
        self.register_buffer("position_embed", position_embed)
        
        logger.debug(
            f"Initialized NxDProjection: "
            f"projection shape {projection_layer.shape}, "
            f"position_embed shape {position_embed.shape}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection layer.
        
        Transforms vision features to language dimension with position information:
        1. Add position embeddings: (B, 576, 1024) + (1, 576, 1024)
        2. Compute mean pooled feature: (B, 1, 1024)
        3. Concatenate: (B, 577, 1024)
        4. Project: (B, 577, 1024) @ (1024, 768) → (B, 577, 768)
        5. Normalize: (B, 577, 768)
        
        Args:
            x: Vision features from final DaViT stage (B, 576, 1024)
        
        Returns:
            Projected features in language dimension (B, 577, 768)
        
        Requirements:
            - 2.4: Compile projection to avoid CPU-GPU transfer
            - 2.5: Add position embeddings on Neuron
        """
        # Add position embeddings to vision features
        x = x + self.position_embed
        
        # Add mean-pooled feature as first token (global representation)
        mean_feature = x.mean(dim=1, keepdim=True)  # (B, 1, 1024)
        x = torch.cat([mean_feature, x], dim=1)  # (B, 577, 1024)
        
        # Project to language dimension
        x = x @ self.projection_layer  # (B, 577, 768)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x


class NxDEncoder(nn.Module):
    """
    NxD Inference wrapper for BART encoder.
    
    This wrapper encapsulates the BART encoder for processing combined vision
    and text embeddings. The encoder uses self-attention to create contextualized
    representations that will be used by the decoder for generation.
    
    The encoder processes:
    - Vision embeddings: 577 tokens (1 global + 576 spatial)
    - Text embeddings: Variable length (task prompt)
    - Combined: Up to 600 tokens (padded to fixed length)
    
    Attributes:
        encoder: BART encoder module from Florence-2
    
    Requirements:
        - 3.4: Compile encoder with maximum sequence length of 600 tokens
    """
    
    def __init__(self, encoder: nn.Module):
        """
        Initialize encoder wrapper.
        
        Args:
            encoder: BART encoder module from Florence-2 language model
        
        Example:
            >>> encoder = model.language_model.model.encoder
            >>> encoder_wrapper = NxDEncoder(encoder)
        """
        super().__init__()
        self.encoder = encoder
        
        logger.debug("Initialized NxDEncoder")
    
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Processes combined vision and text embeddings through BART encoder
        to create contextualized representations. The input is padded to
        a fixed length (600 tokens) for static shape compilation.
        
        Args:
            inputs_embeds: Combined embeddings (B, max_seq_len, 768)
                          where max_seq_len = 600
        
        Returns:
            Encoder hidden states (B, max_seq_len, 768)
        
        Requirements:
            - 3.4: Process sequences up to 600 tokens
        """
        # Run encoder with embeddings input
        # Note: We use inputs_embeds instead of input_ids because we have
        # already combined vision and text embeddings
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        
        return encoder_outputs.last_hidden_state


class NxDDecoder(nn.Module):
    """
    NxD Inference wrapper for BART decoder with bucketing.
    
    This wrapper combines the BART decoder, token embedding layer, and language
    modeling head into a single module. It implements a bucketing strategy where
    multiple decoder models are compiled for different sequence lengths (buckets)
    to handle variable-length generation efficiently.
    
    Bucket strategy:
    - Buckets: [1, 4, 8, 16, 32, 64] tokens
    - At inference: Select smallest bucket that fits current sequence
    - Pad input to bucket size for static shape
    
    The decoder performs autoregressive generation:
    1. Embed input token IDs
    2. Apply decoder with cross-attention to encoder outputs
    3. Apply language modeling head to get logits
    4. Select next token from logits
    
    Attributes:
        decoder: BART decoder module from Florence-2
        embed_tokens: Token embedding layer
        lm_head: Language modeling head (projects to vocabulary)
    
    Requirements:
        - 3.1: Compile decoder models for bucket sizes [1, 4, 8, 16, 32, 64]
    """
    
    def __init__(
        self,
        decoder: nn.Module,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
    ):
        """
        Initialize decoder wrapper.
        
        Args:
            decoder: BART decoder module from Florence-2
            embed_tokens: Token embedding layer
            lm_head: Language modeling head
        
        Example:
            >>> decoder = model.language_model.model.decoder
            >>> embed = model.language_model.model.shared
            >>> lm_head = model.language_model.lm_head
            >>> decoder_wrapper = NxDDecoder(decoder, embed, lm_head)
        """
        super().__init__()
        self.decoder = decoder
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        
        logger.debug("Initialized NxDDecoder")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Performs a single decoder step for autoregressive generation:
        1. Embed input token IDs: (B, seq_len) → (B, seq_len, 768)
        2. Apply decoder with cross-attention to encoder outputs
        3. Project to vocabulary: (B, seq_len, 768) → (B, seq_len, vocab_size)
        
        The input is padded to the bucket size for static shape compilation.
        
        Args:
            input_ids: Token IDs (B, bucket_size)
            encoder_hidden_states: Encoder outputs (B, enc_len, 768)
        
        Returns:
            Logits over vocabulary (B, bucket_size, vocab_size)
        
        Requirements:
            - 3.1: Support bucketed compilation for efficient generation
        """
        # Embed input tokens
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Run decoder with cross-attention to encoder outputs
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
        )
        
        # Project to vocabulary
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        
        return logits
