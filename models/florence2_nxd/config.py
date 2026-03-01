"""
Configuration module for Florence-2 NxD Inference integration.

This module defines the configuration dataclass and constants for the Florence-2
model using neuronx-distributed-inference APIs.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import torch


# Vision encoder constants
VISION_HIDDEN_SIZE = 1024
LANGUAGE_HIDDEN_SIZE = 768
NUM_VISION_STAGES = 4

# Vision stage input/output shapes
# Format: (batch_size, sequence_length_or_channels, hidden_size_or_height, width)
VISION_STAGE_SHAPES = [
    (1, 3, 768, 768),      # Stage 0 input: RGB image
    (1, 36864, 128),       # Stage 1 input: flattened patches
    (1, 9216, 256),        # Stage 2 input
    (1, 2304, 512),        # Stage 3 input
]

VISION_OUTPUT_SHAPE = (1, 576, 1024)  # Final vision features

# Language model constants
VOCAB_SIZE = 51289
MAX_ENCODER_LENGTH = 600
DECODER_BUCKETS = [1, 4, 8, 16, 32, 64]
MAX_DECODER_LENGTH = 64

# Token IDs
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 1

# Supported tasks
SUPPORTED_TASKS = [
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<OD>",
    "<OCR>",
    "<REGION_CAPTION>",
]

# Default model name
DEFAULT_MODEL_NAME = "microsoft/Florence-2-base"


@dataclass
class Florence2NxDConfig:
    """
    Configuration for Florence-2 NxD Inference model.
    
    This dataclass contains all configuration parameters needed for compiling
    and running Florence-2 with neuronx-distributed-inference on AWS Inferentia2.
    
    Attributes:
        vision_hidden_size: Hidden dimension size for vision encoder output
        language_hidden_size: Hidden dimension size for language model
        vocab_size: Size of the vocabulary
        num_vision_stages: Number of stages in DaViT vision encoder
        vision_stage_shapes: Input shapes for each vision encoder stage
        vision_output_shape: Output shape from vision encoder
        max_encoder_length: Maximum sequence length for encoder
        decoder_buckets: List of bucket sizes for decoder compilation
        max_decoder_length: Maximum decoder sequence length
        dtype: Tensor data type (BF16 for Inferentia2)
        tp_degree: Tensor parallelism degree (1, 2, 4, or 8)
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        model_dir: Directory containing compiled models
        model_name: HuggingFace model identifier
    """
    
    # Model architecture
    vision_hidden_size: int = VISION_HIDDEN_SIZE
    language_hidden_size: int = LANGUAGE_HIDDEN_SIZE
    vocab_size: int = VOCAB_SIZE
    
    # Vision encoder
    num_vision_stages: int = NUM_VISION_STAGES
    vision_stage_shapes: List[Tuple[int, ...]] = field(
        default_factory=lambda: VISION_STAGE_SHAPES.copy()
    )
    vision_output_shape: Tuple[int, int, int] = VISION_OUTPUT_SHAPE
    
    # Language model
    max_encoder_length: int = MAX_ENCODER_LENGTH
    decoder_buckets: List[int] = field(
        default_factory=lambda: DECODER_BUCKETS.copy()
    )
    max_decoder_length: int = MAX_DECODER_LENGTH
    
    # Precision and hardware
    dtype: torch.dtype = torch.bfloat16
    tp_degree: int = 1
    neuron_core_placement: List[int] = field(default_factory=list)
    
    # Generation
    bos_token_id: int = BOS_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID
    pad_token_id: int = PAD_TOKEN_ID
    
    # Paths
    model_dir: str = "./compiled_nxd"
    model_name: str = DEFAULT_MODEL_NAME
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            AssertionError: If any configuration parameter is invalid
        """
        assert self.tp_degree in [1, 2, 4, 8], \
            f"tp_degree must be 1, 2, 4, or 8, got {self.tp_degree}"
        
        assert self.max_decoder_length <= max(self.decoder_buckets), \
            f"max_decoder_length ({self.max_decoder_length}) must not exceed " \
            f"largest bucket ({max(self.decoder_buckets)})"
        
        assert self.dtype == torch.bfloat16, \
            f"Only BF16 is supported, got {self.dtype}"
        
        assert self.num_vision_stages == len(self.vision_stage_shapes), \
            f"num_vision_stages ({self.num_vision_stages}) must match " \
            f"length of vision_stage_shapes ({len(self.vision_stage_shapes)})"
        
        assert len(self.decoder_buckets) > 0, \
            "decoder_buckets must contain at least one bucket size"
        
        assert all(b > 0 for b in self.decoder_buckets), \
            "All decoder bucket sizes must be positive"
        
        assert self.max_encoder_length > 0, \
            "max_encoder_length must be positive"
        
        assert self.vocab_size > 0, \
            "vocab_size must be positive"
        
        # Validate NeuronCore placement if specified
        if self.neuron_core_placement:
            assert len(self.neuron_core_placement) == self.tp_degree, \
                f"neuron_core_placement length ({len(self.neuron_core_placement)}) " \
                f"must match tp_degree ({self.tp_degree})"
            
            assert all(c >= 0 for c in self.neuron_core_placement), \
                "All NeuronCore IDs must be non-negative"
            
            assert len(set(self.neuron_core_placement)) == len(self.neuron_core_placement), \
                "NeuronCore IDs must be unique (no duplicates allowed)"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Auto-generate neuron_core_placement if not specified
        if not self.neuron_core_placement and self.tp_degree > 1:
            self.neuron_core_placement = list(range(self.tp_degree))
        
        self.validate()
    
    def get_neuron_core_placement(self) -> List[int]:
        """
        Get NeuronCore placement configuration.
        
        Returns a list of NeuronCore IDs to use for tensor parallelism.
        If not explicitly configured, returns sequential core IDs [0, 1, ..., tp_degree-1].
        
        Returns:
            List of NeuronCore IDs
        
        Requirements:
            - 9.5: Provide configuration options for tensor parallel placement
        """
        if self.neuron_core_placement:
            return self.neuron_core_placement
        else:
            # Default: use sequential cores starting from 0
            return list(range(self.tp_degree))
