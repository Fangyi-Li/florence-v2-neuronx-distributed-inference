"""
Backward Compatibility Layer for Florence-2 NxD Inference.

This module provides a compatibility wrapper that maintains the same API
as the original Florence2NeuronBF16 implementation, allowing existing code
to work with the new NxD Inference implementation without modifications.

The compatibility layer:
- Accepts the same input formats (image path, PIL Image)
- Returns the same output formats (decoded text strings)
- Maintains the same method signatures
- Provides the same initialization parameters

Requirements:
    - 15.1: Implement same API as Florence2NeuronBF16
    - 15.1: Accept same input formats (image path, PIL Image)
    - 15.1: Return same output formats

Example:
    >>> # Old API (still works)
    >>> model = Florence2NeuronBF16("./compiled_bf16", core_id="0")
    >>> result = model("image.jpg", "<CAPTION>")
    
    >>> # New API with compatibility wrapper
    >>> model = Florence2NeuronBF16Compat("./compiled_nxd", core_id="0")
    >>> result = model("image.jpg", "<CAPTION>")  # Same interface!
"""

import os
import torch
from typing import Union, Optional
from pathlib import Path
import PIL.Image

from .model import Florence2NxDModel
from .config import Florence2NxDConfig
from .logging_config import get_logger
from .errors import InvalidTaskError, ImageLoadError


logger = get_logger(__name__)


class Florence2NeuronBF16Compat:
    """
    Backward compatibility wrapper for Florence2NeuronBF16.
    
    This class provides the same API as the original Florence2NeuronBF16
    implementation but uses the new NxD Inference backend. It allows
    existing code to work without modifications.
    
    The wrapper:
    - Maintains the same __call__ interface
    - Accepts image paths, PIL Images, or tensors
    - Returns decoded text strings
    - Supports the same initialization parameters
    
    Attributes:
        model: Underlying Florence2NxDModel instance
        processor: HuggingFace processor for image preprocessing
        tokenizer: HuggingFace tokenizer for text processing
    
    Requirements:
        - 15.1: Implement same API as Florence2NeuronBF16
        - 15.1: Accept same input formats (image path, PIL Image)
        - 15.1: Return same output formats
    
    Example:
        >>> model = Florence2NeuronBF16Compat("./compiled_nxd", core_id="0")
        >>> result = model("image.jpg", "<CAPTION>")
        >>> print(result)
        'A cat sitting on a couch'
    """
    
    def __init__(self, model_dir: str = "./compiled_nxd", core_id: str = "0"):
        """
        Initialize Florence-2 compatibility wrapper.
        
        This constructor maintains the same signature as Florence2NeuronBF16
        for backward compatibility. It sets up the NeuronCore environment
        and loads the NxD Inference model.
        
        Args:
            model_dir: Directory containing compiled models
            core_id: NeuronCore ID to use (e.g., "0", "1", or "0,1")
        
        Requirements:
            - 15.1: Accept same initialization parameters as Florence2NeuronBF16
        """
        # Set up NeuronCore environment (same as old implementation)
        os.environ["NEURON_RT_VISIBLE_CORES"] = core_id
        
        # Determine number of cores from core_id
        if ',' in core_id:
            num_cores = len(core_id.split(','))
        elif '-' in core_id:
            start, end = core_id.split('-')
            num_cores = int(end) - int(start) + 1
        else:
            num_cores = 1
        
        os.environ["NEURON_RT_NUM_CORES"] = str(num_cores)
        
        logger.info(f"Loading Florence-2 Neuron BF16 (NC{core_id})...")
        
        # Load the NxD Inference model
        # Use tp_degree=1 for single-core compatibility
        self.model = Florence2NxDModel(model_dir, tp_degree=1)
        
        # Store references to processor and tokenizer for convenience
        self.processor = self.model.processor
        self.tokenizer = self.model.tokenizer
        
        logger.info("Ready!")
    
    def __call__(
        self,
        image: Union[str, PIL.Image.Image, torch.Tensor],
        task: str = "<CAPTION>",
        max_tokens: int = 100
    ) -> str:
        """
        Run inference on an image.
        
        This method maintains the same signature as Florence2NeuronBF16.__call__
        for backward compatibility. It accepts various image formats and returns
        the decoded text output.
        
        Args:
            image: PIL Image, path to image file, or tensor
            task: Florence-2 task prompt (e.g., "<CAPTION>", "<OD>", "<OCR>")
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text response (decoded string)
        
        Raises:
            InvalidTaskError: If task is not supported
            ImageLoadError: If image cannot be loaded
            FileNotFoundError: If image path does not exist
        
        Requirements:
            - 15.1: Accept same input formats (image path, PIL Image)
            - 15.1: Return same output formats (decoded text)
        
        Example:
            >>> model = Florence2NeuronBF16Compat("./compiled_nxd")
            >>> result = model("image.jpg", "<CAPTION>")
            >>> print(result)
            'A cat sitting on a couch'
        """
        # Load image if it's a path
        if isinstance(image, str):
            try:
                image = PIL.Image.open(image).convert("RGB")
            except Exception as e:
                raise ImageLoadError(image, str(e))
        
        # Validate inputs
        self.model.validate_inputs(image, task)
        
        # Preprocess image and tokenize text using processor
        inputs = self.processor(text=task, images=image, return_tensors="pt")
        
        # Convert to BF16 for consistency with old implementation
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        input_ids = inputs["input_ids"]
        
        # Run generation through NxD model
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=max_tokens
            )
        
        # Decode the generated tokens (same as old implementation)
        result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return result


# Alias for backward compatibility
# This allows code using the old class name to work without changes
Florence2NeuronBF16 = Florence2NeuronBF16Compat


def create_compatible_model(
    model_dir: str = "./compiled_nxd",
    core_id: str = "0"
) -> Florence2NeuronBF16Compat:
    """
    Factory function to create a backward-compatible Florence-2 model.
    
    This function provides a convenient way to create a compatibility
    wrapper with the same interface as the original implementation.
    
    Args:
        model_dir: Directory containing compiled models
        core_id: NeuronCore ID to use
    
    Returns:
        Florence2NeuronBF16Compat instance
    
    Requirements:
        - 15.1: Provide backward-compatible API
    
    Example:
        >>> model = create_compatible_model("./compiled_nxd", core_id="0")
        >>> result = model("image.jpg", "<CAPTION>")
    """
    return Florence2NeuronBF16Compat(model_dir, core_id)
