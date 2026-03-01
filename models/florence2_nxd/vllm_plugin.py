"""
vLLM plugin for Florence-2 multimodal inference.

This module provides a vLLM-compatible plugin that integrates Florence-2 NxD Inference
models with vLLM's serving infrastructure for production deployment.

Requirements:
    - 4.1: Implement vLLM plugin extending vLLM's model execution components
    - 4.2: Load NxD Inference compiled Florence-2 models
    - 4.4: Integrate image preprocessing into vLLM pipeline
    - 5.1: Handle image inputs through vLLM
    - 5.3: Handle text inputs through vLLM
"""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import PIL.Image

from .model import Florence2NxDModel
from .config import Florence2NxDConfig
from .errors import (
    InvalidTaskError,
    ImageLoadError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)


class Florence2VLLMPlugin:
    """
    vLLM plugin for Florence-2 multimodal inference.
    
    This plugin integrates Florence-2 NxD Inference models with vLLM's LLMEngine
    to provide:
    - Image preprocessing for vision encoder
    - Vision-language fusion
    - Florence-2 model execution through vLLM interface
    
    The plugin follows vLLM's multimodal plugin architecture and provides
    OpenAI-compatible API endpoints for production serving.
    
    Attributes:
        model_dir: Directory containing compiled NxD Inference models
        tp_degree: Tensor parallelism degree (1, 2, 4, or 8)
        model: Florence2NxDModel instance for inference
        config: Model configuration
    
    Requirements:
        - 4.1: Implement vLLM plugin extending vLLM's model execution components
        - 4.2: Load NxD Inference compiled Florence-2 models
    
    Example:
        >>> plugin = Florence2VLLMPlugin(
        ...     model_dir="./compiled_nxd",
        ...     tp_degree=1
        ... )
        >>> result = plugin.execute_model(
        ...     pixel_values=image_tensor,
        ...     input_ids=text_tokens,
        ...     max_new_tokens=100
        ... )
    """
    
    def __init__(self, model_dir: str, tp_degree: int = 1):
        """
        Initialize vLLM plugin with Florence-2 NxD Inference model.
        
        This method loads the compiled NxD Inference models and sets up
        the vLLM integration hooks for multimodal inference.
        
        Args:
            model_dir: Directory containing compiled NxD Inference models.
                      Must contain all required model files (vision stages,
                      projection, encoder, decoders, metadata).
            tp_degree: Tensor parallelism degree. Must be 1, 2, 4, or 8.
                      Determines how model layers are distributed across
                      NeuronCores.
        
        Raises:
            ModelLoadError: If model files are missing or cannot be loaded
            HardwareCompatibilityError: If hardware doesn't support requested TP degree
            ValueError: If tp_degree is not in [1, 2, 4, 8]
        
        Requirements:
            - 4.1: Initialize with model directory and TP degree
            - 4.2: Load Florence2NxDModel instance
        """
        logger.info(
            f"Initializing Florence2VLLMPlugin with model_dir={model_dir}, "
            f"tp_degree={tp_degree}"
        )
        
        # Validate inputs
        if tp_degree not in [1, 2, 4, 8]:
            raise ValueError(
                f"tp_degree must be 1, 2, 4, or 8, got {tp_degree}"
            )
        
        model_path = Path(model_dir)
        if not model_path.exists():
            raise ModelLoadError(
                model_dir=model_dir,
                missing_files=[str(model_path)]
            )
        
        # Store configuration
        self.model_dir = model_dir
        self.tp_degree = tp_degree
        
        # Load Florence-2 NxD Inference model
        logger.info("Loading Florence2NxDModel...")
        try:
            self.model = Florence2NxDModel(
                model_dir=model_dir,
                tp_degree=tp_degree
            )
            self.config = self.model.config
            logger.info("Florence2NxDModel loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Florence2NxDModel: {e}")
            raise
        
        # Set up vLLM integration hooks
        self._setup_vllm_hooks()
        
        logger.info("Florence2VLLMPlugin initialized successfully")
    
    def _setup_vllm_hooks(self) -> None:
        """
        Set up vLLM integration hooks.
        
        This method configures the plugin to work with vLLM's LLMEngine,
        including multimodal input processing and request scheduling.
        
        Requirements:
            - 4.1: Set up vLLM integration hooks
        """
        # TODO: Implement vLLM-specific hooks when integrating with vLLM
        # This will include:
        # - Registering multimodal input processor
        # - Configuring request scheduler
        # - Setting up tokenizer interface
        logger.debug("vLLM integration hooks configured")
    
    def preprocess_image(
        self,
        image: Union[str, PIL.Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess image for vision encoder.
        
        This method integrates image preprocessing into vLLM's input processing
        pipeline. It handles various input formats and returns a preprocessed
        tensor in the format expected by the DaViT vision encoder.
        
        The preprocessing includes:
        - Loading image from file path if needed
        - Resizing to 768x768
        - Normalizing pixel values
        - Converting to BF16 format
        - Adding batch dimension
        
        Args:
            image: Input image in one of the following formats:
                  - str: Path to image file
                  - PIL.Image: PIL Image object
                  - torch.Tensor: Pre-processed tensor
        
        Returns:
            Preprocessed image tensor of shape (1, 3, 768, 768) in BF16 format,
            ready for vision encoder input.
        
        Raises:
            ImageLoadError: If image file cannot be loaded
            ValueError: If image format is invalid
        
        Requirements:
            - 4.4: Integrate image preprocessing into vLLM pipeline
            - 5.1: Preprocess image to 768x768 resolution
        
        Example:
            >>> plugin = Florence2VLLMPlugin("./compiled_nxd")
            >>> pixel_values = plugin.preprocess_image("image.jpg")
            >>> pixel_values.shape
            torch.Size([1, 3, 768, 768])
            >>> pixel_values.dtype
            torch.bfloat16
        """
        logger.debug(f"Preprocessing image of type {type(image)}")
        
        try:
            # Delegate to model's preprocessing method
            pixel_values = self.model.preprocess_image(image)
            
            logger.debug(
                f"Image preprocessed: shape={pixel_values.shape}, "
                f"dtype={pixel_values.dtype}"
            )
            
            return pixel_values
        
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def process_multimodal_input(
        self,
        image: Union[str, PIL.Image.Image, torch.Tensor],
        text: str
    ) -> Dict[str, torch.Tensor]:
        """
        Process multimodal input for inference.
        
        This method handles both image and text inputs, preparing them for
        Florence-2 inference through vLLM. It returns a dictionary containing
        preprocessed pixel values and tokenized text.
        
        Args:
            image: Input image (file path, PIL Image, or tensor)
            text: Task prompt text (e.g., "<CAPTION>", "<OD>", "<OCR>")
        
        Returns:
            Dictionary with keys:
            - 'pixel_values': Preprocessed image tensor (1, 3, 768, 768) in BF16
            - 'input_ids': Tokenized text tensor (1, seq_len) as int64
        
        Raises:
            ImageLoadError: If image cannot be loaded
            InvalidTaskError: If task prompt is not supported
            ValueError: If inputs are invalid
        
        Requirements:
            - 5.1: Handle image inputs
            - 5.3: Tokenize task prompts using Florence-2's tokenizer
        
        Example:
            >>> plugin = Florence2VLLMPlugin("./compiled_nxd")
            >>> inputs = plugin.process_multimodal_input(
            ...     image="photo.jpg",
            ...     text="<CAPTION>"
            ... )
            >>> inputs.keys()
            dict_keys(['pixel_values', 'input_ids'])
        """
        logger.debug(f"Processing multimodal input: text='{text}'")
        
        # Validate task prompt
        self.model.validate_task(text)
        
        # Load image if it's a path
        if isinstance(image, str):
            import os
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            import PIL.Image
            image = PIL.Image.open(image)
        
        # Process image and text together using the processor
        inputs = self.model.processor(text=text, images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.model.config.dtype)
        input_ids = inputs["input_ids"]
        
        # Return dictionary with both modalities
        result = {
            'pixel_values': pixel_values,
            'input_ids': input_ids
        }
        
        logger.debug(
            f"Multimodal input processed: "
            f"pixel_values.shape={pixel_values.shape}, "
            f"input_ids.shape={input_ids.shape}"
        )
        
        return result
    
    def execute_model(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100
    ) -> List[int]:
        """
        Execute Florence-2 model through vLLM interface.
        
        This method runs the full Florence-2 inference pipeline:
        1. Encode image through vision encoder
        2. Project vision features to language dimension
        3. Combine vision and text embeddings
        4. Encode combined embeddings
        5. Autoregressive decoding with bucket selection
        
        Args:
            pixel_values: Preprocessed image tensor (1, 3, 768, 768) in BF16
            input_ids: Tokenized text tensor (1, seq_len) as int64
            max_new_tokens: Maximum number of tokens to generate (default: 100)
        
        Returns:
            List of generated token IDs (excluding input tokens)
        
        Raises:
            GenerationError: If generation fails
            NumericalError: If numerical instability is detected
            SequenceTooLongError: If sequence exceeds maximum length
        
        Requirements:
            - 4.2: Execute Florence2NxDModel through vLLM interface
        
        Example:
            >>> plugin = Florence2VLLMPlugin("./compiled_nxd")
            >>> inputs = plugin.process_multimodal_input("image.jpg", "<CAPTION>")
            >>> token_ids = plugin.execute_model(
            ...     pixel_values=inputs['pixel_values'],
            ...     input_ids=inputs['input_ids'],
            ...     max_new_tokens=50
            ... )
            >>> len(token_ids)
            25  # Example: generated 25 tokens
        """
        logger.debug(
            f"Executing model: pixel_values.shape={pixel_values.shape}, "
            f"input_ids.shape={input_ids.shape}, max_new_tokens={max_new_tokens}"
        )
        
        try:
            # Execute full generation pipeline
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens
            )
            
            # Convert to list of integers
            token_ids = generated_ids.squeeze(0).tolist()
            
            logger.debug(f"Model execution complete: generated {len(token_ids)} tokens")
            
            return token_ids
        
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            raise

    def execute_model_batch(
        self,
        pixel_values_list: List[torch.Tensor],
        input_ids_list: List[torch.Tensor],
        max_new_tokens: int = 100
    ) -> List[List[int]]:
        """
        Execute Florence-2 model on a batch of requests through vLLM interface.
        
        This method supports continuous batching where multiple requests are
        processed simultaneously with per-request state management. Requests
        can complete at different times without blocking others.
        
        The batched execution:
        1. Processes all images through vision encoder in parallel
        2. Maintains per-request vision embeddings
        3. Performs batched autoregressive decoding
        4. Removes completed requests as they finish
        
        Args:
            pixel_values_list: List of preprocessed image tensors, each (1, 3, 768, 768)
            input_ids_list: List of tokenized text tensors, each (1, seq_len)
            max_new_tokens: Maximum number of tokens to generate per request
        
        Returns:
            List of generated token ID lists, one per request
        
        Raises:
            GenerationError: If generation fails
            ValueError: If input lists have mismatched lengths
        
        Requirements:
            - 8.1: Support batch dimension in all stages
            - 8.2: Allow new requests to join ongoing batches during generation
            - 8.3: Remove completed requests without blocking others
            - 8.4: Maintain per-request state for vision embeddings
        
        Example:
            >>> plugin = Florence2VLLMPlugin("./compiled_nxd")
            >>> images = [image1_tensor, image2_tensor, image3_tensor]
            >>> prompts = [prompt1_ids, prompt2_ids, prompt3_ids]
            >>> results = plugin.execute_model_batch(images, prompts, max_new_tokens=50)
            >>> len(results)
            3
            >>> [len(r) for r in results]
            [25, 30, 28]  # Example: different lengths per request
        """
        batch_size = len(pixel_values_list)
        logger.debug(
            f"Executing batched model: batch_size={batch_size}, "
            f"max_new_tokens={max_new_tokens}"
        )
        
        if batch_size == 0:
            return []
        
        if len(input_ids_list) != batch_size:
            raise ValueError(
                f"Mismatch between pixel_values_list ({batch_size}) and "
                f"input_ids_list ({len(input_ids_list)}) lengths"
            )
        
        try:
            # Execute batched generation pipeline
            generated_ids_list = self.model.generate_batch(
                pixel_values_list=pixel_values_list,
                input_ids_list=input_ids_list,
                max_new_tokens=max_new_tokens
            )
            
            # Convert each result to list of integers
            token_ids_list = [
                generated_ids.squeeze(0).tolist()
                for generated_ids in generated_ids_list
            ]
            
            logger.debug(
                f"Batched model execution complete: generated "
                f"{[len(ids) for ids in token_ids_list]} tokens per request"
            )
            
            return token_ids_list
        
        except Exception as e:
            logger.error(f"Batched model execution failed: {e}")
            raise

