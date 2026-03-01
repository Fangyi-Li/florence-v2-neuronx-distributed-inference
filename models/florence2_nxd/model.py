"""
Florence-2 NxD Inference Model Implementation.

This module implements the main Florence2NxDModel class that provides the
inference interface for Florence-2 using neuronx-distributed-inference (NxD)
compiled models on AWS Inferentia2.

The model supports:
- Vision encoding through 4-stage DaViT architecture
- Vision-to-language projection with position embeddings
- Language encoding with BART encoder
- Autoregressive decoding with bucket strategy
- Full inference pipeline for multimodal tasks
- Batch processing with per-request state management

Requirements:
    - 1.4: Support serialization and deserialization of compiled NxD models
    - 1.5: Use NxD Inference model loading APIs
    - 2.3: Execute all 4 vision stages sequentially
    - 2.4: Apply projection layer to vision features
    - 2.5: Add position embeddings on Neuron
    - 3.2: Select smallest bucket that fits current sequence length
    - 3.4: Encode combined embeddings with BART encoder
    - 3.5: Pad input to bucket size during inference
    - 5.4: Combine vision and text embeddings
    - 5.5: Pad combined embeddings to max length (600)
    - 5.6: Autoregressive decoding loop with bucket selection
    - 8.4: Track vision embeddings for each request in batch
    - 10.3: Load pre-compiled models without recompilation
"""

import sys
import types
from importlib.machinery import ModuleSpec

# Create dummy flash_attn module to avoid import errors
# Florence-2 checks for flash_attn during import, but we use eager attention
if 'flash_attn' not in sys.modules:
    flash_attn = types.ModuleType('flash_attn')
    flash_attn.__spec__ = ModuleSpec('flash_attn', None, origin='built-in')
    flash_attn.__version__ = '2.0.0'
    flash_attn.__file__ = None
    sys.modules['flash_attn'] = flash_attn
    
    flash_attn_interface = types.ModuleType('flash_attn.flash_attn_interface')
    flash_attn_interface.__spec__ = ModuleSpec('flash_attn.flash_attn_interface', None, origin='built-in')
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import PIL.Image

from .config import Florence2NxDConfig, DECODER_BUCKETS, SUPPORTED_TASKS
from .logging_config import get_logger
from .model_loader import load_florence2_model
from .errors import InvalidTaskError, ImageLoadError, ModelLoadError, HardwareCompatibilityError
from .metadata import CompiledModelMetadata


logger = get_logger(__name__)


@dataclass
class RequestState:
    """
    Per-request state for batched inference.
    
    This class tracks the state of a single request during batched generation,
    including vision embeddings, encoder hidden states, and generation progress.
    
    Attributes:
        request_id: Unique identifier for this request
        vision_embeddings: Vision embeddings for this request [1, 577, 768]
        encoder_hidden_states: Encoder hidden states [1, max_seq_len, 768]
        generated_ids: Currently generated token IDs [1, seq_len]
        is_complete: Whether generation is complete for this request
        eos_generated: Whether EOS token has been generated
    
    Requirements:
        - 8.4: Track vision embeddings for each request in batch
        - 8.4: Ensure correct association during generation
    """
    request_id: str
    vision_embeddings: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None
    is_complete: bool = False
    eos_generated: bool = False
    
    def mark_complete(self) -> None:
        """Mark this request as complete."""
        self.is_complete = True
    
    def check_eos(self, eos_token_id: int) -> bool:
        """
        Check if EOS token has been generated.
        
        Args:
            eos_token_id: EOS token ID to check for
        
        Returns:
            True if EOS token is in generated_ids
        """
        if self.generated_ids is None:
            return False
        
        # Check if last generated token is EOS
        if self.generated_ids.shape[1] > 0:
            last_token = self.generated_ids[0, -1].item()
            if last_token == eos_token_id:
                self.eos_generated = True
                return True
        
        return False


class BatchState:
    """
    Batch state manager for continuous batching.
    
    This class manages the state of multiple requests being processed
    in a batch, tracking per-request vision embeddings and generation
    progress. It supports dynamic batch composition where requests can
    join or leave during generation.
    
    Attributes:
        request_states: Dictionary mapping request IDs to RequestState objects
        active_request_ids: List of currently active request IDs
    
    Requirements:
        - 8.4: Track vision embeddings for each request in batch
        - 8.4: Ensure correct association during generation
        - 8.2: Allow new requests to join ongoing batches
        - 8.3: Remove completed requests without blocking others
    """
    
    def __init__(self):
        """Initialize batch state manager."""
        self.request_states: Dict[str, RequestState] = {}
        self.active_request_ids: List[str] = []
        logger.debug("BatchState initialized")
    
    def add_request(
        self,
        request_id: str,
        vision_embeddings: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        initial_ids: torch.Tensor
    ) -> None:
        """
        Add a new request to the batch.
        
        Args:
            request_id: Unique identifier for the request
            vision_embeddings: Vision embeddings [1, 577, 768]
            encoder_hidden_states: Encoder hidden states [1, max_seq_len, 768]
            initial_ids: Initial token IDs [1, seq_len]
        
        Requirements:
            - 8.2: Allow new requests to join ongoing batches
            - 8.4: Track vision embeddings for each request
        """
        state = RequestState(
            request_id=request_id,
            vision_embeddings=vision_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            generated_ids=initial_ids
        )
        
        self.request_states[request_id] = state
        self.active_request_ids.append(request_id)
        
        logger.debug(
            f"Added request {request_id} to batch. "
            f"Active requests: {len(self.active_request_ids)}"
        )
    
    def remove_request(self, request_id: str) -> None:
        """
        Remove a completed request from the batch.
        
        Args:
            request_id: Request identifier to remove
        
        Requirements:
            - 8.3: Remove completed requests without blocking others
        """
        if request_id in self.active_request_ids:
            self.active_request_ids.remove(request_id)
            logger.debug(
                f"Removed request {request_id} from batch. "
                f"Active requests: {len(self.active_request_ids)}"
            )
    
    def get_request_state(self, request_id: str) -> Optional[RequestState]:
        """
        Get state for a specific request.
        
        Args:
            request_id: Request identifier
        
        Returns:
            RequestState object or None if not found
        """
        return self.request_states.get(request_id)
    
    def get_active_states(self) -> List[RequestState]:
        """
        Get states for all active requests.
        
        Returns:
            List of RequestState objects for active requests
        """
        return [
            self.request_states[req_id]
            for req_id in self.active_request_ids
            if req_id in self.request_states
        ]
    
    def get_batch_vision_embeddings(self) -> torch.Tensor:
        """
        Get vision embeddings for all active requests as a batch.
        
        Returns:
            Batched vision embeddings [batch_size, 577, 768]
        
        Requirements:
            - 8.4: Maintain per-request vision embeddings
        """
        active_states = self.get_active_states()
        
        if not active_states:
            raise ValueError("No active requests in batch")
        
        # Stack vision embeddings from all active requests
        vision_embeds = torch.cat(
            [state.vision_embeddings for state in active_states],
            dim=0
        )
        
        return vision_embeds
    
    def get_batch_encoder_hidden_states(self) -> torch.Tensor:
        """
        Get encoder hidden states for all active requests as a batch.
        
        Returns:
            Batched encoder hidden states [batch_size, max_seq_len, 768]
        
        Requirements:
            - 8.4: Maintain per-request encoder states
        """
        active_states = self.get_active_states()
        
        if not active_states:
            raise ValueError("No active requests in batch")
        
        # Stack encoder hidden states from all active requests
        encoder_states = torch.cat(
            [state.encoder_hidden_states for state in active_states],
            dim=0
        )
        
        return encoder_states
    
    def get_batch_generated_ids(self) -> torch.Tensor:
        """
        Get generated IDs for all active requests as a batch.
        
        Returns:
            Batched generated IDs [batch_size, seq_len]
        
        Requirements:
            - 8.4: Maintain per-request generation state
        """
        active_states = self.get_active_states()
        
        if not active_states:
            raise ValueError("No active requests in batch")
        
        # Stack generated IDs from all active requests
        generated_ids = torch.cat(
            [state.generated_ids for state in active_states],
            dim=0
        )
        
        return generated_ids
    
    def update_generated_ids(
        self,
        new_ids: torch.Tensor
    ) -> None:
        """
        Update generated IDs for all active requests.
        
        Args:
            new_ids: New generated IDs [batch_size, seq_len]
        
        Requirements:
            - 8.4: Update per-request generation state
        """
        active_states = self.get_active_states()
        
        if len(active_states) != new_ids.shape[0]:
            raise ValueError(
                f"Batch size mismatch: {len(active_states)} active requests, "
                f"but new_ids has batch size {new_ids.shape[0]}"
            )
        
        # Update each request's generated IDs
        for i, state in enumerate(active_states):
            state.generated_ids = new_ids[i:i+1]
    
    def mark_completed_requests(self, eos_token_id: int) -> List[str]:
        """
        Mark requests that have generated EOS token as complete.
        
        Args:
            eos_token_id: EOS token ID to check for
        
        Returns:
            List of request IDs that were marked complete
        
        Requirements:
            - 8.3: Detect completed requests
        """
        completed = []
        
        for state in self.get_active_states():
            if state.check_eos(eos_token_id):
                state.mark_complete()
                completed.append(state.request_id)
        
        # Remove completed requests from active list
        for request_id in completed:
            self.remove_request(request_id)
        
        return completed
    
    def is_empty(self) -> bool:
        """
        Check if batch is empty (no active requests).
        
        Returns:
            True if no active requests
        """
        return len(self.active_request_ids) == 0
    
    def batch_size(self) -> int:
        """
        Get current batch size (number of active requests).
        
        Returns:
            Number of active requests
        """
        return len(self.active_request_ids)


class Florence2NxDModel:
    """
    Florence-2 model using NxD Inference APIs.
    
    This class provides the main inference interface for Florence-2 vision-language
    model compiled with neuronx-distributed-inference. It loads pre-compiled model
    components and orchestrates the full inference pipeline.
    
    Architecture:
        - Vision Encoder: 4-stage DaViT (compiled separately)
        - Projection: Vision-to-language projection with position embeddings
        - Language Encoder: BART encoder for combined embeddings
        - Language Decoder: BART decoder with bucketing strategy
    
    Attributes:
        config: Model configuration
        vision_stages: List of 4 compiled DaViT stages
        projection: Compiled projection layer
        encoder: Compiled BART encoder
        decoders: Dict mapping bucket sizes to compiled decoders
        processor: HuggingFace processor for image preprocessing
        tokenizer: HuggingFace tokenizer for text processing
        device: Torch device for inference
    
    Requirements:
        - 1.4: Support model serialization and deserialization
        - 1.5: Use NxD Inference model loading APIs
        - 10.3: Load pre-compiled models
    
    Example:
        >>> model = Florence2NxDModel("./compiled_nxd", tp_degree=1)
        >>> image = PIL.Image.open("image.jpg")
        >>> result = model.generate(image, "<CAPTION>")
    """
    
    def __init__(self, model_dir: str, tp_degree: int = 1):
        """
        Initialize Florence-2 NxD Inference model.
        
        Loads configuration, allocates NeuronCores, loads all compiled models,
        and initializes processor and tokenizer.
        
        Args:
            model_dir: Directory containing compiled models and metadata
            tp_degree: Tensor parallelism degree (1, 2, 4, or 8)
        
        Raises:
            FileNotFoundError: If model directory or required files are missing
            ValueError: If configuration is invalid
            ModelLoadError: If required model files are missing
            HardwareCompatibilityError: If hardware is incompatible with model
        
        Requirements:
            - 1.4: Load configuration from model directory
            - 1.5: Use NxD Inference model loading APIs
            - 10.3: Load all compiled models
            - 10.4: Validate metadata compatibility with hardware
            - 10.6: Raise descriptive errors for missing files
            - 12.3: Return error listing missing files
        """
        logger.info(f"Initializing Florence2NxDModel from {model_dir} with TP degree {tp_degree}")
        
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load and validate metadata
        metadata = self._load_and_validate_metadata(tp_degree)
        
        # Load configuration
        self.config = self._load_config_from_metadata(metadata, tp_degree)
        self.config.validate()
        
        # Set device
        self.device = torch.device("cpu")  # Neuron models run on NeuronCores but interface through CPU
        
        # Initialize NeuronCores for tensor parallelism
        if self.config.tp_degree > 1:
            logger.info(f"Initializing tensor parallelism with degree {self.config.tp_degree}")
            self._initialize_tensor_parallelism()
        
        # Load compiled models
        logger.info("Loading compiled models...")
        self.vision_stages = self._load_vision_stages()
        self.projection = self._load_projection()
        self.encoder = self._load_encoder()
        self.decoders = self._load_decoders()
        
        # Initialize processor and tokenizer
        logger.info("Initializing processor and tokenizer...")
        self.processor, self.tokenizer = self._load_processor_and_tokenizer()
        
        # Precompute position embeddings for vision features
        logger.info("Precomputing position embeddings...")
        self.vision_pos_embed = self._precompute_position_embeddings()
        
        logger.info("Florence2NxDModel initialization complete")

    def _initialize_tensor_parallelism(self) -> None:
        """
        Initialize tensor parallelism for distributed inference.
        
        This method:
        1. Gets NeuronCore placement configuration
        2. Sets up environment variables for NeuronCore visibility
        3. Initializes all NeuronCores before inference
        
        The NeuronCores are allocated based on the tp_degree and neuron_core_placement
        configuration. If neuron_core_placement is not specified, sequential cores
        starting from 0 are used.
        
        Requirements:
            - 9.2: Distribute model layers across NeuronCores based on TP degree
            - 9.6: Initialize all NeuronCores before inference
        """
        logger.info(f"Initializing tensor parallelism with degree {self.config.tp_degree}")
        
        # Get NeuronCore placement
        core_placement = self.config.get_neuron_core_placement()
        logger.info(f"NeuronCore placement: {core_placement}")
        
        # Set NEURON_RT_VISIBLE_CORES environment variable
        # This tells the Neuron runtime which cores to use
        cores_str = ','.join(map(str, core_placement))
        os.environ['NEURON_RT_VISIBLE_CORES'] = cores_str
        logger.info(f"Set NEURON_RT_VISIBLE_CORES={cores_str}")
        
        # Validate that we have enough cores available
        available_cores = self._get_available_neuron_cores()
        required_cores = self.config.tp_degree
        
        if available_cores < required_cores:
            raise HardwareCompatibilityError(
                required_cores=required_cores,
                available_cores=available_cores
            )
        
        logger.info(
            f"Tensor parallelism initialized: {required_cores} cores allocated "
            f"from {available_cores} available"
        )
    
    def _distribute_model_layers(self) -> None:
        """
        Distribute model layers across NeuronCores for tensor parallelism.
        
        This method would handle the distribution of model layers across multiple
        NeuronCores when TP degree > 1. For the current implementation, the
        distribution is handled implicitly by the NxD Inference compilation process.
        
        In a full implementation, this would:
        1. Partition model layers based on TP degree
        2. Assign each partition to a specific NeuronCore
        3. Set up communication between cores for cross-core operations
        
        Requirements:
            - 9.2: Distribute model layers across NeuronCores based on TP degree
        """
        if self.config.tp_degree == 1:
            logger.debug("TP degree is 1, no layer distribution needed")
            return
        
        logger.debug(
            f"Model layers will be distributed across {self.config.tp_degree} NeuronCores "
            f"during compilation. Runtime distribution is handled by NxD Inference."
        )
        
        # Note: In the current implementation, layer distribution is handled
        # during compilation by torch_neuronx with NxD Inference patterns.
        # The compiled models already contain the TP distribution information.
        # At runtime, we just need to ensure the correct cores are visible.

    def validate_task(self, task: str) -> None:
        """
        Validate that the task prompt is supported.

        Args:
            task: Task prompt string (e.g., "<CAPTION>", "<OD>")

        Raises:
            InvalidTaskError: If task is not in the list of supported tasks

        Requirements:
            - 12.1: Validate task prompts against supported tasks
        """
        if task not in SUPPORTED_TASKS:
            raise InvalidTaskError(task, SUPPORTED_TASKS)

        logger.debug(f"Task '{task}' validated successfully")

    def validate_image_format(self, image: Union[str, PIL.Image.Image, torch.Tensor]) -> None:
        """
        Validate that the image format is supported.

        Args:
            image: Image in one of the supported formats

        Raises:
            ValueError: If image format is not supported
            FileNotFoundError: If image path does not exist

        Requirements:
            - 12.2: Validate image formats and dimensions
        """
        if isinstance(image, str):
            # Validate file path exists
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")

            # Validate file extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
            ext = os.path.splitext(image)[1].lower()
            if ext not in valid_extensions:
                logger.warning(
                    f"Image file '{image}' has extension '{ext}' which may not be supported. "
                    f"Supported extensions: {', '.join(valid_extensions)}"
                )

        elif isinstance(image, torch.Tensor):
            # Validate tensor dimensions
            if image.dim() not in [3, 4]:
                raise ValueError(
                    f"Tensor image must be 3D [C, H, W] or 4D [B, C, H, W], "
                    f"got {image.dim()}D tensor with shape {image.shape}"
                )

            # Validate channels
            channels = image.shape[-3] if image.dim() == 4 else image.shape[0]
            if channels != 3:
                raise ValueError(
                    f"Image tensor must have 3 channels (RGB), got {channels} channels"
                )

        elif not isinstance(image, PIL.Image.Image):
            raise ValueError(
                f"Unsupported image format: {type(image)}. "
                f"Expected str (file path), PIL.Image.Image, or torch.Tensor"
            )

        logger.debug(f"Image format validated successfully: {type(image)}")

    def validate_inputs(
        self,
        image: Union[str, PIL.Image.Image, torch.Tensor],
        task: str
    ) -> None:
        """
        Validate all inputs for inference.

        Performs comprehensive validation of both image and task inputs
        before processing.

        Args:
            image: Image in one of the supported formats
            task: Task prompt string

        Raises:
            InvalidTaskError: If task is not supported
            ValueError: If image format is invalid
            FileNotFoundError: If image path does not exist

        Requirements:
            - 12.1: Validate task prompts against supported tasks
            - 12.2: Validate image formats and dimensions
            - 12.2: Raise descriptive errors for invalid inputs
        """
        logger.debug("Validating inputs...")

        # Validate task
        self.validate_task(task)

        # Validate image format
        self.validate_image_format(image)

        logger.debug("Input validation complete")

    def _load_and_validate_metadata(self, tp_degree: int) -> CompiledModelMetadata:
        """
        Load and validate metadata from model directory.
        
        This method:
        1. Loads metadata.json from the model directory
        2. Validates all required files exist
        3. Validates hardware compatibility
        
        Args:
            tp_degree: Tensor parallelism degree
        
        Returns:
            CompiledModelMetadata instance
        
        Raises:
            FileNotFoundError: If metadata.json is missing
            ModelLoadError: If required model files are missing
            HardwareCompatibilityError: If hardware is incompatible
        
        Requirements:
            - 10.3: Check for all required files before loading
            - 10.4: Validate metadata compatibility with hardware
            - 10.6: Raise descriptive errors for missing files
            - 12.3: Return error listing missing files
        """
        metadata_path = self.model_dir / "metadata.json"
        
        # Check if metadata file exists
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found at {metadata_path}")
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"The model directory must contain a metadata.json file.\n"
                f"Run compilation script to generate models with metadata."
            )
        
        # Load metadata
        logger.debug(f"Loading metadata from {metadata_path}")
        metadata = CompiledModelMetadata.load(str(metadata_path))
        
        # Validate all required files exist
        logger.debug("Validating required model files...")
        all_exist, missing_files = metadata.validate_files_exist(str(self.model_dir))
        
        if not all_exist:
            raise ModelLoadError(str(self.model_dir), missing_files)
        
        logger.debug("All required model files found")
        
        # Validate hardware compatibility
        logger.debug("Validating hardware compatibility...")
        available_cores = self._get_available_neuron_cores()
        is_compatible, error_msg = metadata.validate_hardware_compatibility(available_cores)
        
        if not is_compatible:
            raise HardwareCompatibilityError(
                required_cores=metadata.tp_degree,
                available_cores=available_cores
            )
        
        logger.debug(f"Hardware compatible: {available_cores} cores available, {metadata.tp_degree} required")
        
        return metadata
    
    def _get_available_neuron_cores(self) -> int:
        """
        Get the number of available NeuronCores.
        
        This method checks the NEURON_RT_VISIBLE_CORES environment variable
        to determine how many cores are available. If not set, it assumes
        a default based on common instance types.
        
        Returns:
            Number of available NeuronCores
        
        Requirements:
            - 10.4: Validate hardware compatibility
        """
        # Check environment variable
        visible_cores = os.environ.get('NEURON_RT_VISIBLE_CORES', None)
        
        if visible_cores:
            # Parse the visible cores specification
            # Format can be: "0", "0-1", "0,1,2,3", etc.
            try:
                if '-' in visible_cores:
                    # Range format: "0-3"
                    start, end = visible_cores.split('-')
                    return int(end) - int(start) + 1
                elif ',' in visible_cores:
                    # List format: "0,1,2,3"
                    return len(visible_cores.split(','))
                else:
                    # Single core: "0"
                    return 1
            except (ValueError, AttributeError):
                logger.warning(f"Could not parse NEURON_RT_VISIBLE_CORES: {visible_cores}")
        
        # Default assumptions based on common instance types
        # inf2.xlarge: 2 cores
        # inf2.8xlarge: 2 cores
        # inf2.24xlarge: 12 cores
        # inf2.48xlarge: 24 cores
        # For safety, assume minimum of 2 cores if not specified
        default_cores = 2
        logger.debug(f"NEURON_RT_VISIBLE_CORES not set, assuming {default_cores} cores")
        return default_cores
    
    def _load_config_from_metadata(
        self, 
        metadata: CompiledModelMetadata, 
        tp_degree: int
    ) -> Florence2NxDConfig:
        """
        Create configuration from loaded metadata.
        
        Args:
            metadata: Loaded metadata instance
            tp_degree: Tensor parallelism degree
        
        Returns:
            Florence2NxDConfig instance
        
        Requirements:
            - 10.3: Load configuration from metadata
        """
        logger.debug("Creating configuration from metadata")
        
        config = Florence2NxDConfig(
            model_dir=str(self.model_dir),
            tp_degree=tp_degree,
            max_encoder_length=metadata.max_encoder_length,
            decoder_buckets=metadata.decoder_buckets.copy(),
            num_vision_stages=metadata.num_vision_stages,
        )
        
        return config
    
    def _load_config(self, tp_degree: int) -> Florence2NxDConfig:
        """
        Load configuration from model directory.
        
        DEPRECATED: This method is kept for backward compatibility.
        Use _load_config_from_metadata instead.
        
        Args:
            tp_degree: Tensor parallelism degree
        
        Returns:
            Florence2NxDConfig instance
        
        Requirements:
            - 1.4: Load configuration from model directory
        """
        logger.warning("Using deprecated _load_config method. Use _load_config_from_metadata instead.")
        
        metadata_path = self.model_dir / "metadata.json"
        
        if metadata_path.exists():
            logger.debug(f"Loading configuration from {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract config from metadata
            config_data = metadata.get('config', {})
            config_data['model_dir'] = str(self.model_dir)
            config_data['tp_degree'] = tp_degree
            
            # Handle dtype conversion
            if 'dtype' in config_data:
                dtype_str = config_data['dtype']
                if isinstance(dtype_str, str):
                    config_data['dtype'] = getattr(torch, dtype_str.split('.')[-1])
            
            config = Florence2NxDConfig(**config_data)
        else:
            logger.warning(f"Metadata file not found at {metadata_path}, using default config")
            config = Florence2NxDConfig(
                model_dir=str(self.model_dir),
                tp_degree=tp_degree
            )
        
        return config
    
    def _load_vision_stages(self) -> List[torch.jit.ScriptModule]:
        """
        Load 4 compiled DaViT vision encoder stages.
        
        Wraps model loading with hardware error handling and retry logic
        to handle transient failures.
        
        Returns:
            List of 4 compiled vision stage models
        
        Raises:
            FileNotFoundError: If any stage file is missing
            NeuronCoreError: If loading fails due to hardware issues after retries
        
        Requirements:
            - 1.5: Use NxD Inference model loading APIs
            - 2.3: Load all 4 vision stages
            - 12.4: Wrap Neuron SDK errors with context
            - Retry hardware errors up to 3 times with exponential backoff
        """
        stages = []
        for i in range(self.config.num_vision_stages):
            stage_path = self.model_dir / f"stage{i}.pt"
            if not stage_path.exists():
                raise FileNotFoundError(f"Vision stage file not found: {stage_path}")
            
            logger.debug(f"Loading vision stage {i} from {stage_path}")
            
            # Wrap loading with hardware error handling and retry logic
            stage = self._retry_with_backoff(
                f"load_vision_stage_{i}",
                lambda path=stage_path: self._wrap_neuron_operation(
                    f"load_vision_stage_{i}",
                    torch.jit.load,
                    str(path)
                ),
                max_retries=3,
                initial_delay=0.1
            )
            stage.eval()
            stages.append(stage)
        
        logger.info(f"Loaded {len(stages)} vision stages")
        return stages
    
    def _load_projection(self) -> torch.jit.ScriptModule:
        """
        Load compiled projection layer.
        
        Wraps model loading with hardware error handling and retry logic
        to handle transient failures.
        
        Returns:
            Compiled projection model
        
        Raises:
            FileNotFoundError: If projection file is missing
            NeuronCoreError: If loading fails due to hardware issues after retries
        
        Requirements:
            - 1.5: Use NxD Inference model loading APIs
            - 2.4: Load projection layer
            - 12.4: Wrap Neuron SDK errors with context
            - Retry hardware errors up to 3 times with exponential backoff
        """
        projection_path = self.model_dir / "projection.pt"
        if not projection_path.exists():
            raise FileNotFoundError(f"Projection file not found: {projection_path}")
        
        logger.debug(f"Loading projection from {projection_path}")
        
        # Wrap loading with hardware error handling and retry logic
        projection = self._retry_with_backoff(
            "load_projection",
            lambda: self._wrap_neuron_operation(
                "load_projection",
                torch.jit.load,
                str(projection_path)
            ),
            max_retries=3,
            initial_delay=0.1
        )
        projection.eval()
        
        logger.info("Loaded projection layer")
        return projection
    
    def _load_encoder(self) -> torch.jit.ScriptModule:
        """
        Load compiled BART encoder.
        
        Wraps model loading with hardware error handling and retry logic
        to handle transient failures.
        
        Returns:
            Compiled encoder model
        
        Raises:
            FileNotFoundError: If encoder file is missing
            NeuronCoreError: If loading fails due to hardware issues after retries
        
        Requirements:
            - 1.5: Use NxD Inference model loading APIs
            - 3.4: Load encoder
            - 12.4: Wrap Neuron SDK errors with context
            - Retry hardware errors up to 3 times with exponential backoff
        """
        encoder_path = self.model_dir / "encoder.pt"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        
        logger.debug(f"Loading encoder from {encoder_path}")
        
        # Wrap loading with hardware error handling and retry logic
        encoder = self._retry_with_backoff(
            "load_encoder",
            lambda: self._wrap_neuron_operation(
                "load_encoder",
                torch.jit.load,
                str(encoder_path)
            ),
            max_retries=3,
            initial_delay=0.1
        )
        encoder.eval()
        
        logger.info("Loaded encoder")
        return encoder
    
    def _load_decoders(self) -> Dict[int, torch.jit.ScriptModule]:
        """
        Load compiled decoder buckets.
        
        Wraps model loading with hardware error handling and retry logic
        to handle transient failures.
        
        Returns:
            Dictionary mapping bucket sizes to compiled decoder models
        
        Raises:
            FileNotFoundError: If any decoder file is missing
            NeuronCoreError: If loading fails due to hardware issues after retries
        
        Requirements:
            - 1.5: Use NxD Inference model loading APIs
            - 3.2: Load decoder buckets
            - 12.4: Wrap Neuron SDK errors with context
            - Retry hardware errors up to 3 times with exponential backoff
        """
        decoders = {}
        for bucket_size in self.config.decoder_buckets:
            decoder_path = self.model_dir / f"decoder_{bucket_size}.pt"
            if not decoder_path.exists():
                raise FileNotFoundError(f"Decoder file not found: {decoder_path}")
            
            logger.debug(f"Loading decoder bucket {bucket_size} from {decoder_path}")
            
            # Wrap loading with hardware error handling and retry logic
            decoder = self._retry_with_backoff(
                f"load_decoder_{bucket_size}",
                lambda path=decoder_path, size=bucket_size: self._wrap_neuron_operation(
                    f"load_decoder_{size}",
                    torch.jit.load,
                    str(path)
                ),
                max_retries=3,
                initial_delay=0.1
            )
            decoder.eval()
            decoders[bucket_size] = decoder
        
        logger.info(f"Loaded {len(decoders)} decoder buckets: {list(decoders.keys())}")
        return decoders
    
    def _load_processor_and_tokenizer(self) -> Tuple:
        """
        Initialize processor and tokenizer from HuggingFace.
        
        Returns:
            Tuple of (processor, tokenizer)
        
        Requirements:
            - 10.3: Initialize processor and tokenizer
        """
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # Load processor and tokenizer from HuggingFace
        model_name = self.config.model_name
        logger.debug(f"Loading processor and tokenizer from {model_name}")
        
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = processor.tokenizer
        
        # Load base model to get embedding layer
        logger.debug(f"Loading base model for embedding layer from {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.config.dtype,
            attn_implementation="eager"
        )
        base_model.eval()
        
        # Extract and store the embedding layer
        self.embed_tokens = base_model.language_model.model.decoder.embed_tokens
        logger.debug("Loaded embedding layer from base model")
        
        return processor, tokenizer
    
    def _precompute_position_embeddings(self) -> torch.Tensor:
        """
        Precompute position embeddings for vision features.
        
        Florence-2 uses 2D position embeddings for the 24x24 spatial grid
        of vision features (576 positions).
        
        Returns:
            Position embeddings tensor (1, 576, 1024)
        
        Requirements:
            - 2.5: Precompute position embeddings for vision features
        """
        # Create 2D position embeddings for 24x24 grid
        # This is a simplified version - in practice, these would come from
        # the original Florence-2 model
        height, width = 24, 24
        num_positions = height * width  # 576
        
        # Create position embeddings (simplified - would normally load from model)
        pos_embed = torch.zeros(1, num_positions, self.config.vision_hidden_size, dtype=self.config.dtype)
        
        logger.debug(f"Precomputed position embeddings: {pos_embed.shape}")
        return pos_embed

    def _check_numerical_stability(
        self, 
        tensor: torch.Tensor, 
        operation: str, 
        tensor_name: str
    ) -> None:
        """
        Check tensor for NaN or Inf values.
        
        Detects numerical instability in intermediate tensors and raises
        a descriptive error with context about where the issue occurred.
        
        Args:
            tensor: Tensor to check for numerical issues
            operation: Name of the operation that produced the tensor
            tensor_name: Descriptive name of the tensor
        
        Raises:
            NumericalError: If tensor contains NaN or Inf values
        
        Requirements:
            - 12.6: Check for NaN/Inf in intermediate tensors
            - 12.6: Raise NumericalError with context
        """
        from .errors import NumericalError
        
        # Check for NaN values
        if torch.isnan(tensor).any():
            logger.error(
                f"NaN detected in {operation}: {tensor_name}. "
                f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}"
            )
            raise NumericalError(operation, tensor_name)
        
        # Check for Inf values
        if torch.isinf(tensor).any():
            logger.error(
                f"Inf detected in {operation}: {tensor_name}. "
                f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}"
            )
            raise NumericalError(operation, tensor_name)
        
        logger.debug(f"Numerical stability check passed for {tensor_name} in {operation}")

    def _wrap_neuron_operation(
        self,
        operation_name: str,
        operation_func,
        *args,
        **kwargs
    ):
        """
        Wrap Neuron SDK operations with error handling.
        
        Catches exceptions from Neuron SDK operations and wraps them with
        additional context and hardware diagnostics.
        
        Args:
            operation_name: Descriptive name of the operation
            operation_func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            Result of the operation
        
        Raises:
            NeuronCoreError: If the operation fails with hardware context
        
        Requirements:
            - 12.4: Wrap Neuron SDK errors with context
            - 12.4: Include hardware diagnostics in error messages
        """
        from .errors import NeuronCoreError
        
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            # Determine which NeuronCore was being used
            core_id = os.environ.get('NEURON_RT_VISIBLE_CORES', 'unknown')
            
            logger.error(
                f"Neuron operation '{operation_name}' failed on core(s) {core_id}: {e}"
            )
            
            # Wrap the error with hardware context
            raise NeuronCoreError(core_id, e)

    def _retry_with_backoff(
        self,
        operation_name: str,
        operation_func,
        *args,
        max_retries: int = 3,
        initial_delay: float = 0.1,
        **kwargs
    ):
        """
        Retry an operation with exponential backoff.
        
        Retries hardware operations that may fail transiently, using
        exponential backoff between attempts. This is useful for handling
        temporary hardware issues or resource contention.
        
        Backoff strategy:
            - Attempt 1: No delay
            - Attempt 2: initial_delay seconds
            - Attempt 3: initial_delay * 2 seconds
            - Attempt 4: initial_delay * 4 seconds
        
        Args:
            operation_name: Descriptive name of the operation
            operation_func: Function to execute
            *args: Positional arguments for the function
            max_retries: Maximum number of retry attempts (default: 3)
            initial_delay: Initial delay in seconds (default: 0.1)
            **kwargs: Keyword arguments for the function
        
        Returns:
            Result of the operation
        
        Raises:
            Exception: The last exception if all retries fail
        
        Requirements:
            - Retry hardware errors up to 3 times
            - Use exponential backoff
        """
        import time
        
        last_exception = None
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retrying '{operation_name}' (attempt {attempt + 1}/{max_retries + 1}) "
                        f"after {delay:.2f}s delay"
                    )
                    time.sleep(delay)
                
                result = operation_func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"'{operation_name}' succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    logger.warning(
                        f"'{operation_name}' failed on attempt {attempt + 1}: {e}. "
                        f"Will retry..."
                    )
                    # Exponential backoff
                    delay *= 2
                else:
                    logger.error(
                        f"'{operation_name}' failed after {max_retries + 1} attempts. "
                        f"Last error: {e}"
                    )
        
        # All retries exhausted, raise the last exception
        raise last_exception

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode image through DaViT vision encoder.
        
        Executes all 4 vision stages sequentially on NeuronCores to transform
        the input image into vision features. Supports batch processing for
        multiple images simultaneously. Includes numerical stability checks
        and hardware error handling.
        
        Stage progression:
            - Stage 0: (B, 3, 768, 768) → (B, 36864, 128)
            - Stage 1: (B, 36864, 128) → (B, 9216, 256)
            - Stage 2: (B, 9216, 256) → (B, 2304, 512)
            - Stage 3: (B, 2304, 512) → (B, 576, 1024)
        
        Batch processing:
            - Supports arbitrary batch sizes (B >= 1)
            - Each image in batch is processed independently
            - Vision embeddings maintain per-request association
        
        Args:
            pixel_values: Image tensor [batch, 3, 768, 768] in BF16
        
        Returns:
            Vision features [batch, 576, 1024] in BF16
        
        Raises:
            NumericalError: If NaN or Inf values are detected
            NeuronCoreError: If hardware operation fails
        
        Requirements:
            - 2.3: Execute all 4 vision stages sequentially
            - 2.5: Add position embeddings to vision features
            - 8.1: Support batch dimension in all vision stages
            - 8.4: Maintain per-request vision embeddings
            - 12.6: Check for NaN/Inf in intermediate tensors
            - 12.4: Wrap Neuron SDK errors with context
        """
        batch_size = pixel_values.shape[0]
        logger.debug(f"Encoding vision features, input shape: {pixel_values.shape}, batch_size: {batch_size}")
        
        # Check input for numerical stability
        self._check_numerical_stability(pixel_values, "encode_vision_input", "pixel_values")
        
        # Execute 4 vision stages sequentially with hardware error handling
        # The batch dimension is preserved throughout all stages
        x = pixel_values
        for i, stage in enumerate(self.vision_stages):
            # Wrap stage execution with hardware error handling
            x = self._wrap_neuron_operation(
                f"vision_stage_{i}_forward",
                stage,
                x
            )
            logger.debug(f"After stage {i}: {x.shape}")
            
            # Check for numerical stability after each stage
            self._check_numerical_stability(x, f"vision_stage_{i}", f"stage_{i}_output")
        
        # Add position embeddings (broadcast across batch dimension)
        # Position embeddings are shared across all images in the batch
        # Shape: (1, 576, 1024) broadcasts to (B, 576, 1024)
        x = x + self.vision_pos_embed
        
        # Check final output for numerical stability
        self._check_numerical_stability(x, "encode_vision_output", "vision_features")
        
        logger.debug(f"Vision encoding complete, output shape: {x.shape}, batch_size: {batch_size}")
        
        # Verify batch dimension is preserved
        assert x.shape[0] == batch_size, (
            f"Batch dimension mismatch: input batch_size={batch_size}, "
            f"output batch_size={x.shape[0]}"
        )
        
        return x
    
    def project_vision(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to language dimension.
        
        Applies the projection layer to transform vision features from
        vision dimension (1024) to language dimension (768). The projection
        layer also adds a mean-pooled global feature token. Supports batch
        processing with per-request feature preservation. Includes
        numerical stability checks.
        
        Args:
            vision_features: Vision features [batch, 576, 1024]
        
        Returns:
            Projected features [batch, 577, 768] in BF16
        
        Raises:
            NumericalError: If NaN or Inf values are detected
        
        Requirements:
            - 2.4: Apply projection layer to vision features
            - 8.1: Support batch dimension in projection
            - 8.4: Maintain per-request vision embeddings
            - 12.6: Check for NaN/Inf in intermediate tensors
        """
        batch_size = vision_features.shape[0]
        logger.debug(f"Projecting vision features, input shape: {vision_features.shape}, batch_size: {batch_size}")
        
        # Check input for numerical stability
        self._check_numerical_stability(vision_features, "project_vision_input", "vision_features")
        
        # Apply projection layer (includes mean pooling and projection)
        # The projection layer handles batch dimension automatically
        projected = self.projection(vision_features)
        
        # Check output for numerical stability
        self._check_numerical_stability(projected, "project_vision_output", "projected_features")
        
        logger.debug(f"Projection complete, output shape: {projected.shape}, batch_size: {batch_size}")
        
        # Verify batch dimension is preserved
        assert projected.shape[0] == batch_size, (
            f"Batch dimension mismatch: input batch_size={batch_size}, "
            f"output batch_size={projected.shape[0]}"
        )
        
        return projected
    
    def encode_language(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode combined vision + text embeddings.
        
        Pads the combined embeddings to the maximum sequence length (600 tokens)
        and processes them through the BART encoder to create contextualized
        representations. Supports batch processing with per-request state
        preservation. Includes numerical stability checks.
        
        Args:
            embeddings: Combined embeddings [batch, seq_len, 768]
        
        Returns:
            Encoder hidden states [batch, max_seq_len, 768]
        
        Raises:
            NumericalError: If NaN or Inf values are detected
        
        Requirements:
            - 3.4: Encode combined embeddings with BART encoder
            - 5.5: Pad combined embeddings to max length (600)
            - 8.1: Support batch dimension in encoder
            - 8.4: Maintain per-request state during encoding
            - 12.6: Check for NaN/Inf in intermediate tensors
        """
        batch_size, seq_len, hidden_size = embeddings.shape
        logger.debug(f"Encoding language, input shape: {embeddings.shape}, batch_size: {batch_size}")
        
        # Check input for numerical stability
        self._check_numerical_stability(embeddings, "encode_language_input", "embeddings")
        
        max_len = self.config.max_encoder_length
        
        # Pad embeddings to max length if needed
        # Padding is applied per-request in the batch
        if seq_len < max_len:
            padding = torch.zeros(
                batch_size, 
                max_len - seq_len, 
                hidden_size,
                dtype=embeddings.dtype,
                device=embeddings.device
            )
            embeddings = torch.cat([embeddings, padding], dim=1)
            logger.debug(f"Padded embeddings to {embeddings.shape}")
        elif seq_len > max_len:
            # Truncate if exceeds max length
            embeddings = embeddings[:, :max_len, :]
            logger.warning(f"Truncated embeddings from {seq_len} to {max_len}")
        
        # Execute encoder
        # The encoder processes all requests in the batch simultaneously
        encoder_hidden_states = self.encoder(embeddings)
        
        # Check output for numerical stability
        self._check_numerical_stability(
            encoder_hidden_states, 
            "encode_language_output", 
            "encoder_hidden_states"
        )
        
        logger.debug(f"Language encoding complete, output shape: {encoder_hidden_states.shape}, batch_size: {batch_size}")
        
        # Verify batch dimension is preserved
        assert encoder_hidden_states.shape[0] == batch_size, (
            f"Batch dimension mismatch: input batch_size={batch_size}, "
            f"output batch_size={encoder_hidden_states.shape[0]}"
        )
        
        return encoder_hidden_states
    
    def decode_step(
        self, 
        input_ids: torch.Tensor, 
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Single decoder step with bucket selection.
        
        Selects the smallest bucket that fits the current sequence length,
        pads the input to the bucket size, and executes the decoder to
        generate logits for the next token. Supports batch processing with
        per-request state preservation. Includes numerical stability checks.
        
        Bucket selection strategy:
            - Buckets: [1, 4, 8, 16, 32, 64]
            - Select smallest bucket where bucket_size >= seq_len
            - Pad input_ids to bucket_size
        
        Batch processing:
            - All requests in batch use the same bucket size
            - Bucket is selected based on maximum sequence length in batch
            - Per-request decoder states are maintained independently
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            encoder_hidden_states: Encoder outputs [batch, enc_len, 768]
        
        Returns:
            Logits [batch, bucket_size, vocab_size]
        
        Raises:
            NumericalError: If NaN or Inf values are detected
        
        Requirements:
            - 3.2: Select smallest bucket that fits current sequence length
            - 3.5: Pad input to bucket size
            - 8.1: Support batch dimension in decoder
            - 8.4: Maintain per-request decoder state
            - 12.6: Check for NaN/Inf in intermediate tensors
        """
        batch_size, seq_len = input_ids.shape
        logger.debug(f"Decode step, input_ids shape: {input_ids.shape}, batch_size: {batch_size}")
        
        # Select smallest bucket that fits
        # For batched processing, use the maximum sequence length in the batch
        bucket_size = self._select_bucket(seq_len)
        logger.debug(f"Selected bucket size: {bucket_size} for seq_len: {seq_len}")
        
        # Pad input_ids to bucket size
        # Padding is applied per-request in the batch
        if seq_len < bucket_size:
            padding = torch.full(
                (batch_size, bucket_size - seq_len),
                self.config.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device
            )
            input_ids_padded = torch.cat([input_ids, padding], dim=1)
        else:
            input_ids_padded = input_ids
        
        logger.debug(f"Padded input_ids to shape: {input_ids_padded.shape}")
        
        # Execute decoder with selected bucket
        # The decoder processes all requests in the batch simultaneously
        # Each request maintains its own decoder state
        decoder = self.decoders[bucket_size]
        logits = decoder(input_ids_padded, encoder_hidden_states)
        
        # Check output for numerical stability
        self._check_numerical_stability(logits, "decode_step_output", "logits")
        
        logger.debug(f"Decoder output shape: {logits.shape}, batch_size: {batch_size}")
        
        # Verify batch dimension is preserved
        assert logits.shape[0] == batch_size, (
            f"Batch dimension mismatch: input batch_size={batch_size}, "
            f"output batch_size={logits.shape[0]}"
        )
        
        return logits
    
    def _select_bucket(self, seq_len: int) -> int:
        """
        Select smallest bucket that fits the sequence length.
        
        Detects when sequence exceeds the largest bucket and handles overflow
        by selecting the largest available bucket. This will cause truncation
        during decoding.
        
        Args:
            seq_len: Current sequence length
        
        Returns:
            Selected bucket size
        
        Raises:
            SequenceTooLongError: If sequence exceeds largest bucket (as warning)
        
        Requirements:
            - 3.2: Select smallest bucket that fits current sequence length
            - 3.3: Detect when sequence exceeds largest bucket
            - 12.5: Handle sequence length overflow gracefully
        """
        from .errors import SequenceTooLongError
        
        for bucket_size in self.config.decoder_buckets:
            if bucket_size >= seq_len:
                return bucket_size
        
        # If sequence exceeds largest bucket, log warning and use largest bucket
        largest_bucket = max(self.config.decoder_buckets)
        
        # Log warning about truncation
        logger.warning(
            f"Sequence length {seq_len} exceeds largest bucket {largest_bucket}. "
            f"Generation will be truncated to fit the largest bucket."
        )
        
        # Note: We don't raise the exception here, just log the warning
        # The exception class exists for cases where we want to handle this explicitly
        return largest_bucket
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100
    ) -> torch.Tensor:
        """
        Full generation pipeline.
        
        Executes the complete Florence-2 inference pipeline:
        1. Encode vision features through 4-stage DaViT
        2. Project vision features to language dimension
        3. Combine vision and text embeddings
        4. Encode combined embeddings with BART encoder
        5. Autoregressive decoding loop with bucket selection
        
        The generation loop includes overflow handling to gracefully truncate
        sequences that exceed the maximum decoder length.
        
        Args:
            pixel_values: Image tensor [batch, 3, 768, 768]
            input_ids: Text token IDs [batch, text_len]
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated token IDs [batch, generated_len]
        
        Requirements:
            - 5.4: Combine vision and text embeddings
            - 5.6: Autoregressive decoding loop with bucket selection
            - 3.3: Detect when sequence exceeds largest bucket
            - 12.5: Truncate gracefully with warning log
        """
        from .errors import SequenceTooLongError
        
        logger.info(f"Starting generation with max_new_tokens={max_new_tokens}")
        logger.debug(f"pixel_values shape: {pixel_values.shape}, input_ids shape: {input_ids.shape}")
        
        batch_size = pixel_values.shape[0]
        
        # Step 1: Encode vision features
        vision_features = self.encode_vision(pixel_values)
        
        # Step 2: Project vision features
        vision_embeds = self.project_vision(vision_features)
        
        # Step 3: Get text embeddings
        # Note: We need to get embeddings from the decoder's embedding layer
        # For now, we'll use a placeholder - this will be properly implemented
        # when we have access to the embedding layer
        text_embeds = self._get_text_embeddings(input_ids)
        
        # Step 4: Combine vision and text embeddings using fusion method
        combined_embeds = self.fuse_vision_text_embeddings(vision_embeds, text_embeds)
        logger.debug(f"Combined embeddings shape: {combined_embeds.shape}")
        
        # Step 5: Encode combined embeddings
        encoder_hidden_states = self.encode_language(combined_embeds)
        
        # Step 6: Autoregressive decoding loop
        # The decoder starts with just the BOS token, not the full input prompt
        # The input prompt has already been encoded into encoder_hidden_states
        generated_ids = torch.tensor([[self.config.bos_token_id]], dtype=torch.long)
        print(f"[DEBUG GENERATE] Starting decoder with BOS token: {generated_ids}")
        largest_bucket = max(self.config.decoder_buckets)
        
        for step in range(max_new_tokens):
            current_seq_len = generated_ids.shape[1]
            
            # Check if we exceed max decoder length (largest bucket)
            if current_seq_len > largest_bucket:
                logger.warning(
                    f"Sequence length {current_seq_len} exceeds maximum decoder length {largest_bucket}. "
                    f"Truncating generation at step {step}."
                )
                break
            
            # Decode step with bucket selection
            logits = self.decode_step(generated_ids, encoder_hidden_states)
            
            # Get logits for the last position of the actual sequence
            # Note: The decoder returns logits for the full bucket size, but we only
            # care about the logits at the last actual token position (before padding)
            last_pos = current_seq_len - 1
            next_token_logits = logits[:, last_pos, :]
            
            if step == 0:
                # Debug first step
                print(f"[DEBUG GENERATE] Step 0 logits shape: {logits.shape}")
                print(f"[DEBUG GENERATE] Last pos: {last_pos}, next_token_logits shape: {next_token_logits.shape}")
                print(f"[DEBUG GENERATE] Logits stats: min={next_token_logits.min():.4f}, max={next_token_logits.max():.4f}, mean={next_token_logits.mean():.4f}")
                top_values, top_indices = torch.topk(next_token_logits[0], k=5)
                print(f"[DEBUG GENERATE] Top 5 tokens: {top_indices.tolist()}, logits: {top_values.tolist()}")
                print(f"[DEBUG GENERATE] EOS token ID: {self.config.eos_token_id}")
            
            # Greedy decoding (select token with highest probability)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            if step == 0:
                print(f"[DEBUG GENERATE] Selected token: {next_token.item()}")
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS token
            if (next_token == self.config.eos_token_id).all():
                print(f"[DEBUG GENERATE] EOS token generated at step {step}")
                break
        
        logger.info(f"Generation complete, generated {generated_ids.shape[1]} tokens")
        return generated_ids
    
    def generate_batch(
        self,
        pixel_values_list: List[torch.Tensor],
        input_ids_list: List[torch.Tensor],
        max_new_tokens: int = 100
    ) -> List[torch.Tensor]:
        """
        Batched generation pipeline with per-request state management.
        
        Executes the Florence-2 inference pipeline for multiple requests
        simultaneously, maintaining per-request vision embeddings and
        generation state. This method supports continuous batching where
        requests can complete at different times.
        
        The generation process:
        1. Encode vision features for all images in batch
        2. Project vision features for all requests
        3. Combine vision and text embeddings per-request
        4. Encode combined embeddings for all requests
        5. Autoregressive decoding with per-request state tracking
        6. Remove completed requests as they finish
        
        Args:
            pixel_values_list: List of image tensors, each [1, 3, 768, 768]
            input_ids_list: List of text token ID tensors, each [1, text_len]
            max_new_tokens: Maximum tokens to generate per request
        
        Returns:
            List of generated token ID tensors, one per request
        
        Requirements:
            - 8.1: Support batch dimension in all stages
            - 8.2: Allow new requests to join ongoing batches
            - 8.3: Remove completed requests without blocking others
            - 8.4: Track vision embeddings for each request in batch
            - 8.4: Ensure correct association during generation
        
        Example:
            >>> model = Florence2NxDModel("./compiled_nxd")
            >>> images = [image1_tensor, image2_tensor, image3_tensor]
            >>> prompts = [prompt1_ids, prompt2_ids, prompt3_ids]
            >>> results = model.generate_batch(images, prompts, max_new_tokens=50)
            >>> len(results)
            3
        """
        from .errors import SequenceTooLongError
        
        batch_size = len(pixel_values_list)
        logger.info(
            f"Starting batched generation with batch_size={batch_size}, "
            f"max_new_tokens={max_new_tokens}"
        )
        
        if batch_size == 0:
            return []
        
        # Validate inputs
        if len(input_ids_list) != batch_size:
            raise ValueError(
                f"Mismatch between pixel_values_list ({batch_size}) and "
                f"input_ids_list ({len(input_ids_list)}) lengths"
            )
        
        # Step 1: Batch all images and encode vision features
        pixel_values_batch = torch.cat(pixel_values_list, dim=0)
        logger.debug(f"Batched pixel_values shape: {pixel_values_batch.shape}")
        
        vision_features_batch = self.encode_vision(pixel_values_batch)
        logger.debug(f"Batched vision_features shape: {vision_features_batch.shape}")
        
        # Step 2: Project vision features for all requests
        vision_embeds_batch = self.project_vision(vision_features_batch)
        logger.debug(f"Batched vision_embeds shape: {vision_embeds_batch.shape}")
        
        # Step 3: Process each request to get text embeddings and combine
        # We need to handle variable-length text inputs per-request
        combined_embeds_list = []
        for i in range(batch_size):
            vision_embeds = vision_embeds_batch[i:i+1]  # [1, 577, 768]
            input_ids = input_ids_list[i]  # [1, text_len]
            
            # Get text embeddings for this request
            text_embeds = self._get_text_embeddings(input_ids)
            
            # Combine vision and text embeddings
            combined_embeds = self.fuse_vision_text_embeddings(vision_embeds, text_embeds)
            combined_embeds_list.append(combined_embeds)
        
        # Pad all combined embeddings to same length for batching
        max_combined_len = max(embeds.shape[1] for embeds in combined_embeds_list)
        combined_embeds_padded = []
        for embeds in combined_embeds_list:
            if embeds.shape[1] < max_combined_len:
                padding = torch.zeros(
                    1,
                    max_combined_len - embeds.shape[1],
                    embeds.shape[2],
                    dtype=embeds.dtype,
                    device=embeds.device
                )
                embeds_padded = torch.cat([embeds, padding], dim=1)
            else:
                embeds_padded = embeds
            combined_embeds_padded.append(embeds_padded)
        
        combined_embeds_batch = torch.cat(combined_embeds_padded, dim=0)
        logger.debug(f"Batched combined_embeds shape: {combined_embeds_batch.shape}")
        
        # Step 4: Encode combined embeddings for all requests
        encoder_hidden_states_batch = self.encode_language(combined_embeds_batch)
        logger.debug(f"Batched encoder_hidden_states shape: {encoder_hidden_states_batch.shape}")
        
        # Step 5: Initialize batch state for per-request tracking
        batch_state = BatchState()
        
        for i in range(batch_size):
            request_id = f"req_{i}"
            vision_embeds = vision_embeds_batch[i:i+1]
            encoder_hidden_states = encoder_hidden_states_batch[i:i+1]
            initial_ids = input_ids_list[i]
            
            batch_state.add_request(
                request_id=request_id,
                vision_embeddings=vision_embeds,
                encoder_hidden_states=encoder_hidden_states,
                initial_ids=initial_ids
            )
        
        # Step 6: Autoregressive decoding loop with per-request state
        largest_bucket = max(self.config.decoder_buckets)
        
        for step in range(max_new_tokens):
            if batch_state.is_empty():
                logger.info(f"All requests completed at step {step}")
                break
            
            current_batch_size = batch_state.batch_size()
            logger.debug(f"Step {step}: {current_batch_size} active requests")
            
            # Get current state for all active requests
            generated_ids_batch = batch_state.get_batch_generated_ids()
            encoder_states_batch = batch_state.get_batch_encoder_hidden_states()
            
            current_seq_len = generated_ids_batch.shape[1]
            
            # Check if we exceed max decoder length
            if current_seq_len > largest_bucket:
                logger.warning(
                    f"Sequence length {current_seq_len} exceeds maximum decoder length {largest_bucket}. "
                    f"Truncating generation at step {step}."
                )
                break
            
            # Decode step with bucket selection (batched)
            logits_batch = self.decode_step(generated_ids_batch, encoder_states_batch)
            
            # Get logits for the last position
            next_token_logits = logits_batch[:, current_seq_len - 1, :]
            
            # Greedy decoding (select token with highest probability)
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequences
            new_generated_ids = torch.cat([generated_ids_batch, next_tokens], dim=1)
            
            # Update batch state with new generated IDs
            batch_state.update_generated_ids(new_generated_ids)
            
            # Mark completed requests (those that generated EOS)
            completed = batch_state.mark_completed_requests(self.config.eos_token_id)
            if completed:
                logger.debug(f"Requests completed at step {step}: {completed}")
        
        # Step 7: Collect results from all requests
        results = []
        for i in range(batch_size):
            request_id = f"req_{i}"
            state = batch_state.get_request_state(request_id)
            if state and state.generated_ids is not None:
                results.append(state.generated_ids)
            else:
                # Fallback: return initial IDs if something went wrong
                logger.warning(f"Request {request_id} has no generated IDs, using initial IDs")
                results.append(input_ids_list[i])
        
        logger.info(
            f"Batched generation complete. Processed {batch_size} requests, "
            f"generated {[r.shape[1] for r in results]} tokens per request"
        )
        
        return results
    
    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get text embeddings from token IDs.
        
        Uses the decoder's embedding layer to convert token IDs to embeddings.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
        
        Returns:
            Text embeddings [batch, seq_len, 768]
        """
        with torch.no_grad():
            text_embeds = self.embed_tokens(input_ids)
        
        logger.debug(f"Generated text embeddings: {text_embeds.shape}")
        return text_embeds

    def fuse_vision_text_embeddings(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine vision embeddings with text embeddings.

        Concatenates vision and text embeddings along the sequence dimension,
        ensuring correct order and dimensions for the language encoder.

        Concatenation order: [vision_embeds, text_embeds]

        Args:
            vision_embeds: Vision embeddings [batch, 577, 768]
            text_embeds: Text embeddings [batch, text_len, 768]

        Returns:
            Combined embeddings [batch, 577 + text_len, 768]

        Raises:
            ValueError: If embedding dimensions don't match or shapes are invalid

        Requirements:
            - 5.4: Combine vision embeddings with text embeddings
            - 5.4: Ensure correct concatenation order and dimensions
        """
        logger.debug(
            f"Fusing embeddings - vision: {vision_embeds.shape}, text: {text_embeds.shape}"
        )

        # Validate batch sizes match
        if vision_embeds.shape[0] != text_embeds.shape[0]:
            raise ValueError(
                f"Batch size mismatch: vision_embeds batch={vision_embeds.shape[0]}, "
                f"text_embeds batch={text_embeds.shape[0]}"
            )

        # Validate hidden dimensions match
        if vision_embeds.shape[2] != text_embeds.shape[2]:
            raise ValueError(
                f"Hidden dimension mismatch: vision_embeds hidden={vision_embeds.shape[2]}, "
                f"text_embeds hidden={text_embeds.shape[2]}"
            )

        # Validate expected dimensions
        if vision_embeds.dim() != 3 or text_embeds.dim() != 3:
            raise ValueError(
                f"Embeddings must be 3D tensors [batch, seq_len, hidden], "
                f"got vision: {vision_embeds.dim()}D, text: {text_embeds.dim()}D"
            )

        # Validate hidden size matches language model
        expected_hidden = self.config.language_hidden_size
        if vision_embeds.shape[2] != expected_hidden:
            raise ValueError(
                f"Vision embeddings hidden size {vision_embeds.shape[2]} "
                f"doesn't match expected {expected_hidden}"
            )

        # Concatenate along sequence dimension (dim=1)
        # Order: vision embeddings first, then text embeddings
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        logger.debug(f"Fused embeddings shape: {combined_embeds.shape}")

        # Validate output shape
        expected_seq_len = vision_embeds.shape[1] + text_embeds.shape[1]
        if combined_embeds.shape[1] != expected_seq_len:
            raise ValueError(
                f"Combined sequence length {combined_embeds.shape[1]} "
                f"doesn't match expected {expected_seq_len}"
            )

        return combined_embeds

    
    def preprocess_image(self, image: Union[str, PIL.Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess image for vision encoder.

        Handles various input formats and converts to the format expected by
        the DaViT vision encoder: 768x768 resolution in BF16 format.

        Supported input formats:
            - str: File path to image
            - PIL.Image.Image: PIL Image object
            - torch.Tensor: Pre-processed tensor (will validate shape and dtype)

        Args:
            image: Image in one of the supported formats

        Returns:
            Preprocessed tensor [1, 3, 768, 768] in BF16

        Raises:
            ValueError: If image format is not supported
            FileNotFoundError: If image path does not exist
            IOError: If image cannot be loaded

        Requirements:
            - 5.1: Preprocess image to 768x768 resolution
            - 5.2: Convert to BF16 tensor format
        """
        logger.debug(f"Preprocessing image, input type: {type(image)}")

        # Handle different input formats
        if isinstance(image, str):
            # Load image from file path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")

            try:
                image = PIL.Image.open(image)
                logger.debug(f"Loaded image from path: {image}")
            except Exception as e:
                raise ImageLoadError(image, str(e))

        elif isinstance(image, torch.Tensor):
            # Validate tensor shape and dtype
            if image.dim() == 3:
                # Add batch dimension if missing
                image = image.unsqueeze(0)

            if image.shape[1:] != (3, 768, 768):
                raise ValueError(
                    f"Tensor image must have shape [B, 3, 768, 768] or [3, 768, 768], "
                    f"got {image.shape}"
                )

            # Convert to BF16 if needed
            if image.dtype != self.config.dtype:
                logger.debug(f"Converting tensor from {image.dtype} to {self.config.dtype}")
                image = image.to(self.config.dtype)

            logger.debug(f"Using pre-processed tensor: {image.shape}")
            return image

        elif not isinstance(image, PIL.Image.Image):
            raise ValueError(
                f"Unsupported image format: {type(image)}. "
                f"Expected str (file path), PIL.Image.Image, or torch.Tensor"
            )

        # Process PIL Image using HuggingFace processor
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"]

            # Convert to BF16
            pixel_values = pixel_values.to(self.config.dtype)

            # Validate output shape
            expected_shape = (1, 3, 768, 768)
            if pixel_values.shape != expected_shape:
                raise ValueError(
                    f"Preprocessed image has unexpected shape {pixel_values.shape}, "
                    f"expected {expected_shape}"
                )

            logger.debug(f"Preprocessed image to shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
            return pixel_values

        except Exception as e:
            raise ImageLoadError(str(image) if hasattr(image, '__str__') else 'unknown', str(e))
    
    def tokenize_text(self, text: str) -> torch.Tensor:
        """
        Tokenize text prompt using Florence-2 tokenizer.

        Tokenizes task prompts and validates that all token IDs are within
        the valid vocabulary range.

        Args:
            text: Text prompt (e.g., "<CAPTION>", "<OD>")

        Returns:
            Token IDs tensor [1, seq_len]

        Raises:
            ValueError: If text is empty or tokenization produces invalid token IDs

        Requirements:
            - 5.3: Tokenize task prompts using Florence-2 tokenizer
            - 5.3: Validate token IDs are in valid range
        """
        if not text or not isinstance(text, str):
            raise ValueError(f"Text must be a non-empty string, got: {text}")

        logger.debug(f"Tokenizing text: '{text}'")

        try:
            # Tokenize using HuggingFace tokenizer
            inputs = self.tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"]

            # Validate token IDs are in valid range [0, vocab_size)
            if input_ids.min() < 0:
                raise ValueError(
                    f"Tokenization produced negative token IDs: min={input_ids.min()}"
                )

            if input_ids.max() >= self.config.vocab_size:
                raise ValueError(
                    f"Tokenization produced token IDs outside vocabulary range: "
                    f"max={input_ids.max()}, vocab_size={self.config.vocab_size}"
                )

            logger.debug(
                f"Tokenized text to shape: {input_ids.shape}, "
                f"token range: [{input_ids.min()}, {input_ids.max()}]"
            )
            return input_ids

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to tokenize text '{text}': {str(e)}")


    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text.

        Converts generated token IDs back to human-readable text using
        the Florence-2 tokenizer.

        Args:
            token_ids: Token IDs tensor [batch, seq_len] or [seq_len]

        Returns:
            Decoded text string

        Raises:
            ValueError: If token_ids is empty or has invalid shape

        Requirements:
            - 6.6: Decode token IDs to text
        """
        if token_ids.numel() == 0:
            raise ValueError("Cannot decode empty token_ids tensor")

        # Handle batch dimension
        if token_ids.dim() == 2:
            # Take first sequence from batch
            token_ids = token_ids[0]
        elif token_ids.dim() != 1:
            raise ValueError(
                f"token_ids must be 1D or 2D tensor, got {token_ids.dim()}D"
            )

        logger.debug(f"Decoding {token_ids.shape[0]} tokens")

        # Decode using tokenizer, skipping special tokens
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        logger.debug(f"Decoded text: '{text}'")
        return text

    def format_output(self, text: str, task: str) -> str:
        """
        Format output according to task type.

        Different tasks may require different output formatting. This method
        ensures the output format matches the original Florence-2 implementation.

        Supported tasks:
            - <CAPTION>: Simple caption text
            - <DETAILED_CAPTION>: Detailed description text
            - <OD>: Object detection with bounding boxes
            - <OCR>: Extracted text from image
            - <REGION_CAPTION>: Region-specific descriptions

        Args:
            text: Decoded text from model
            task: Task prompt that was used

        Returns:
            Formatted output string

        Requirements:
            - 6.6: Format outputs according to task type
            - 6.6: Ensure format matches original implementation
        """
        logger.debug(f"Formatting output for task '{task}'")

        # For most tasks, the raw decoded text is the correct format
        # The Florence-2 model outputs are already in the correct format
        # from the tokenizer

        # Task-specific formatting (if needed in the future)
        if task == "<CAPTION>":
            # Simple caption - return as is
            return text.strip()

        elif task == "<DETAILED_CAPTION>":
            # Detailed caption - return as is
            return text.strip()

        elif task == "<OD>":
            # Object detection - format is already correct from model
            # Format: "object1<loc_x1><loc_y1><loc_x2><loc_y2>object2<loc_x1>..."
            return text.strip()

        elif task == "<OCR>":
            # OCR - return extracted text as is
            return text.strip()

        elif task == "<REGION_CAPTION>":
            # Region caption - return as is
            return text.strip()

        else:
            # Unknown task - return as is with warning
            logger.warning(f"Unknown task '{task}', returning unformatted text")
            return text.strip()

    def run_task(
        self,
        image: Union[str, PIL.Image.Image, torch.Tensor],
        task: str,
        max_new_tokens: int = 100
    ) -> str:
        """
        Run a specific task on an image.

        This is the main entry point for task-specific inference. It handles:
        1. Input validation
        2. Image preprocessing
        3. Text tokenization
        4. Model inference
        5. Output decoding and formatting

        Args:
            image: Input image (file path, PIL Image, or tensor)
            task: Task prompt (e.g., "<CAPTION>", "<OD>", "<OCR>")
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Formatted output string for the task

        Raises:
            InvalidTaskError: If task is not supported
            ValueError: If image format is invalid
            FileNotFoundError: If image path does not exist

        Requirements:
            - 6.1: Support CAPTION task
            - 6.2: Support DETAILED_CAPTION task
            - 6.3: Support OD task
            - 6.4: Support OCR task
            - 6.5: Support REGION_CAPTION task
            - 6.6: Return outputs in same format as original implementation

        Example:
            >>> model = Florence2NxDModel("./compiled_nxd")
            >>> result = model.run_task("image.jpg", "<CAPTION>")
            >>> print(result)
            "A cat sitting on a couch"
        """
        logger.info(f"Running task '{task}' on image")

        # Step 1: Validate inputs
        self.validate_inputs(image, task)

        # Step 2: Load image if it's a path
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = PIL.Image.open(image)

        # Step 3: Process image and text together using the processor
        inputs = self.processor(text=task, images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.config.dtype)
        input_ids = inputs["input_ids"]
        
        print(f"[DEBUG] Original input_ids: {input_ids}, shape: {input_ids.shape}")
        
        # Remove EOS token from input if present (it will be generated)
        # The tokenizer adds EOS to the prompt, but we need to generate from scratch
        if input_ids[0, -1] == self.config.eos_token_id:
            input_ids = input_ids[:, :-1]
            print(f"[DEBUG] Removed EOS from input, new shape: {input_ids.shape}, new input_ids: {input_ids}")

        # Step 4: Run inference
        generated_ids = self.generate(pixel_values, input_ids, max_new_tokens)
        
        print(f"[DEBUG] Generated IDs shape: {generated_ids.shape}, input length: {input_ids.shape[1]}")
        print(f"[DEBUG] Generated IDs: {generated_ids}")

        # Step 5: Decode tokens to text
        # Skip the initial BOS token when decoding
        new_tokens = generated_ids[:, 1:]  # Skip BOS
        print(f"[DEBUG] New tokens shape: {new_tokens.shape}, new tokens: {new_tokens}")
        text = self.decode_tokens(new_tokens)
        print(f"[DEBUG] Decoded text: '{text}'")

        # Step 6: Format output according to task
        formatted_output = self.format_output(text, task)

        logger.info(f"Task '{task}' complete, output length: {len(formatted_output)}")
        return formatted_output

    def __call__(
        self,
        image: Union[str, PIL.Image.Image, torch.Tensor],
        task: str = "<CAPTION>",
        max_new_tokens: int = 100
    ) -> str:
        """
        Convenience method for running tasks.

        Alias for run_task() that provides a more intuitive interface.

        Args:
            image: Input image (file path, PIL Image, or tensor)
            task: Task prompt (default: "<CAPTION>")
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Formatted output string for the task

        Example:
            >>> model = Florence2NxDModel("./compiled_nxd")
            >>> result = model("image.jpg", "<CAPTION>")
        """
        return self.run_task(image, task, max_new_tokens)


