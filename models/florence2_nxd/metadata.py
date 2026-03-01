"""
Metadata management for compiled Florence-2 NxD Inference models.

This module defines the CompiledModelMetadata dataclass for storing and loading
compilation metadata alongside compiled model artifacts.

Requirements:
    - 10.2: Define metadata fields and implement save/load methods
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Florence2NxDConfig


@dataclass
class CompiledModelMetadata:
    """
    Metadata for compiled Florence-2 NxD Inference models.
    
    This dataclass stores all information needed to load and validate
    compiled models, including version information, configuration,
    and file listings.
    
    Attributes:
        model_name: HuggingFace model identifier
        compilation_date: ISO format timestamp of compilation
        nxd_version: neuronx-distributed version used for compilation
        neuronx_version: torch-neuronx version used for compilation
        config: Florence2NxDConfig instance with model configuration
        vision_stage_files: List of vision encoder stage file names
        projection_file: Projection layer file name
        encoder_file: Language encoder file name
        decoder_files: Dictionary mapping bucket sizes to decoder file names
        expected_latency_ms: Expected latency per task in milliseconds
        expected_throughput_qps: Expected throughput in queries per second
    
    Requirements:
        - 10.2: Define metadata fields for model information
    """
    
    # Model information
    model_name: str = "microsoft/Florence-2-base"
    compilation_date: str = ""
    nxd_version: str = ""
    neuronx_version: str = ""
    torch_version: str = ""
    compiler_version: str = "0.1.0"
    
    # Configuration
    tp_degree: int = 1
    dtype: str = "bfloat16"
    max_encoder_length: int = 600
    decoder_buckets: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32, 64])
    num_vision_stages: int = 4
    
    # Compiled artifacts
    vision_stage_files: List[str] = field(default_factory=lambda: [
        "stage0.pt", "stage1.pt", "stage2.pt", "stage3.pt"
    ])
    projection_file: str = "projection.pt"
    encoder_file: str = "encoder.pt"
    decoder_files: Dict[int, str] = field(default_factory=lambda: {
        1: "decoder_1.pt",
        4: "decoder_4.pt",
        8: "decoder_8.pt",
        16: "decoder_16.pt",
        32: "decoder_32.pt",
        64: "decoder_64.pt",
    })
    
    # Performance characteristics
    expected_latency_ms: Dict[str, float] = field(default_factory=dict)
    expected_throughput_qps: float = 4.0
    
    # Shape information
    vision_input_shape: List[int] = field(default_factory=lambda: [1, 3, 768, 768])
    vision_output_shape: List[int] = field(default_factory=lambda: [1, 576, 1024])
    encoder_input_shape: List[int] = field(default_factory=lambda: [1, 600, 768])
    
    def save(self, path: str) -> None:
        """
        Save metadata to JSON file.
        
        This method serializes the metadata to JSON format and saves it
        to the specified path. The compilation_date is automatically set
        to the current timestamp.
        
        Args:
            path: File path where metadata should be saved
        
        Requirements:
            - 10.2: Implement save method for JSON serialization
        """
        # Set compilation date to current time
        self.compilation_date = datetime.now().isoformat()
        
        # Convert to dictionary
        data = asdict(self)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Save to JSON with pretty formatting
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str) -> 'CompiledModelMetadata':
        """
        Load metadata from JSON file.
        
        This method deserializes metadata from a JSON file and creates
        a CompiledModelMetadata instance. It handles both old format
        (with nested config) and new format (flattened fields).
        
        Args:
            path: File path to load metadata from
        
        Returns:
            CompiledModelMetadata instance with loaded data
        
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If metadata file is not valid JSON
        
        Requirements:
            - 10.2: Implement load method for JSON deserialization
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle old format with nested config
        if 'config' in data:
            config = data.pop('config')
            # Merge config fields into top level, but don't override existing fields
            for key, value in config.items():
                if key not in data:
                    data[key] = value
        
        # Filter out any keys that aren't in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    @classmethod
    def from_config(cls, config: Any, model_name: str) -> 'CompiledModelMetadata':
        """
        Create metadata from a Florence2NxDConfig instance.
        
        This is a convenience method for creating metadata during compilation.
        
        Args:
            config: Florence2NxDConfig instance (typed as Any to avoid import)
            model_name: HuggingFace model identifier
        
        Returns:
            CompiledModelMetadata instance
        """
        # Get version information
        try:
            import torch_neuronx
            neuronx_version = torch_neuronx.__version__
        except (ImportError, AttributeError):
            neuronx_version = "unknown"
        
        try:
            import neuronx_distributed
            nxd_version = neuronx_distributed.__version__
        except (ImportError, AttributeError):
            nxd_version = "unknown"
        
        try:
            import torch
            torch_version = torch.__version__
        except ImportError:
            torch_version = "unknown"
        
        return cls(
            model_name=model_name,
            nxd_version=nxd_version,
            neuronx_version=neuronx_version,
            torch_version=torch_version,
            tp_degree=config.tp_degree,
            dtype="bfloat16",
            max_encoder_length=config.max_encoder_length,
            decoder_buckets=config.decoder_buckets.copy(),
            num_vision_stages=config.num_vision_stages,
            vision_stage_files=[f"stage{i}.pt" for i in range(config.num_vision_stages)],
            decoder_files={b: f"decoder_{b}.pt" for b in config.decoder_buckets},
            vision_input_shape=list(config.vision_stage_shapes[0]),
            vision_output_shape=list(config.vision_output_shape),
            encoder_input_shape=[1, config.max_encoder_length, config.language_hidden_size],
        )
    
    def get_all_required_files(self) -> List[str]:
        """
        Get list of all required model files.
        
        Returns:
            List of all file names that should exist for this model
        
        Requirements:
            - 10.3: Support checking for all required files
        """
        files = []
        files.extend(self.vision_stage_files)
        files.append(self.projection_file)
        files.append(self.encoder_file)
        files.extend(self.decoder_files.values())
        return files
    
    def validate_files_exist(self, model_dir: str) -> tuple[bool, List[str]]:
        """
        Validate that all required files exist in the model directory.
        
        Args:
            model_dir: Directory containing compiled models
        
        Returns:
            Tuple of (all_exist, missing_files) where:
            - all_exist: True if all files exist, False otherwise
            - missing_files: List of missing file names
        
        Requirements:
            - 10.3: Check for all required files before loading
        """
        required_files = self.get_all_required_files()
        missing_files = []
        
        for filename in required_files:
            filepath = os.path.join(model_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        return len(missing_files) == 0, missing_files
    
    def validate_hardware_compatibility(self, available_cores: int) -> tuple[bool, Optional[str]]:
        """
        Validate that the model is compatible with available hardware.
        
        Args:
            available_cores: Number of available NeuronCores
        
        Returns:
            Tuple of (is_compatible, error_message) where:
            - is_compatible: True if compatible, False otherwise
            - error_message: Description of incompatibility, or None if compatible
        
        Requirements:
            - 10.4: Validate metadata compatibility with hardware
        """
        required_cores = self.tp_degree
        
        if available_cores < required_cores:
            return False, (
                f"Model requires {required_cores} NeuronCores (TP degree={self.tp_degree}), "
                f"but only {available_cores} are available"
            )
        
        return True, None
    
    def to_dict(self) -> Dict:
        """
        Convert metadata to dictionary.
        
        Returns:
            Dictionary representation of metadata
        """
        return asdict(self)
