"""
Custom error classes for Florence-2 NxD Inference.

This module defines custom exception classes for various error conditions
that can occur during Florence-2 model operations.

Requirements:
    - 12.1: Descriptive error for unsupported task prompts
    - 12.2: Descriptive error for image loading failures
    - 12.3: Descriptive error for missing model files
"""

from typing import List


class Florence2Error(Exception):
    """Base exception class for Florence-2 errors."""
    pass


class InvalidTaskError(Florence2Error, ValueError):
    """
    Raised when an unsupported task prompt is provided.
    
    Requirements:
        - 12.1: Return descriptive error message for invalid task prompts
    """
    
    def __init__(self, task: str, supported_tasks: List[str]):
        """
        Initialize InvalidTaskError.
        
        Args:
            task: The invalid task prompt that was provided
            supported_tasks: List of supported task prompts
        """
        self.task = task
        self.supported_tasks = supported_tasks
        
        tasks_str = ", ".join(f"'{t}'" for t in supported_tasks)
        super().__init__(
            f"Invalid task '{task}'. Supported tasks: {tasks_str}"
        )


class ImageLoadError(Florence2Error, IOError):
    """
    Raised when an image cannot be loaded or processed.
    
    Requirements:
        - 12.2: Return descriptive error indicating image path or format issue
    """
    
    def __init__(self, image_path: str, reason: str):
        """
        Initialize ImageLoadError.
        
        Args:
            image_path: Path to the image that failed to load
            reason: Description of why the image failed to load
        """
        self.image_path = image_path
        self.reason = reason
        
        super().__init__(
            f"Failed to load image '{image_path}': {reason}"
        )


class ModelLoadError(Florence2Error, RuntimeError):
    """
    Raised when model loading fails.
    
    Requirements:
        - 12.3: Return error listing missing files
    """
    
    def __init__(self, model_dir: str, missing_files: List[str]):
        """
        Initialize ModelLoadError.
        
        Args:
            model_dir: Directory where models were expected
            missing_files: List of missing file names
        """
        self.model_dir = model_dir
        self.missing_files = missing_files
        
        files_str = '\n  - '.join(missing_files)
        super().__init__(
            f"Failed to load model from '{model_dir}'.\n"
            f"Missing files:\n  - {files_str}\n"
            f"Run compilation script to generate models."
        )


class HardwareCompatibilityError(Florence2Error, RuntimeError):
    """
    Raised when model is incompatible with available hardware.
    
    Requirements:
        - 12.4: Include hardware diagnostics in error messages
    """
    
    def __init__(self, required_cores: int, available_cores: int, message: str = None):
        """
        Initialize HardwareCompatibilityError.
        
        Args:
            required_cores: Number of NeuronCores required by the model
            available_cores: Number of NeuronCores available
            message: Optional custom error message
        """
        self.required_cores = required_cores
        self.available_cores = available_cores
        
        if message:
            error_msg = message
        else:
            error_msg = (
                f"Model requires {required_cores} NeuronCores, "
                f"but only {available_cores} are available.\n"
                f"Check hardware status with: neuron-top\n"
                f"Verify NEURON_RT_VISIBLE_CORES environment variable"
            )
        
        super().__init__(error_msg)


class GenerationError(Florence2Error, RuntimeError):
    """Base class for generation-related errors."""
    pass


class SequenceTooLongError(GenerationError):
    """
    Raised when sequence exceeds maximum length.
    
    Requirements:
        - 12.5: Handle sequence length overflow gracefully
    """
    
    def __init__(self, length: int, max_length: int):
        """
        Initialize SequenceTooLongError.
        
        Args:
            length: Actual sequence length
            max_length: Maximum allowed sequence length
        """
        self.length = length
        self.max_length = max_length
        
        super().__init__(
            f"Sequence length {length} exceeds maximum {max_length}. "
            f"Generation will be truncated."
        )


class NumericalError(GenerationError):
    """
    Raised when numerical issues are detected.
    
    Requirements:
        - 12.6: Detect and report numerical errors
    """
    
    def __init__(self, operation: str, tensor_name: str):
        """
        Initialize NumericalError.
        
        Args:
            operation: Name of the operation where error occurred
            tensor_name: Name of the tensor containing NaN/Inf
        """
        self.operation = operation
        self.tensor_name = tensor_name
        
        super().__init__(
            f"Numerical error in {operation}: {tensor_name} contains NaN or Inf values"
        )


class NeuronCoreError(Florence2Error, RuntimeError):
    """
    Raised when NeuronCore operations fail.
    
    Requirements:
        - 12.4: Include hardware diagnostics in error messages
    """
    
    def __init__(self, core_id: str, error: Exception):
        """
        Initialize NeuronCoreError.
        
        Args:
            core_id: Identifier of the NeuronCore that failed
            error: The underlying exception
        """
        self.core_id = core_id
        self.error = error
        
        super().__init__(
            f"NeuronCore {core_id} error: {error}\n"
            f"Check hardware status with: neuron-top\n"
            f"Verify NEURON_RT_VISIBLE_CORES environment variable"
        )

