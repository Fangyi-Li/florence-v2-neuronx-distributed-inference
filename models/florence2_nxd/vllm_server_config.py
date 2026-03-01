"""
vLLM server configuration for Florence-2 NxD Inference.

This module provides configuration classes and utilities for setting up
a vLLM server with Florence-2 multimodal support.

Requirements:
    - 11.1: Integrate with vLLM's API server for HTTP request handling
    - 11.3: Accept image inputs through vLLM's multimodal API endpoints
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Florence2ServerConfig:
    """
    Configuration for Florence-2 vLLM server.
    
    This configuration class defines all parameters needed to run a vLLM
    server with Florence-2 multimodal support, including model paths,
    hardware settings, and API endpoint configuration.
    
    Attributes:
        model_dir: Directory containing compiled NxD Inference models
        tp_degree: Tensor parallelism degree (1, 2, 4, or 8)
        host: Server host address
        port: Server port number
        max_concurrent_requests: Maximum number of concurrent requests
        max_new_tokens: Default maximum tokens to generate
        enable_multimodal: Enable multimodal (image + text) endpoints
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Requirements:
        - 11.1: Configure Florence-2 as vLLM model
        - 11.3: Set up multimodal endpoint configuration
    """
    
    # Model configuration
    model_dir: str = "./compiled_nxd"
    tp_degree: int = 1
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 10
    
    # Generation configuration
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    
    # Multimodal configuration
    enable_multimodal: bool = True
    supported_image_formats: List[str] = field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp", "bmp"]
    )
    max_image_size_mb: float = 10.0
    
    # Logging configuration
    log_level: str = "INFO"
    log_requests: bool = True
    log_responses: bool = False  # Can be verbose
    
    # Performance configuration
    enable_continuous_batching: bool = False  # Not yet implemented
    batch_size: int = 1
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate model directory
        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise ValueError(f"Model directory does not exist: {self.model_dir}")
        
        # Validate TP degree
        if self.tp_degree not in [1, 2, 4, 8]:
            raise ValueError(
                f"tp_degree must be 1, 2, 4, or 8, got {self.tp_degree}"
            )
        
        # Validate server configuration
        if not (0 <= self.port <= 65535):
            raise ValueError(f"Invalid port number: {self.port}")
        
        if self.max_concurrent_requests < 1:
            raise ValueError(
                f"max_concurrent_requests must be >= 1, "
                f"got {self.max_concurrent_requests}"
            )
        
        # Validate generation parameters
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be >= 1, got {self.max_new_tokens}"
            )
        
        if not (0.0 < self.temperature <= 2.0):
            raise ValueError(
                f"temperature must be in (0, 2], got {self.temperature}"
            )
        
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(
                f"top_p must be in (0, 1], got {self.top_p}"
            )
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(
                f"log_level must be one of {valid_log_levels}, "
                f"got {self.log_level}"
            )
        
        logger.info("Server configuration validated successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "model_dir": self.model_dir,
            "tp_degree": self.tp_degree,
            "host": self.host,
            "port": self.port,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_multimodal": self.enable_multimodal,
            "supported_image_formats": self.supported_image_formats,
            "max_image_size_mb": self.max_image_size_mb,
            "log_level": self.log_level,
            "log_requests": self.log_requests,
            "log_responses": self.log_responses,
            "enable_continuous_batching": self.enable_continuous_batching,
            "batch_size": self.batch_size,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Florence2ServerConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        
        Returns:
            Florence2ServerConfig instance
        """
        return cls(**config_dict)


@dataclass
class MultimodalEndpointConfig:
    """
    Configuration for multimodal API endpoints.
    
    This class defines the configuration for Florence-2's multimodal
    endpoints that accept both image and text inputs.
    
    Attributes:
        endpoint_path: API endpoint path
        supported_tasks: List of supported Florence-2 tasks
        require_task_prompt: Whether task prompt is required
        default_task: Default task if none specified
    
    Requirements:
        - 11.3: Set up multimodal endpoint configuration
    """
    
    endpoint_path: str = "/v1/chat/completions"
    supported_tasks: List[str] = field(
        default_factory=lambda: [
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<OD>",
            "<OCR>",
            "<REGION_CAPTION>"
        ]
    )
    require_task_prompt: bool = True
    default_task: str = "<CAPTION>"
    
    # Image input configuration
    image_field_name: str = "image"
    accept_image_url: bool = True
    accept_image_base64: bool = True
    accept_image_file: bool = True
    
    def validate(self) -> None:
        """
        Validate endpoint configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.endpoint_path.startswith("/"):
            raise ValueError(
                f"endpoint_path must start with '/', got {self.endpoint_path}"
            )
        
        if not self.supported_tasks:
            raise ValueError("supported_tasks cannot be empty")
        
        if self.require_task_prompt and self.default_task not in self.supported_tasks:
            raise ValueError(
                f"default_task '{self.default_task}' not in supported_tasks"
            )
        
        logger.info("Multimodal endpoint configuration validated successfully")


def create_default_config(
    model_dir: str = "./compiled_nxd",
    tp_degree: int = 1,
    port: int = 8000
) -> Florence2ServerConfig:
    """
    Create default server configuration.
    
    This is a convenience function for creating a server configuration
    with sensible defaults for most use cases.
    
    Args:
        model_dir: Directory containing compiled models
        tp_degree: Tensor parallelism degree
        port: Server port number
    
    Returns:
        Florence2ServerConfig with default settings
    
    Example:
        >>> config = create_default_config(
        ...     model_dir="./my_models",
        ...     tp_degree=2,
        ...     port=8080
        ... )
        >>> config.validate()
    """
    config = Florence2ServerConfig(
        model_dir=model_dir,
        tp_degree=tp_degree,
        port=port
    )
    
    logger.info(
        f"Created default server configuration: "
        f"model_dir={model_dir}, tp_degree={tp_degree}, port={port}"
    )
    
    return config
