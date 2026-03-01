"""
Florence-2 NxD Inference Integration

This package provides Florence-2 vision-language model integration with
neuronx-distributed-inference (NxD Inference) for AWS Inferentia2.
"""

from .config import Florence2NxDConfig
from .metadata import CompiledModelMetadata
from .logging_config import (
    setup_logging,
    get_logger,
    init_package_logging,
    get_package_logger,
)
from .nxd_wrappers import (
    NxDVisionStage,
    NxDProjection,
    NxDEncoder,
    NxDDecoder,
)
from .compile import Florence2Compiler
from .model import Florence2NxDModel
from .vllm_plugin import Florence2VLLMPlugin
from .vllm_server_config import Florence2ServerConfig, MultimodalEndpointConfig
from .vllm_server import Florence2Server, create_server
from .request_scheduler import RequestScheduler, RequestStatus, ScheduledRequest
from .openai_protocol import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from .errors import (
    Florence2Error,
    InvalidTaskError,
    ImageLoadError,
    ModelLoadError,
    HardwareCompatibilityError,
    GenerationError,
    SequenceTooLongError,
    NumericalError,
    NeuronCoreError,
)
from .compat import (
    Florence2NeuronBF16Compat,
    Florence2NeuronBF16,
    create_compatible_model,
)
from .migration import (
    detect_model_format,
    get_model_info,
    validate_legacy_model,
    load_legacy_model,
    create_migration_metadata,
    compare_model_outputs,
    print_migration_guide,
)

__version__ = "0.1.0"
__all__ = [
    "Florence2NxDConfig",
    "CompiledModelMetadata",
    "setup_logging",
    "get_logger",
    "init_package_logging",
    "get_package_logger",
    "NxDVisionStage",
    "NxDProjection",
    "NxDEncoder",
    "NxDDecoder",
    "Florence2Compiler",
    "Florence2NxDModel",
    "Florence2VLLMPlugin",
    "Florence2ServerConfig",
    "MultimodalEndpointConfig",
    "Florence2Server",
    "create_server",
    "RequestScheduler",
    "RequestStatus",
    "ScheduledRequest",
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ErrorResponse",
    "Florence2Error",
    "InvalidTaskError",
    "ImageLoadError",
    "ModelLoadError",
    "HardwareCompatibilityError",
    "GenerationError",
    "SequenceTooLongError",
    "NumericalError",
    "NeuronCoreError",
    "Florence2NeuronBF16Compat",
    "Florence2NeuronBF16",
    "create_compatible_model",
    "detect_model_format",
    "get_model_info",
    "validate_legacy_model",
    "load_legacy_model",
    "create_migration_metadata",
    "compare_model_outputs",
    "print_migration_guide",
]

