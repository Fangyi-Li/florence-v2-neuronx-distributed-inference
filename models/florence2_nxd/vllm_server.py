"""
vLLM API server for Florence-2 multimodal inference.

This module implements a FastAPI-based server that provides OpenAI-compatible
API endpoints for Florence-2 inference with vLLM integration.

Requirements:
    - 11.1: Integrate with vLLM's API server for HTTP request handling
    - 11.2: Load Florence-2 models on server start and initialize inference engine
    - 11.4: Support OpenAI-compatible request and response formats
    - 11.5: Return generated text in API response
    - 11.6: Handle concurrent requests through vLLM's request scheduling
"""

import asyncio
import logging
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from .vllm_plugin import Florence2VLLMPlugin
from .vllm_server_config import Florence2ServerConfig
from .openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse
)
from .request_scheduler import RequestScheduler
from .errors import (
    InvalidTaskError,
    ImageLoadError,
    ModelLoadError,
    GenerationError
)

logger = logging.getLogger(__name__)


class Florence2Server:
    """
    vLLM API server for Florence-2 multimodal inference.
    
    This server provides OpenAI-compatible API endpoints for Florence-2
    inference, with support for multimodal inputs (image + text) and
    concurrent request handling.
    
    Attributes:
        config: Server configuration
        plugin: Florence2VLLMPlugin instance for inference
        app: FastAPI application instance
        request_semaphore: Semaphore for limiting concurrent requests
    
    Requirements:
        - 11.1: Integrate with vLLM's API server for HTTP request handling
        - 11.2: Load Florence-2 models on server start
    
    Example:
        >>> config = Florence2ServerConfig(
        ...     model_dir="./compiled_nxd",
        ...     tp_degree=1,
        ...     port=8000
        ... )
        >>> server = Florence2Server(config)
        >>> server.run()
    """
    
    def __init__(self, config: Florence2ServerConfig):
        """
        Initialize Florence-2 vLLM server.
        
        Args:
            config: Server configuration
        
        Requirements:
            - 11.2: Initialize inference engine
            - 11.6: Initialize request scheduler
        """
        self.config = config
        self.plugin: Optional[Florence2VLLMPlugin] = None
        self.request_semaphore: Optional[asyncio.Semaphore] = None
        self.request_scheduler: Optional[RequestScheduler] = None
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Initializing Florence2Server")
        logger.info(f"Configuration: {config.to_dict()}")
        
        # Validate configuration
        config.validate()
        
        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title="Florence-2 vLLM API",
            description="OpenAI-compatible API for Florence-2 multimodal inference",
            version="1.0.0",
            lifespan=self._lifespan
        )
        
        # Register routes
        self._register_routes()
        
        logger.info("Florence2Server initialized")
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """
        Lifespan context manager for server startup and shutdown.
        
        This method handles:
        - Loading Florence-2 models on server start
        - Initializing the inference engine
        - Cleaning up resources on shutdown
        
        Requirements:
            - 11.2: Load Florence-2 models on server start
        """
        # Startup
        logger.info("Starting Florence-2 vLLM server...")
        
        try:
            # Load Florence-2 models
            logger.info(
                f"Loading Florence-2 models from {self.config.model_dir} "
                f"with TP degree {self.config.tp_degree}"
            )
            
            self.plugin = Florence2VLLMPlugin(
                model_dir=self.config.model_dir,
                tp_degree=self.config.tp_degree
            )
            
            logger.info("Florence-2 models loaded successfully")
            
            # Initialize request semaphore for concurrency control
            self.request_semaphore = asyncio.Semaphore(
                self.config.max_concurrent_requests
            )
            
            # Initialize request scheduler
            self.request_scheduler = RequestScheduler(
                max_concurrent=self.config.max_concurrent_requests
            )
            
            logger.info(
                f"Server ready to accept requests "
                f"(max concurrent: {self.config.max_concurrent_requests})"
            )
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
        
        yield
        
        # Shutdown
        logger.info("Shutting down Florence-2 vLLM server...")
        self.plugin = None
        self.request_semaphore = None
        self.request_scheduler = None
        logger.info("Server shutdown complete")
    
    def _register_routes(self) -> None:
        """
        Register API routes.
        
        This method sets up all API endpoints including:
        - Health check endpoint
        - Chat completions endpoint (OpenAI-compatible)
        - Model information endpoint
        
        Requirements:
            - 11.1: Integrate with vLLM's API server for HTTP request handling
            - 11.4: Support OpenAI-compatible request and response formats
        """
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.plugin is not None
            }
        
        @self.app.get("/v1/models")
        async def list_models():
            """List available models (OpenAI-compatible)."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": "florence-2",
                        "object": "model",
                        "created": 1234567890,
                        "owned_by": "microsoft",
                        "permission": [],
                        "root": "florence-2",
                        "parent": None
                    }
                ]
            }
        
        @self.app.get("/stats")
        async def get_statistics():
            """
            Get server statistics.
            
            Returns statistics about request processing including:
            - Total requests processed
            - Success/failure rates
            - Active requests
            - Queue size
            
            Requirements:
                - 11.6: Provide visibility into concurrent request handling
            """
            if self.request_scheduler is None:
                return {"error": "Scheduler not initialized"}
            
            return self.request_scheduler.get_statistics()
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            """
            Chat completions endpoint (OpenAI-compatible).
            
            This endpoint handles multimodal requests with image and text inputs,
            processes them through Florence-2, and returns generated text in
            OpenAI-compatible format.
            
            Requirements:
                - 11.4: Support OpenAI-compatible request and response formats
                - 11.5: Return generated text in API response
                - 11.6: Handle concurrent requests
            """
            return await self._handle_chat_completion(request)
        
        logger.info("API routes registered")
    
    async def _handle_chat_completion(self, request: Request) -> JSONResponse:
        """
        Handle chat completion request.
        
        This method:
        1. Parses the OpenAI-compatible request
        2. Extracts image and text inputs
        3. Processes through Florence-2
        4. Returns OpenAI-compatible response
        
        Args:
            request: FastAPI request object
        
        Returns:
            JSONResponse with completion or error
        
        Requirements:
            - 11.4: Parse multimodal requests and format responses
            - 11.5: Return generated text in API response
            - 11.6: Handle concurrent requests
        """
        # Acquire semaphore for concurrency control
        async with self.request_semaphore:
            try:
                # Parse request body
                request_data = await request.json()
                
                if self.config.log_requests:
                    logger.info(f"Received request: {request_data}")
                
                # Parse OpenAI request format
                chat_request = ChatCompletionRequest.from_dict(request_data)
                chat_request.validate()
                
                # Extract image and text
                image, text = chat_request.extract_image_and_text()
                
                if image is None:
                    raise ValueError(
                        "No image found in request. "
                        "Florence-2 requires an image input."
                    )
                
                logger.info(f"Processing request: task='{text}'")
                
                # Process multimodal input
                inputs = self.plugin.process_multimodal_input(
                    image=image,
                    text=text
                )
                
                # Execute model
                max_tokens = chat_request.max_tokens or self.config.max_new_tokens
                
                token_ids = self.plugin.execute_model(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids'],
                    max_new_tokens=max_tokens
                )
                
                # Decode tokens to text
                generated_text = self.plugin.model.processor.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True
                )
                
                logger.info(f"Generated text: {generated_text}")
                
                # Create OpenAI-compatible response
                request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                
                response = ChatCompletionResponse.create_from_text(
                    text=generated_text,
                    request_id=request_id,
                    model=chat_request.model,
                    prompt_tokens=inputs['input_ids'].shape[1],
                    completion_tokens=len(token_ids)
                )
                
                response_dict = response.to_dict()
                
                if self.config.log_responses:
                    logger.info(f"Response: {response_dict}")
                
                return JSONResponse(content=response_dict)
            
            except InvalidTaskError as e:
                logger.warning(f"Invalid task error: {e}")
                error_response = ErrorResponse.create(
                    message=str(e),
                    type="invalid_request_error",
                    code="invalid_task"
                )
                return JSONResponse(
                    status_code=400,
                    content=error_response.to_dict()
                )
            
            except ImageLoadError as e:
                logger.warning(f"Image load error: {e}")
                error_response = ErrorResponse.create(
                    message=str(e),
                    type="invalid_request_error",
                    code="image_load_error"
                )
                return JSONResponse(
                    status_code=400,
                    content=error_response.to_dict()
                )
            
            except ValueError as e:
                logger.warning(f"Validation error: {e}")
                error_response = ErrorResponse.create(
                    message=str(e),
                    type="invalid_request_error"
                )
                return JSONResponse(
                    status_code=400,
                    content=error_response.to_dict()
                )
            
            except GenerationError as e:
                logger.error(f"Generation error: {e}")
                error_response = ErrorResponse.create(
                    message=f"Generation failed: {str(e)}",
                    type="server_error",
                    code="generation_error"
                )
                return JSONResponse(
                    status_code=500,
                    content=error_response.to_dict()
                )
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                error_response = ErrorResponse.create(
                    message=f"Internal server error: {str(e)}",
                    type="server_error"
                )
                return JSONResponse(
                    status_code=500,
                    content=error_response.to_dict()
                )
    
    def run(self) -> None:
        """
        Run the server.
        
        This method starts the FastAPI server using uvicorn.
        
        Requirements:
            - 11.1: Integrate with vLLM's API server for HTTP request handling
        
        Example:
            >>> config = Florence2ServerConfig(port=8000)
            >>> server = Florence2Server(config)
            >>> server.run()  # Starts server on port 8000
        """
        logger.info(
            f"Starting server on {self.config.host}:{self.config.port}"
        )
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level.lower()
        )


def create_server(
    model_dir: str = "./compiled_nxd",
    tp_degree: int = 1,
    port: int = 8000,
    **kwargs
) -> Florence2Server:
    """
    Create Florence-2 vLLM server with configuration.
    
    This is a convenience function for creating a server with
    custom configuration.
    
    Args:
        model_dir: Directory containing compiled models
        tp_degree: Tensor parallelism degree
        port: Server port number
        **kwargs: Additional configuration parameters
    
    Returns:
        Florence2Server instance
    
    Example:
        >>> server = create_server(
        ...     model_dir="./my_models",
        ...     tp_degree=2,
        ...     port=8080,
        ...     max_concurrent_requests=20
        ... )
        >>> server.run()
    """
    config = Florence2ServerConfig(
        model_dir=model_dir,
        tp_degree=tp_degree,
        port=port,
        **kwargs
    )
    
    return Florence2Server(config)


def main():
    """
    Main entry point for running the server from command line.
    
    Example:
        $ python -m models.florence2_nxd.vllm_server
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Florence-2 vLLM API Server"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./compiled_nxd",
        help="Directory containing compiled NxD Inference models"
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=1,
        choices=[1, 2, 4, 8],
        help="Tensor parallelism degree"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number"
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=10,
        help="Maximum number of concurrent requests"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Default maximum tokens to generate"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Create server configuration
    config = Florence2ServerConfig(
        model_dir=args.model_dir,
        tp_degree=args.tp_degree,
        host=args.host,
        port=args.port,
        max_concurrent_requests=args.max_concurrent_requests,
        max_new_tokens=args.max_new_tokens,
        log_level=args.log_level
    )
    
    # Create and run server
    server = Florence2Server(config)
    server.run()


if __name__ == "__main__":
    main()
