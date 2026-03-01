"""
OpenAI-compatible protocol for Florence-2 vLLM server.

This module implements OpenAI-compatible request and response formats
for Florence-2 multimodal inference, allowing clients to use standard
OpenAI API interfaces.

Requirements:
    - 11.4: Support OpenAI-compatible request and response formats
"""

import time
import base64
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import PIL.Image
import torch

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """
    OpenAI chat message format.
    
    Attributes:
        role: Message role (system, user, assistant)
        content: Message content (text or multimodal)
    """
    role: str
    content: Union[str, List[Dict[str, Any]]]
    
    def validate(self) -> None:
        """Validate message format."""
        valid_roles = ["system", "user", "assistant"]
        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid role '{self.role}'. Must be one of {valid_roles}"
            )


@dataclass
class ChatCompletionRequest:
    """
    OpenAI chat completion request format.
    
    This class represents an OpenAI-compatible chat completion request
    with support for multimodal inputs (image + text).
    
    Attributes:
        model: Model identifier
        messages: List of chat messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        n: Number of completions to generate
        stream: Whether to stream responses
        stop: Stop sequences
    
    Requirements:
        - 11.4: Parse multimodal requests (image + text)
    """
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    
    def validate(self) -> None:
        """
        Validate request format.
        
        Raises:
            ValueError: If request format is invalid
        """
        if not self.messages:
            raise ValueError("messages cannot be empty")
        
        for msg in self.messages:
            msg.validate()
        
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        
        if self.temperature is not None and not (0.0 < self.temperature <= 2.0):
            raise ValueError(
                f"temperature must be in (0, 2], got {self.temperature}"
            )
        
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        
        if self.n is not None and self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
    
    def extract_image_and_text(self) -> tuple[Optional[PIL.Image.Image], str]:
        """
        Extract image and text from multimodal message content.
        
        This method parses the OpenAI multimodal message format and extracts
        the image and text components for Florence-2 inference.
        
        Returns:
            Tuple of (image, text) where:
            - image: PIL Image object or None if no image
            - text: Task prompt text
        
        Raises:
            ValueError: If message format is invalid
        
        Requirements:
            - 11.4: Parse multimodal requests (image + text)
        """
        image = None
        text_parts = []
        
        # Find the last user message
        user_messages = [msg for msg in self.messages if msg.role == "user"]
        if not user_messages:
            raise ValueError("No user message found in request")
        
        last_user_message = user_messages[-1]
        
        # Handle string content (text only)
        if isinstance(last_user_message.content, str):
            return None, last_user_message.content
        
        # Handle list content (multimodal)
        if isinstance(last_user_message.content, list):
            for content_part in last_user_message.content:
                if not isinstance(content_part, dict):
                    continue
                
                content_type = content_part.get("type")
                
                if content_type == "text":
                    text_parts.append(content_part.get("text", ""))
                
                elif content_type == "image_url":
                    image_url = content_part.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = image_url
                    
                    # Parse image from URL or base64
                    image = self._parse_image_url(url)
        
        # Combine text parts
        text = " ".join(text_parts).strip()
        
        # Default to CAPTION if no text provided
        if not text:
            text = "<CAPTION>"
        
        return image, text
    
    def _parse_image_url(self, url: str) -> PIL.Image.Image:
        """
        Parse image from URL or base64 data.
        
        Args:
            url: Image URL or base64 data URI
        
        Returns:
            PIL Image object
        
        Raises:
            ValueError: If image format is invalid
        """
        # Handle base64 data URI
        if url.startswith("data:image/"):
            # Format: data:image/jpeg;base64,<base64_data>
            try:
                header, base64_data = url.split(",", 1)
                image_bytes = base64.b64decode(base64_data)
                image = PIL.Image.open(BytesIO(image_bytes))
                return image
            except Exception as e:
                raise ValueError(f"Failed to parse base64 image: {e}")
        
        # Handle file path
        elif Path(url).exists():
            try:
                image = PIL.Image.open(url)
                return image
            except Exception as e:
                raise ValueError(f"Failed to load image from path '{url}': {e}")
        
        # Handle HTTP URL (not implemented in this version)
        else:
            raise ValueError(
                f"Unsupported image URL format: {url}. "
                f"Supported formats: base64 data URI, local file path"
            )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatCompletionRequest':
        """
        Create request from dictionary.
        
        Args:
            data: Dictionary containing request data
        
        Returns:
            ChatCompletionRequest instance
        """
        # Parse messages
        messages = []
        for msg_data in data.get("messages", []):
            messages.append(
                ChatMessage(
                    role=msg_data.get("role"),
                    content=msg_data.get("content")
                )
            )
        
        return cls(
            model=data.get("model", "florence-2"),
            messages=messages,
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            n=data.get("n"),
            stream=data.get("stream"),
            stop=data.get("stop")
        )


@dataclass
class ChatCompletionChoice:
    """
    OpenAI chat completion choice.
    
    Attributes:
        index: Choice index
        message: Generated message
        finish_reason: Reason for completion (stop, length, etc.)
    """
    index: int
    message: ChatMessage
    finish_reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "message": {
                "role": self.message.role,
                "content": self.message.content
            },
            "finish_reason": self.finish_reason
        }


@dataclass
class UsageInfo:
    """
    Token usage information.
    
    Attributes:
        prompt_tokens: Number of tokens in prompt
        completion_tokens: Number of tokens in completion
        total_tokens: Total number of tokens
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class ChatCompletionResponse:
    """
    OpenAI chat completion response format.
    
    This class represents an OpenAI-compatible chat completion response
    with generated text and metadata.
    
    Attributes:
        id: Unique response identifier
        object: Object type (always "chat.completion")
        created: Unix timestamp
        model: Model identifier
        choices: List of completion choices
        usage: Token usage information
    
    Requirements:
        - 11.4: Format responses according to OpenAI spec
    """
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "florence-2"
    choices: List[ChatCompletionChoice] = field(default_factory=list)
    usage: Optional[UsageInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of response
        
        Requirements:
            - 11.4: Format responses according to OpenAI spec
        """
        result = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [choice.to_dict() for choice in self.choices]
        }
        
        if self.usage is not None:
            result["usage"] = self.usage.to_dict()
        
        return result
    
    @classmethod
    def create_from_text(
        cls,
        text: str,
        request_id: str,
        model: str = "florence-2",
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> 'ChatCompletionResponse':
        """
        Create response from generated text.
        
        This is a convenience method for creating a response from
        Florence-2's generated text output.
        
        Args:
            text: Generated text
            request_id: Unique request identifier
            model: Model identifier
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
        
        Returns:
            ChatCompletionResponse instance
        
        Example:
            >>> response = ChatCompletionResponse.create_from_text(
            ...     text="A photo of a cat",
            ...     request_id="req_123",
            ...     prompt_tokens=10,
            ...     completion_tokens=5
            ... )
            >>> response.to_dict()
        """
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=text),
            finish_reason="stop"
        )
        
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        return cls(
            id=request_id,
            model=model,
            choices=[choice],
            usage=usage
        )


@dataclass
class ErrorResponse:
    """
    OpenAI error response format.
    
    Attributes:
        error: Error details
    """
    error: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"error": self.error}
    
    @classmethod
    def create(
        cls,
        message: str,
        type: str = "invalid_request_error",
        code: Optional[str] = None
    ) -> 'ErrorResponse':
        """
        Create error response.
        
        Args:
            message: Error message
            type: Error type
            code: Error code
        
        Returns:
            ErrorResponse instance
        """
        error_dict = {
            "message": message,
            "type": type
        }
        
        if code is not None:
            error_dict["code"] = code
        
        return cls(error=error_dict)
