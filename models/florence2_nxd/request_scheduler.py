"""
Request scheduler for Florence-2 vLLM server.

This module implements request scheduling and queue management for
handling concurrent requests efficiently.

Requirements:
    - 11.6: Handle multiple concurrent requests through vLLM's request scheduler
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Request status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledRequest:
    """
    Scheduled request with metadata.
    
    Attributes:
        request_id: Unique request identifier
        image: Input image
        text: Task prompt text
        max_new_tokens: Maximum tokens to generate
        status: Current request status
        created_at: Request creation timestamp
        started_at: Processing start timestamp
        completed_at: Processing completion timestamp
        result: Generated result (if completed)
        error: Error message (if failed)
    """
    request_id: str
    image: Any
    text: str
    max_new_tokens: int = 100
    status: RequestStatus = RequestStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def get_latency_ms(self) -> Optional[float]:
        """
        Get request latency in milliseconds.
        
        Returns:
            Latency in milliseconds, or None if not completed
        """
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at) * 1000
    
    def get_queue_time_ms(self) -> Optional[float]:
        """
        Get time spent in queue in milliseconds.
        
        Returns:
            Queue time in milliseconds, or None if not started
        """
        if self.started_at is None:
            return None
        return (self.started_at - self.created_at) * 1000


class RequestScheduler:
    """
    Request scheduler for concurrent request handling.
    
    This scheduler manages a queue of incoming requests and processes
    them concurrently up to a maximum limit. It provides:
    - Request queuing and prioritization
    - Concurrent execution with configurable limits
    - Request status tracking
    - Performance metrics
    
    Attributes:
        max_concurrent: Maximum number of concurrent requests
        request_queue: Queue of pending requests
        active_requests: Dictionary of currently processing requests
        completed_requests: Dictionary of completed requests (for tracking)
        semaphore: Semaphore for limiting concurrency
    
    Requirements:
        - 11.6: Integrate with vLLM's request scheduler
        - 11.6: Handle multiple concurrent requests
    
    Example:
        >>> scheduler = RequestScheduler(max_concurrent=10)
        >>> request_id = await scheduler.submit_request(
        ...     image=image,
        ...     text="<CAPTION>",
        ...     max_new_tokens=100
        ... )
        >>> result = await scheduler.get_result(request_id)
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize request scheduler.
        
        Args:
            max_concurrent: Maximum number of concurrent requests
        
        Requirements:
            - 11.6: Configure concurrent request limit
        """
        self.max_concurrent = max_concurrent
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.active_requests: Dict[str, ScheduledRequest] = {}
        self.completed_requests: Dict[str, ScheduledRequest] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Statistics
        self.total_requests = 0
        self.total_completed = 0
        self.total_failed = 0
        
        logger.info(
            f"RequestScheduler initialized with max_concurrent={max_concurrent}"
        )
    
    async def submit_request(
        self,
        image: Any,
        text: str,
        max_new_tokens: int = 100
    ) -> str:
        """
        Submit a request for processing.
        
        This method creates a scheduled request and adds it to the queue.
        The request will be processed when a worker becomes available.
        
        Args:
            image: Input image
            text: Task prompt text
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Unique request identifier
        
        Requirements:
            - 11.6: Handle multiple concurrent requests
        
        Example:
            >>> request_id = await scheduler.submit_request(
            ...     image=my_image,
            ...     text="<CAPTION>"
            ... )
        """
        # Create request
        request_id = f"req_{uuid.uuid4().hex[:16]}"
        
        request = ScheduledRequest(
            request_id=request_id,
            image=image,
            text=text,
            max_new_tokens=max_new_tokens
        )
        
        # Add to queue
        await self.request_queue.put(request)
        self.active_requests[request_id] = request
        self.total_requests += 1
        
        logger.debug(
            f"Request {request_id} submitted: text='{text}', "
            f"queue_size={self.request_queue.qsize()}"
        )
        
        return request_id
    
    async def process_request(
        self,
        request: ScheduledRequest,
        process_fn
    ) -> None:
        """
        Process a single request.
        
        This method executes the processing function for a request
        and updates the request status accordingly.
        
        Args:
            request: Scheduled request to process
            process_fn: Async function to process the request
        
        Requirements:
            - 11.6: Process requests concurrently
        """
        async with self.semaphore:
            try:
                # Update status
                request.status = RequestStatus.PROCESSING
                request.started_at = time.time()
                
                logger.debug(
                    f"Processing request {request.request_id}: "
                    f"queue_time={request.get_queue_time_ms():.1f}ms"
                )
                
                # Process request
                result = await process_fn(
                    image=request.image,
                    text=request.text,
                    max_new_tokens=request.max_new_tokens
                )
                
                # Update status
                request.status = RequestStatus.COMPLETED
                request.completed_at = time.time()
                request.result = result
                
                self.total_completed += 1
                
                logger.debug(
                    f"Request {request.request_id} completed: "
                    f"latency={request.get_latency_ms():.1f}ms"
                )
            
            except Exception as e:
                # Update status
                request.status = RequestStatus.FAILED
                request.completed_at = time.time()
                request.error = str(e)
                
                self.total_failed += 1
                
                logger.error(
                    f"Request {request.request_id} failed: {e}",
                    exc_info=True
                )
            
            finally:
                # Move to completed
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]
                self.completed_requests[request.request_id] = request
    
    async def process_batch(
        self,
        requests: List[ScheduledRequest],
        batch_process_fn
    ) -> None:
        """
        Process a batch of requests together.
        
        This method executes the batch processing function for multiple
        requests simultaneously, supporting continuous batching where
        requests can complete at different times.
        
        Args:
            requests: List of scheduled requests to process as a batch
            batch_process_fn: Async function to process the batch
        
        Requirements:
            - 8.2: Support dynamic batch composition
            - 8.3: Allow requests to complete at different times
            - 11.6: Process multiple requests concurrently
        """
        if not requests:
            return
        
        batch_size = len(requests)
        logger.debug(f"Processing batch of {batch_size} requests")
        
        # Acquire semaphore for all requests in batch
        # This ensures we don't exceed max_concurrent limit
        semaphores = []
        for _ in range(batch_size):
            await self.semaphore.acquire()
            semaphores.append(self.semaphore)
        
        try:
            # Update status for all requests
            for request in requests:
                request.status = RequestStatus.PROCESSING
                request.started_at = time.time()
                
                logger.debug(
                    f"Processing request {request.request_id} in batch: "
                    f"queue_time={request.get_queue_time_ms():.1f}ms"
                )
            
            # Process batch
            results = await batch_process_fn(
                images=[req.image for req in requests],
                texts=[req.text for req in requests],
                max_new_tokens=[req.max_new_tokens for req in requests]
            )
            
            # Update status for all requests
            for i, request in enumerate(requests):
                request.status = RequestStatus.COMPLETED
                request.completed_at = time.time()
                request.result = results[i] if i < len(results) else None
                
                self.total_completed += 1
                
                logger.debug(
                    f"Request {request.request_id} completed in batch: "
                    f"latency={request.get_latency_ms():.1f}ms"
                )
        
        except Exception as e:
            # Update status for all requests as failed
            for request in requests:
                request.status = RequestStatus.FAILED
                request.completed_at = time.time()
                request.error = str(e)
                
                self.total_failed += 1
                
                logger.error(
                    f"Request {request.request_id} failed in batch: {e}",
                    exc_info=True
                )
        
        finally:
            # Release semaphores
            for sem in semaphores:
                sem.release()
            
            # Move all requests to completed
            for request in requests:
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]
                self.completed_requests[request.request_id] = request
    
    async def get_result(
        self,
        request_id: str,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Get result for a request.
        
        This method waits for a request to complete and returns the result.
        
        Args:
            request_id: Request identifier
            timeout: Maximum time to wait in seconds (None = wait forever)
        
        Returns:
            Request result
        
        Raises:
            ValueError: If request not found
            TimeoutError: If timeout exceeded
            RuntimeError: If request failed
        
        Example:
            >>> result = await scheduler.get_result(request_id, timeout=30.0)
        """
        start_time = time.time()
        
        while True:
            # Check if completed
            if request_id in self.completed_requests:
                request = self.completed_requests[request_id]
                
                if request.status == RequestStatus.COMPLETED:
                    return request.result
                elif request.status == RequestStatus.FAILED:
                    raise RuntimeError(f"Request failed: {request.error}")
                elif request.status == RequestStatus.CANCELLED:
                    raise RuntimeError("Request was cancelled")
            
            # Check if still active
            elif request_id not in self.active_requests:
                raise ValueError(f"Request not found: {request_id}")
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Request {request_id} timed out after {timeout}s"
                    )
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    def get_request_status(self, request_id: str) -> Optional[RequestStatus]:
        """
        Get status of a request.
        
        Args:
            request_id: Request identifier
        
        Returns:
            Request status, or None if not found
        """
        if request_id in self.active_requests:
            return self.active_requests[request_id].status
        elif request_id in self.completed_requests:
            return self.completed_requests[request_id].status
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.
        
        Returns:
            Dictionary with statistics:
            - total_requests: Total number of requests submitted
            - total_completed: Total number of completed requests
            - total_failed: Total number of failed requests
            - active_requests: Number of currently active requests
            - queue_size: Number of requests in queue
            - success_rate: Percentage of successful requests
        """
        total_processed = self.total_completed + self.total_failed
        success_rate = (
            (self.total_completed / total_processed * 100)
            if total_processed > 0
            else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "active_requests": len(self.active_requests),
            "queue_size": self.request_queue.qsize(),
            "success_rate": success_rate
        }
    
    def clear_completed(self, max_age_seconds: float = 3600) -> int:
        """
        Clear old completed requests from memory.
        
        This method removes completed requests older than the specified
        age to prevent memory buildup.
        
        Args:
            max_age_seconds: Maximum age of completed requests to keep
        
        Returns:
            Number of requests cleared
        """
        current_time = time.time()
        to_remove = []
        
        for request_id, request in self.completed_requests.items():
            if request.completed_at is not None:
                age = current_time - request.completed_at
                if age > max_age_seconds:
                    to_remove.append(request_id)
        
        for request_id in to_remove:
            del self.completed_requests[request_id]
        
        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old completed requests")
        
        return len(to_remove)


class RequestBatcher:
    """
    Request batcher for batching multiple requests together.
    
    This class collects requests and batches them together for
    more efficient processing. This is useful for continuous batching
    scenarios where multiple requests can be processed simultaneously.
    
    Note: This is a placeholder for future continuous batching support.
    The current implementation processes requests individually.
    
    Requirements:
        - 11.6: Support for future continuous batching
    """
    
    def __init__(self, max_batch_size: int = 4, batch_timeout_ms: float = 100):
        """
        Initialize request batcher.
        
        Args:
            max_batch_size: Maximum number of requests in a batch
            batch_timeout_ms: Maximum time to wait for batch to fill
        """
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        logger.info(
            f"RequestBatcher initialized: max_batch_size={max_batch_size}, "
            f"batch_timeout_ms={batch_timeout_ms}"
        )
    
    async def collect_batch(
        self,
        request_queue: asyncio.Queue
    ) -> list[ScheduledRequest]:
        """
        Collect a batch of requests from the queue.
        
        This method waits for requests to arrive and collects them
        into a batch up to max_batch_size or until timeout.
        
        Args:
            request_queue: Queue to collect requests from
        
        Returns:
            List of requests in the batch
        """
        batch = []
        start_time = time.time()
        
        while len(batch) < self.max_batch_size:
            # Calculate remaining timeout
            elapsed_ms = (time.time() - start_time) * 1000
            remaining_ms = self.batch_timeout_ms - elapsed_ms
            
            if remaining_ms <= 0:
                break
            
            try:
                # Try to get a request with timeout
                request = await asyncio.wait_for(
                    request_queue.get(),
                    timeout=remaining_ms / 1000
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break
        
        return batch

