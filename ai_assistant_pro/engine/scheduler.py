"""
Continuous batching scheduler for optimal throughput

Implements dynamic batching strategy for concurrent request handling
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class RequestStatus(Enum):
    """Status of a generation request"""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class GenerationRequest:
    """Generation request with metadata"""

    request_id: int
    prompt_tokens: List[int]
    max_tokens: int
    temperature: float
    top_p: float
    status: RequestStatus
    generated_tokens: List[int]
    arrival_time: float
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def prompt_len(self) -> int:
        """Length of prompt"""
        return len(self.prompt_tokens)

    @property
    def total_len(self) -> int:
        """Total length including generated tokens"""
        return self.prompt_len + len(self.generated_tokens)

    @property
    def is_finished(self) -> bool:
        """Check if request is finished"""
        return (
            self.status == RequestStatus.FINISHED
            or len(self.generated_tokens) >= self.max_tokens
        )


class ContinuousBatchScheduler:
    """
    Continuous batching scheduler

    Features:
    - Dynamic batch composition
    - Preemption support
    - Priority-based scheduling
    - Optimal GPU utilization

    Args:
        max_batch_size: Maximum batch size
        max_num_sequences: Maximum number of concurrent sequences
        block_size: KV-cache block size
    """

    def __init__(
        self,
        max_batch_size: int = 64,
        max_num_sequences: int = 256,
        block_size: int = 16,
    ):
        self.max_batch_size = max_batch_size
        self.max_num_sequences = max_num_sequences
        self.block_size = block_size

        # Request queues
        self.waiting_queue: List[GenerationRequest] = []
        self.running_requests: Dict[int, GenerationRequest] = {}
        self.finished_requests: Dict[int, GenerationRequest] = {}

        # Statistics
        self.total_requests = 0
        self.next_request_id = 0

    def add_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> int:
        """
        Add a new generation request

        Args:
            prompt_tokens: Tokenized prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Request ID
        """
        request_id = self.next_request_id
        self.next_request_id += 1

        request = GenerationRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            status=RequestStatus.WAITING,
            generated_tokens=[],
            arrival_time=time.time(),
        )

        self.waiting_queue.append(request)
        self.total_requests += 1

        return request_id

    def schedule(self, num_free_blocks: int) -> List[GenerationRequest]:
        """
        Schedule requests for next iteration

        Args:
            num_free_blocks: Number of free KV-cache blocks

        Returns:
            List of requests to process
        """
        # Check running requests
        batch = []

        # Add running requests to batch
        for request in self.running_requests.values():
            if not request.is_finished:
                batch.append(request)

        # Try to add waiting requests
        while len(batch) < self.max_batch_size and self.waiting_queue:
            # Check if we have enough memory
            request = self.waiting_queue[0]

            # Calculate blocks needed for this request
            blocks_needed = (
                request.prompt_len + self.block_size - 1
            ) // self.block_size

            if blocks_needed <= num_free_blocks:
                # Move to running
                request = self.waiting_queue.pop(0)
                request.status = RequestStatus.RUNNING
                request.start_time = time.time()
                self.running_requests[request.request_id] = request
                batch.append(request)

                # Update available blocks
                num_free_blocks -= blocks_needed
            else:
                # Not enough memory
                break

        return batch

    def update_finished(self, request_id: int, new_token: int, is_eos: bool = False) -> None:
        """
        Update request with new generated token

        Args:
            request_id: Request ID
            new_token: Newly generated token
            is_eos: Whether this is end-of-sequence
        """
        if request_id in self.running_requests:
            request = self.running_requests[request_id]
            request.generated_tokens.append(new_token)

            # Check if finished
            if is_eos or len(request.generated_tokens) >= request.max_tokens:
                request.status = RequestStatus.FINISHED
                request.finish_time = time.time()

                # Move to finished
                self.finished_requests[request_id] = request
                del self.running_requests[request_id]

    def get_request(self, request_id: int) -> Optional[GenerationRequest]:
        """
        Get request by ID

        Args:
            request_id: Request ID

        Returns:
            Request or None if not found
        """
        if request_id in self.running_requests:
            return self.running_requests[request_id]
        elif request_id in self.finished_requests:
            return self.finished_requests[request_id]
        else:
            # Check waiting queue
            for request in self.waiting_queue:
                if request.request_id == request_id:
                    return request
        return None

    def get_statistics(self) -> Dict[str, float]:
        """
        Get scheduler statistics

        Returns:
            Dictionary with statistics
        """
        total_finished = len(self.finished_requests)
        if total_finished == 0:
            return {
                "total_requests": self.total_requests,
                "waiting": len(self.waiting_queue),
                "running": len(self.running_requests),
                "finished": 0,
                "avg_latency": 0.0,
                "avg_throughput": 0.0,
            }

        # Calculate average latency
        latencies = []
        for request in self.finished_requests.values():
            if request.start_time and request.finish_time:
                latency = request.finish_time - request.start_time
                latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Calculate throughput (tokens/second)
        total_tokens = sum(
            len(r.generated_tokens) for r in self.finished_requests.values()
        )
        total_time = sum(latencies)
        avg_throughput = total_tokens / total_time if total_time > 0 else 0.0

        return {
            "total_requests": self.total_requests,
            "waiting": len(self.waiting_queue),
            "running": len(self.running_requests),
            "finished": total_finished,
            "avg_latency": avg_latency,
            "avg_throughput": avg_throughput,
        }

    def cancel_request(self, request_id: int) -> bool:
        """
        Cancel a request

        Args:
            request_id: Request ID

        Returns:
            True if cancelled, False if not found
        """
        # Check waiting queue
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                request.status = RequestStatus.CANCELLED
                self.waiting_queue.pop(i)
                return True

        # Check running requests
        if request_id in self.running_requests:
            request = self.running_requests[request_id]
            request.status = RequestStatus.CANCELLED
            del self.running_requests[request_id]
            return True

        return False

    def clear_finished(self) -> None:
        """Clear finished requests from memory"""
        self.finished_requests.clear()


class PriorityScheduler(ContinuousBatchScheduler):
    """
    Priority-based scheduler

    Extends continuous batching with priority support
    """

    def __init__(
        self,
        max_batch_size: int = 64,
        max_num_sequences: int = 256,
        block_size: int = 16,
    ):
        super().__init__(max_batch_size, max_num_sequences, block_size)
        self.request_priorities: Dict[int, float] = {}

    def add_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        priority: float = 1.0,
    ) -> int:
        """Add request with priority"""
        request_id = super().add_request(
            prompt_tokens, max_tokens, temperature, top_p
        )
        self.request_priorities[request_id] = priority
        return request_id

    def schedule(self, num_free_blocks: int) -> List[GenerationRequest]:
        """
        Schedule with priority

        Higher priority requests are scheduled first
        """
        # Sort waiting queue by priority
        self.waiting_queue.sort(
            key=lambda r: self.request_priorities.get(r.request_id, 1.0),
            reverse=True,
        )

        return super().schedule(num_free_blocks)
