"""Base types and protocol for generation engines.

This module defines the interface that all generation engines must implement,
allowing seamless switching between local models (vLLM) and API providers (Gemini, OpenAI).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator


class StopReason(str, Enum):
    """Reason why generation stopped."""

    END_OF_SEQUENCE = "end_of_sequence"  # Normal completion
    MAX_TOKENS = "max_tokens"  # Hit token limit
    STOP_SEQUENCE = "stop_sequence"  # Hit a stop sequence
    ERROR = "error"  # Generation failed
    UNKNOWN = "unknown"


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic, higher = more random).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling (only consider top k tokens).
        stop_sequences: List of strings that stop generation when encountered.
        system_prompt: System instruction/prompt for the model.
        json_mode: If True, request JSON output from the model.
        seed: Random seed for reproducible generation (if supported).
    """

    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    system_prompt: str | None = None
    json_mode: bool = False
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Message:
    """A message in a conversation.

    Attributes:
        role: The role of the message sender ("user", "assistant", "system").
        content: The text content of the message.
    """

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class TokenUsage:
    """Token usage statistics for a generation.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the generated completion.
        total_tokens: Total tokens used (prompt + completion).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GenerationResult:
    """Result of a text generation request.

    Attributes:
        text: The generated text.
        stop_reason: Why generation stopped.
        usage: Token usage statistics.
        model: The model that generated this result.
        raw_response: The raw response from the underlying API (for debugging).
    """

    text: str
    stop_reason: StopReason = StopReason.UNKNOWN
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None
    raw_response: Any = None

    @property
    def is_complete(self) -> bool:
        """Check if generation completed normally."""
        return self.stop_reason in (StopReason.END_OF_SEQUENCE, StopReason.STOP_SEQUENCE)


@dataclass
class StreamChunk:
    """A chunk of streamed generation output.

    Attributes:
        text: The text in this chunk.
        is_final: Whether this is the last chunk.
        stop_reason: Why generation stopped (only set on final chunk).
    """

    text: str
    is_final: bool = False
    stop_reason: StopReason | None = None


class GenerationEngine(ABC):
    """Abstract base class for text generation engines.

    All generation engines (vLLM, Gemini, OpenAI, etc.) must implement this interface.

    Example:
        engine = GeminiEngine(api_key="...", model="gemini-pro")

        # Simple generation
        result = engine.generate("What is 2+2?")
        print(result.text)

        # With configuration
        config = GenerationConfig(temperature=0.0, max_tokens=100)
        result = engine.generate("Write a haiku", config=config)

        # Chat completion
        messages = [
            Message(role="user", content="Hello!"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
        result = engine.chat(messages)

        # Streaming
        for chunk in engine.generate_stream("Tell me a story"):
            print(chunk.text, end="", flush=True)
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name/identifier of the current model."""
        ...

    @property
    @abstractmethod
    def engine_type(self) -> str:
        """Get the engine type (e.g., 'gemini', 'vllm', 'openai')."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: The input text prompt.
            config: Generation configuration (optional).

        Returns:
            GenerationResult with the generated text and metadata.

        Raises:
            GenerationError: If generation fails.
        """
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate a chat completion from a conversation history.

        Args:
            messages: List of messages in the conversation.
            config: Generation configuration (optional).

        Returns:
            GenerationResult with the generated response and metadata.

        Raises:
            GenerationError: If generation fails.
        """
        ...

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[StreamChunk]:
        """Generate text from a prompt with streaming output.

        Default implementation falls back to non-streaming generate.
        Engines that support streaming should override this method.

        Args:
            prompt: The input text prompt.
            config: Generation configuration (optional).

        Yields:
            StreamChunk objects with generated text pieces.
        """
        # Default: fall back to non-streaming
        result = self.generate(prompt, config)
        yield StreamChunk(
            text=result.text,
            is_final=True,
            stop_reason=result.stop_reason,
        )

    def chat_stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> Iterator[StreamChunk]:
        """Generate a chat completion with streaming output.

        Default implementation falls back to non-streaming chat.
        Engines that support streaming should override this method.

        Args:
            messages: List of messages in the conversation.
            config: Generation configuration (optional).

        Yields:
            StreamChunk objects with generated text pieces.
        """
        # Default: fall back to non-streaming
        result = self.chat(messages, config)
        yield StreamChunk(
            text=result.text,
            is_final=True,
            stop_reason=result.stop_reason,
        )

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationResult]:
        """Generate text for multiple prompts.

        Default implementation processes sequentially.
        Engines that support batching should override for efficiency.

        Args:
            prompts: List of input prompts.
            config: Generation configuration (optional).

        Returns:
            List of GenerationResult objects.
        """
        return [self.generate(prompt, config) for prompt in prompts]

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine is available and ready to use.

        Returns:
            True if the engine can accept requests, False otherwise.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Default implementation provides a rough estimate.
        Engines with tokenizers should override for accuracy.

        Args:
            text: The text to tokenize.

        Returns:
            Estimated token count.
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4


class GenerationError(Exception):
    """Exception raised when generation fails."""

    def __init__(
        self,
        message: str,
        engine: str | None = None,
        model: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.engine = engine
        self.model = model
        self.cause = cause

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.engine:
            parts.append(f"engine={self.engine}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.cause:
            parts.append(f"cause={self.cause}")
        return " | ".join(parts)
