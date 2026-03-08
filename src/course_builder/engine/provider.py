"""Engine Provider for pipeline integration.

This module provides a centralized way to manage generation engines for the pipeline,
with separate configurations for content generation and validation.

The standard pattern for pipeline integration:
1. Create an EngineProvider with explicit model configurations
2. Pass the provider to pipeline steps
3. Steps use provider.generation_engine or provider.validation_engine as needed

Example:
    # In main pipeline script
    provider = EngineProvider(
        engine_type="gemini",
        generation_model="gemini-flash-latest",  # Fast model for content
        validation_model="gemini-pro",           # Smarter model for validation
    )

    pipeline = Pipeline(
        steps=[
            ItemContentGenerationStep(engine=provider.generation_engine),
            HierarchicalValidationStep(engine=provider.validation_engine),
        ]
    )
"""

from dataclasses import dataclass, field
from typing import Any

from course_builder.engine.base import GenerationEngine
from course_builder.engine.factory import create_engine


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific model.

    All parameters are keyword-only to enforce explicit configuration.

    Attributes:
        engine_type: Type of engine ("gemini", "vllm", "vllm-server").
        model: Model identifier (e.g., "gemini-pro", "meta-llama/Llama-3.1-8B-Instruct").
        api_key: API key for cloud providers (optional, uses env var if not set).
        base_url: Base URL for server-based engines (optional).
        extra: Additional engine-specific configuration.
    """

    engine_type: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_engine_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for engine creation."""
        kwargs = {"model": self.model}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        kwargs.update(self.extra)
        return kwargs


class EngineProvider:
    """Centralized provider for generation and validation engines.

    This class manages two separate engines:
    - generation_engine: For content generation (can use faster/cheaper model)
    - validation_engine: For quality validation (should use smarter model)

    All constructor parameters are keyword-only to enforce explicit configuration.

    Example:
        # Using Gemini API
        provider = EngineProvider(
            engine_type="gemini",
            generation_model="gemini-flash-latest",
            validation_model="gemini-pro",
        )

        # Using vLLM server
        provider = EngineProvider(
            engine_type="vllm-server",
            generation_model="meta-llama/Llama-3.1-8B-Instruct",
            validation_model="meta-llama/Llama-3.1-70B-Instruct",
            base_url="http://localhost:8000/v1",
        )

        # Mixed engines (generation=local, validation=API)
        provider = EngineProvider.from_configs(
            generation_config=ModelConfig(
                engine_type="vllm-server",
                model="meta-llama/Llama-3.1-8B-Instruct",
                base_url="http://localhost:8000/v1",
            ),
            validation_config=ModelConfig(
                engine_type="gemini",
                model="gemini-pro",
            ),
        )
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        engine_type: str,
        generation_model: str,
        validation_model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        validation_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the engine provider.

        All parameters are keyword-only.

        Args:
            engine_type: Type of engine to use ("gemini", "vllm", "vllm-server").
            generation_model: Model for content generation (can be faster/cheaper).
            validation_model: Model for validation (should be smarter/more capable).
            api_key: API key for cloud providers (optional).
            base_url: Base URL for server-based engines (optional).
            generation_kwargs: Additional kwargs for generation engine.
            validation_kwargs: Additional kwargs for validation engine.
        """
        self._engine_type = engine_type
        self._generation_model = generation_model
        self._validation_model = validation_model
        self._api_key = api_key
        self._base_url = base_url
        self._generation_kwargs = generation_kwargs or {}
        self._validation_kwargs = validation_kwargs or {}

        # Lazy initialization
        self._generation_engine: GenerationEngine | None = None
        self._validation_engine: GenerationEngine | None = None

    @classmethod
    def from_configs(
        cls,
        *,  # Force keyword-only arguments
        generation_config: ModelConfig,
        validation_config: ModelConfig,
    ) -> "EngineProvider":
        """Create provider from separate generation and validation configs.

        Use this when you want different engine types for generation vs validation.

        Args:
            generation_config: Configuration for the generation engine.
            validation_config: Configuration for the validation engine.

        Returns:
            EngineProvider instance.
        """
        provider = cls.__new__(cls)
        provider._engine_type = None  # Mixed engines
        provider._generation_model = generation_config.model
        provider._validation_model = validation_config.model
        provider._api_key = None
        provider._base_url = None
        provider._generation_kwargs = {}
        provider._validation_kwargs = {}

        # Create engines directly from configs
        provider._generation_engine = create_engine(
            generation_config.engine_type,
            **generation_config.to_engine_kwargs(),
        )
        provider._validation_engine = create_engine(
            validation_config.engine_type,
            **validation_config.to_engine_kwargs(),
        )

        return provider

    def _build_engine_kwargs(self, model: str, extra_kwargs: dict) -> dict[str, Any]:
        """Build kwargs for engine creation."""
        kwargs = {"model": model}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        kwargs.update(extra_kwargs)
        return kwargs

    @property
    def generation_engine(self) -> GenerationEngine:
        """Get the generation engine (lazy initialization)."""
        if self._generation_engine is None:
            kwargs = self._build_engine_kwargs(
                self._generation_model,
                self._generation_kwargs,
            )
            self._generation_engine = create_engine(self._engine_type, **kwargs)
        return self._generation_engine

    @property
    def validation_engine(self) -> GenerationEngine:
        """Get the validation engine (lazy initialization)."""
        if self._validation_engine is None:
            kwargs = self._build_engine_kwargs(
                self._validation_model,
                self._validation_kwargs,
            )
            self._validation_engine = create_engine(self._engine_type, **kwargs)
        return self._validation_engine

    @property
    def generation_model(self) -> str:
        """Get the generation model name."""
        return self._generation_model

    @property
    def validation_model(self) -> str:
        """Get the validation model name."""
        return self._validation_model

    @property
    def engine_type(self) -> str | None:
        """Get the engine type (None if using mixed engines)."""
        return self._engine_type

    def __repr__(self) -> str:
        return (
            f"EngineProvider("
            f"engine_type={self._engine_type!r}, "
            f"generation_model={self._generation_model!r}, "
            f"validation_model={self._validation_model!r})"
        )


# =============================================================================
# Convenience functions for common configurations
# =============================================================================


def create_gemini_provider(
    *,  # Force keyword-only
    generation_model: str = "gemini-flash-latest",
    validation_model: str = "gemini-pro",
    api_key: str | None = None,
) -> EngineProvider:
    """Create an EngineProvider configured for Gemini API.

    Args:
        generation_model: Model for generation (default: gemini-flash-latest).
        validation_model: Model for validation (default: gemini-pro).
        api_key: Google API key (optional, uses GOOGLE_API_KEY env var).

    Returns:
        Configured EngineProvider.
    """
    return EngineProvider(
        engine_type="gemini",
        generation_model=generation_model,
        validation_model=validation_model,
        api_key=api_key,
    )


def create_vllm_provider(
    *,  # Force keyword-only
    model: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> EngineProvider:
    """Create an EngineProvider configured for offline vLLM (direct GPU inference).

    Args:
        model: Model to load (e.g., "meta-llama/Llama-3-8B-Instruct").
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.

    Returns:
        Configured EngineProvider.
    """
    return EngineProvider(
        engine_type="vllm",
        generation_model=model,
        validation_model=model,  # Same model for both (single GPU)
        generation_kwargs={
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        },
    )


def create_vllm_server_provider(
    *,  # Force keyword-only
    generation_model: str,
    validation_model: str | None = None,
    base_url: str = "http://localhost:8000/v1",
) -> EngineProvider:
    """Create an EngineProvider configured for vLLM server.

    Args:
        generation_model: Model for generation.
        validation_model: Model for validation (defaults to generation_model).
        base_url: vLLM server URL.

    Returns:
        Configured EngineProvider.
    """
    return EngineProvider(
        engine_type="vllm-server",
        generation_model=generation_model,
        validation_model=validation_model or generation_model,
        base_url=base_url,
    )


def create_hybrid_provider(
    *,  # Force keyword-only
    generation_engine_type: str,
    generation_model: str,
    validation_engine_type: str,
    validation_model: str,
    generation_api_key: str | None = None,
    generation_base_url: str | None = None,
    validation_api_key: str | None = None,
    validation_base_url: str | None = None,
) -> EngineProvider:
    """Create an EngineProvider with different engines for generation vs validation.

    Useful when you want to use a local model for generation (cheaper/faster)
    but a cloud API for validation (smarter).

    Args:
        generation_engine_type: Engine type for generation.
        generation_model: Model for generation.
        validation_engine_type: Engine type for validation.
        validation_model: Model for validation.
        generation_api_key: API key for generation engine.
        generation_base_url: Base URL for generation engine.
        validation_api_key: API key for validation engine.
        validation_base_url: Base URL for validation engine.

    Returns:
        Configured EngineProvider with separate engines.
    """
    gen_extra = {}
    if generation_api_key:
        gen_extra["api_key"] = generation_api_key
    if generation_base_url:
        gen_extra["base_url"] = generation_base_url

    val_extra = {}
    if validation_api_key:
        val_extra["api_key"] = validation_api_key
    if validation_base_url:
        val_extra["base_url"] = validation_base_url

    return EngineProvider.from_configs(
        generation_config=ModelConfig(
            engine_type=generation_engine_type,
            model=generation_model,
            **gen_extra,
        ),
        validation_config=ModelConfig(
            engine_type=validation_engine_type,
            model=validation_model,
            **val_extra,
        ),
    )
