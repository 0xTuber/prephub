"""Factory and registry for generation engines.

This module provides:
- Engine registry for dynamic engine discovery
- Factory functions for easy engine instantiation
- Configuration-based engine creation

Example:
    # Simple creation with defaults
    engine = create_engine("gemini")

    # With specific model
    engine = create_engine("gemini", model="gemini-pro")

    # vLLM local
    engine = create_engine("vllm", model="meta-llama/Llama-3.1-8B-Instruct")

    # vLLM server
    engine = create_engine("vllm-server", base_url="http://localhost:8000/v1")

    # From config dict
    config = {
        "engine": "gemini",
        "model": "gemini-flash-latest",
        "temperature": 0.7,
    }
    engine = create_engine_from_config(config)
"""

from dataclasses import dataclass, field
from typing import Any, Type

from course_builder.engine.base import GenerationEngine, GenerationError


@dataclass
class EngineSpec:
    """Specification for a registered engine type."""

    name: str
    engine_class: Type[GenerationEngine]
    default_model: str | None = None
    description: str = ""
    required_packages: list[str] = field(default_factory=list)
    env_vars: list[str] = field(default_factory=list)


# Global engine registry
_ENGINE_REGISTRY: dict[str, EngineSpec] = {}


def register_engine(
    name: str,
    engine_class: Type[GenerationEngine],
    default_model: str | None = None,
    description: str = "",
    required_packages: list[str] | None = None,
    env_vars: list[str] | None = None,
) -> None:
    """Register an engine type in the global registry.

    Args:
        name: Unique identifier for the engine (e.g., "gemini", "vllm").
        engine_class: The engine class to instantiate.
        default_model: Default model to use if none specified.
        description: Human-readable description of the engine.
        required_packages: Python packages required for this engine.
        env_vars: Environment variables used by this engine.
    """
    _ENGINE_REGISTRY[name] = EngineSpec(
        name=name,
        engine_class=engine_class,
        default_model=default_model,
        description=description,
        required_packages=required_packages or [],
        env_vars=env_vars or [],
    )


def unregister_engine(name: str) -> None:
    """Remove an engine from the registry."""
    _ENGINE_REGISTRY.pop(name, None)


def list_engines() -> list[EngineSpec]:
    """List all registered engines."""
    return list(_ENGINE_REGISTRY.values())


def get_engine_spec(name: str) -> EngineSpec | None:
    """Get the specification for a registered engine."""
    return _ENGINE_REGISTRY.get(name)


def create_engine(
    engine_type: str,
    model: str | None = None,
    **kwargs,
) -> GenerationEngine:
    """Create a generation engine by type.

    Args:
        engine_type: Type of engine ("gemini", "vllm", "vllm-server", "openai").
        model: Model name/identifier. If None, uses engine default.
        **kwargs: Additional arguments passed to the engine constructor.

    Returns:
        Configured GenerationEngine instance.

    Raises:
        GenerationError: If engine type is unknown or creation fails.

    Example:
        # Gemini with default model
        engine = create_engine("gemini")

        # Gemini with specific model
        engine = create_engine("gemini", model="gemini-pro")

        # vLLM local
        engine = create_engine("vllm", model="meta-llama/Llama-3.1-8B-Instruct")

        # vLLM server
        engine = create_engine("vllm-server", base_url="http://localhost:8000/v1")
    """
    spec = _ENGINE_REGISTRY.get(engine_type)

    if spec is None:
        available = ", ".join(_ENGINE_REGISTRY.keys())
        raise GenerationError(
            f"Unknown engine type: '{engine_type}'. Available: {available}",
            engine=engine_type,
        )

    # Use provided model or fall back to default
    effective_model = model or spec.default_model

    if effective_model:
        kwargs["model"] = effective_model

    try:
        return spec.engine_class(**kwargs)
    except TypeError as e:
        raise GenerationError(
            f"Invalid arguments for {engine_type} engine: {e}",
            engine=engine_type,
            model=effective_model,
            cause=e,
        )
    except Exception as e:
        raise GenerationError(
            f"Failed to create {engine_type} engine: {e}",
            engine=engine_type,
            model=effective_model,
            cause=e,
        )


@dataclass
class EngineConfig:
    """Configuration for engine creation.

    Attributes:
        engine: Engine type ("gemini", "vllm", "vllm-server").
        model: Model name/identifier.
        api_key: API key (for API-based engines).
        base_url: Base URL (for server-based engines).
        temperature: Default sampling temperature.
        max_tokens: Default max tokens for generation.
        extra: Additional engine-specific options.
    """

    engine: str
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "EngineConfig":
        """Create config from dictionary."""
        known_keys = {"engine", "model", "api_key", "base_url", "temperature", "max_tokens"}
        extra = {k: v for k, v in config.items() if k not in known_keys}

        return cls(
            engine=config["engine"],
            model=config.get("model"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            temperature=config.get("temperature"),
            max_tokens=config.get("max_tokens"),
            extra=extra,
        )

    def to_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for engine creation."""
        kwargs = {}
        if self.model:
            kwargs["model"] = self.model
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        kwargs.update(self.extra)
        return kwargs


def create_engine_from_config(config: dict[str, Any] | EngineConfig) -> GenerationEngine:
    """Create an engine from a configuration dictionary or EngineConfig.

    Args:
        config: Configuration dict or EngineConfig object.

    Returns:
        Configured GenerationEngine instance.

    Example:
        config = {
            "engine": "gemini",
            "model": "gemini-pro",
            "api_key": "your-key",
        }
        engine = create_engine_from_config(config)

        # Or with EngineConfig
        config = EngineConfig(engine="vllm", model="llama-3.1-8b")
        engine = create_engine_from_config(config)
    """
    if isinstance(config, dict):
        config = EngineConfig.from_dict(config)

    return create_engine(config.engine, **config.to_kwargs())


# =============================================================================
# Auto-register built-in engines
# =============================================================================


def _register_builtin_engines():
    """Register all built-in engines."""
    # Gemini
    try:
        from course_builder.engine.gemini import GeminiEngine

        register_engine(
            name="gemini",
            engine_class=GeminiEngine,
            default_model="gemini-flash-latest",
            description="Google Gemini API (cloud)",
            required_packages=["google-genai"],
            env_vars=["GOOGLE_API_KEY"],
        )
    except ImportError:
        pass

    # vLLM (local)
    try:
        from course_builder.engine.vllm import VLLMEngine

        register_engine(
            name="vllm",
            engine_class=VLLMEngine,
            default_model=None,  # Must specify model
            description="vLLM local inference (requires GPU)",
            required_packages=["vllm"],
        )
    except ImportError:
        pass

    # vLLM Server
    try:
        from course_builder.engine.vllm import VLLMServerEngine

        register_engine(
            name="vllm-server",
            engine_class=VLLMServerEngine,
            default_model=None,  # Uses server's model
            description="vLLM OpenAI-compatible server",
        )
    except ImportError:
        pass


# Register engines on module import
_register_builtin_engines()


# =============================================================================
# Convenience functions
# =============================================================================


def get_default_engine() -> GenerationEngine:
    """Get the default engine based on available configuration.

    Checks for environment variables and returns the first available engine.

    Order of preference:
    1. Gemini (if GOOGLE_API_KEY is set)
    2. vLLM server (if VLLM_BASE_URL is set)
    3. Raises error if none available

    Returns:
        A configured GenerationEngine.

    Raises:
        GenerationError: If no engine can be configured.
    """
    import os

    # Try Gemini first
    if os.environ.get("GOOGLE_API_KEY"):
        return create_engine("gemini")

    # Try vLLM server
    vllm_url = os.environ.get("VLLM_BASE_URL")
    if vllm_url:
        return create_engine("vllm-server", base_url=vllm_url)

    raise GenerationError(
        "No generation engine available. Set GOOGLE_API_KEY or VLLM_BASE_URL.",
    )


def check_engine_availability(engine_type: str) -> tuple[bool, str]:
    """Check if an engine type is available and ready to use.

    Args:
        engine_type: Engine type to check.

    Returns:
        Tuple of (is_available, reason_if_not).
    """
    spec = get_engine_spec(engine_type)

    if spec is None:
        return False, f"Unknown engine type: {engine_type}"

    # Check required packages
    for package in spec.required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            return False, f"Missing package: {package}"

    # Check environment variables
    import os

    for env_var in spec.env_vars:
        if not os.environ.get(env_var):
            return False, f"Missing environment variable: {env_var}"

    return True, "Available"
