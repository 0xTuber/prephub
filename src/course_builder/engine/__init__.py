"""Generation Engine - Unified interface for text generation.

This module provides a unified interface for text generation that supports:
- Google Gemini API (cloud)
- vLLM local inference (GPU)
- vLLM OpenAI-compatible server

Quick Start:
    from course_builder.engine import create_engine, GenerationConfig

    # Create engine (auto-detects based on env vars)
    engine = create_engine("gemini")

    # Simple generation
    result = engine.generate("What is 2+2?")
    print(result.text)

    # With configuration
    config = GenerationConfig(temperature=0.0, max_tokens=100)
    result = engine.generate("Write a haiku", config=config)

    # Chat completion
    from course_builder.engine import Message
    messages = [
        Message(role="user", content="Hello!"),
    ]
    result = engine.chat(messages)

Engine Types:
    - "gemini": Google Gemini API (requires GOOGLE_API_KEY)
    - "vllm": vLLM local inference (requires GPU + vllm package)
    - "vllm-server": vLLM OpenAI-compatible server

Example Configurations:
    # Gemini
    engine = create_engine("gemini", model="gemini-pro", api_key="...")

    # vLLM local
    engine = create_engine("vllm", model="meta-llama/Llama-3.1-8B-Instruct")

    # vLLM server
    engine = create_engine("vllm-server", base_url="http://localhost:8000/v1")
"""

from course_builder.engine.base import (
    GenerationConfig,
    GenerationEngine,
    GenerationError,
    GenerationResult,
    Message,
    StopReason,
    StreamChunk,
    TokenUsage,
)
from course_builder.engine.factory import (
    EngineConfig,
    EngineSpec,
    check_engine_availability,
    create_engine,
    create_engine_from_config,
    get_default_engine,
    get_engine_spec,
    list_engines,
    register_engine,
    unregister_engine,
)
from course_builder.engine.provider import (
    EngineProvider,
    ModelConfig,
    create_gemini_provider,
    create_hybrid_provider,
    create_vllm_provider,
    create_vllm_server_provider,
)


# Lazy imports for engines (to avoid import errors if packages not installed)
def get_gemini_engine():
    """Get the GeminiEngine class (lazy import)."""
    from course_builder.engine.gemini import GeminiEngine

    return GeminiEngine


def get_vllm_engine():
    """Get the VLLMEngine class (lazy import)."""
    from course_builder.engine.vllm import VLLMEngine

    return VLLMEngine


def get_vllm_server_engine():
    """Get the VLLMServerEngine class (lazy import)."""
    from course_builder.engine.vllm import VLLMServerEngine

    return VLLMServerEngine


__all__ = [
    # Base types
    "GenerationConfig",
    "GenerationEngine",
    "GenerationError",
    "GenerationResult",
    "Message",
    "StopReason",
    "StreamChunk",
    "TokenUsage",
    # Factory functions
    "create_engine",
    "create_engine_from_config",
    "get_default_engine",
    "register_engine",
    "unregister_engine",
    "list_engines",
    "get_engine_spec",
    "check_engine_availability",
    "EngineConfig",
    "EngineSpec",
    # Provider (for pipeline integration)
    "EngineProvider",
    "ModelConfig",
    "create_gemini_provider",
    "create_vllm_provider",
    "create_vllm_server_provider",
    "create_hybrid_provider",
    # Lazy engine getters
    "get_gemini_engine",
    "get_vllm_engine",
    "get_vllm_server_engine",
]
