"""Tests for engine factory and registry."""

from unittest.mock import patch

import pytest

from course_builder.engine.base import GenerationEngine, GenerationError, GenerationResult
from course_builder.engine.factory import (
    EngineConfig,
    check_engine_availability,
    create_engine,
    create_engine_from_config,
    get_engine_spec,
    list_engines,
    register_engine,
    unregister_engine,
)


class MockEngine(GenerationEngine):
    """Mock engine for testing."""

    def __init__(self, model: str = "mock-model", **kwargs):
        self._model = model
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def engine_type(self) -> str:
        return "mock"

    def generate(self, prompt, config=None):
        return GenerationResult(text=f"Mock response to: {prompt}")

    def chat(self, messages, config=None):
        return GenerationResult(text="Mock chat response")

    def is_available(self) -> bool:
        return True


class TestEngineRegistry:
    """Tests for engine registration and lookup."""

    def test_register_engine(self):
        # Register a mock engine
        register_engine(
            name="test-mock",
            engine_class=MockEngine,
            default_model="default-mock",
            description="Test mock engine",
            required_packages=["mock-pkg"],
            env_vars=["MOCK_API_KEY"],
        )

        try:
            spec = get_engine_spec("test-mock")
            assert spec is not None
            assert spec.name == "test-mock"
            assert spec.engine_class == MockEngine
            assert spec.default_model == "default-mock"
            assert spec.description == "Test mock engine"
            assert "mock-pkg" in spec.required_packages
            assert "MOCK_API_KEY" in spec.env_vars
        finally:
            unregister_engine("test-mock")

    def test_unregister_engine(self):
        register_engine("temp-engine", MockEngine)
        assert get_engine_spec("temp-engine") is not None

        unregister_engine("temp-engine")
        assert get_engine_spec("temp-engine") is None

    def test_list_engines(self):
        engines = list_engines()
        # Should at least have gemini registered
        engine_names = [e.name for e in engines]
        assert "gemini" in engine_names

    def test_get_engine_spec_unknown(self):
        spec = get_engine_spec("nonexistent-engine")
        assert spec is None


class TestCreateEngine:
    """Tests for create_engine function."""

    def test_create_engine_with_registered_type(self):
        register_engine("create-test", MockEngine, default_model="test-default")

        try:
            engine = create_engine("create-test")
            assert isinstance(engine, MockEngine)
            assert engine.model_name == "test-default"
        finally:
            unregister_engine("create-test")

    def test_create_engine_with_custom_model(self):
        register_engine("create-test-2", MockEngine)

        try:
            engine = create_engine("create-test-2", model="custom-model")
            assert engine.model_name == "custom-model"
        finally:
            unregister_engine("create-test-2")

    def test_create_engine_with_kwargs(self):
        register_engine("create-test-3", MockEngine)

        try:
            engine = create_engine("create-test-3", model="test", extra_param="value")
            assert engine._kwargs.get("extra_param") == "value"
        finally:
            unregister_engine("create-test-3")

    def test_create_engine_unknown_type_raises(self):
        with pytest.raises(GenerationError) as exc_info:
            create_engine("unknown-engine-xyz")

        assert "Unknown engine type" in str(exc_info.value)
        assert "unknown-engine-xyz" in str(exc_info.value)


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_from_dict_minimal(self):
        config = EngineConfig.from_dict({"engine": "test"})
        assert config.engine == "test"
        assert config.model is None
        assert config.api_key is None

    def test_from_dict_full(self):
        config = EngineConfig.from_dict({
            "engine": "gemini",
            "model": "gemini-pro",
            "api_key": "secret",
            "base_url": "http://localhost",
            "temperature": 0.5,
            "max_tokens": 100,
            "custom_param": "value",
        })
        assert config.engine == "gemini"
        assert config.model == "gemini-pro"
        assert config.api_key == "secret"
        assert config.base_url == "http://localhost"
        assert config.temperature == 0.5
        assert config.max_tokens == 100
        assert config.extra["custom_param"] == "value"

    def test_to_kwargs(self):
        config = EngineConfig(
            engine="test",
            model="test-model",
            api_key="key",
            extra={"param": "value"},
        )
        kwargs = config.to_kwargs()
        assert kwargs["model"] == "test-model"
        assert kwargs["api_key"] == "key"
        assert kwargs["param"] == "value"
        assert "engine" not in kwargs  # engine is not passed to constructor


class TestCreateEngineFromConfig:
    """Tests for create_engine_from_config function."""

    def test_from_dict(self):
        register_engine("config-test", MockEngine)

        try:
            config = {"engine": "config-test", "model": "from-config"}
            engine = create_engine_from_config(config)
            assert isinstance(engine, MockEngine)
            assert engine.model_name == "from-config"
        finally:
            unregister_engine("config-test")

    def test_from_engine_config(self):
        register_engine("config-test-2", MockEngine)

        try:
            config = EngineConfig(engine="config-test-2", model="from-object")
            engine = create_engine_from_config(config)
            assert engine.model_name == "from-object"
        finally:
            unregister_engine("config-test-2")


class TestCheckEngineAvailability:
    """Tests for check_engine_availability function."""

    def test_unknown_engine(self):
        available, reason = check_engine_availability("nonexistent-xyz")
        assert available is False
        assert "Unknown engine type" in reason

    def test_gemini_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            # Clear GOOGLE_API_KEY
            import os

            original = os.environ.pop("GOOGLE_API_KEY", None)
            try:
                available, reason = check_engine_availability("gemini")
                # May fail on missing package or missing env var
                if not available:
                    assert "GOOGLE_API_KEY" in reason or "package" in reason.lower()
            finally:
                if original:
                    os.environ["GOOGLE_API_KEY"] = original

    def test_gemini_with_api_key(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            available, reason = check_engine_availability("gemini")
            # Should pass env var check (may still fail on package check depending on environment)
            # At minimum, should not fail on env var
            if not available:
                assert "GOOGLE_API_KEY" not in reason


class TestGeminiEngineIntegration:
    """Integration tests for Gemini engine (requires API key)."""

    @pytest.fixture
    def skip_without_api_key(self):
        import os

        if not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

    def test_create_gemini_engine(self, skip_without_api_key):
        engine = create_engine("gemini")
        assert engine.engine_type == "gemini"
        assert "gemini" in engine.model_name.lower()

    def test_gemini_generate(self, skip_without_api_key):
        engine = create_engine("gemini", model="gemini-flash-latest")
        result = engine.generate("What is 2+2? Reply with just the number.")
        assert "4" in result.text
        assert result.is_complete


class TestVLLMServerEngineCreation:
    """Tests for vLLM server engine creation."""

    def test_create_vllm_server_engine(self):
        engine = create_engine(
            "vllm-server",
            base_url="http://localhost:8000/v1",
            model="test-model",
        )
        assert engine.engine_type == "vllm-server"

    def test_vllm_server_default_url(self):
        engine = create_engine("vllm-server")
        assert "localhost:8000" in engine._base_url
