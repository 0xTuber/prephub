"""Tests for EngineProvider and pipeline integration."""

import os
from unittest.mock import MagicMock, patch

import pytest

from course_builder.engine.base import GenerationEngine, GenerationResult, StopReason, TokenUsage
from course_builder.engine.provider import (
    EngineProvider,
    ModelConfig,
    create_gemini_provider,
    create_hybrid_provider,
    create_vllm_provider,
    create_vllm_server_provider,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_basic_config(self):
        config = ModelConfig(
            engine_type="gemini",
            model="gemini-pro",
        )
        assert config.engine_type == "gemini"
        assert config.model == "gemini-pro"
        assert config.api_key is None
        assert config.base_url is None

    def test_config_with_all_fields(self):
        config = ModelConfig(
            engine_type="vllm-server",
            model="llama-3.1-8b",
            api_key="test-key",
            base_url="http://localhost:8000/v1",
            extra={"timeout": 30},
        )
        assert config.engine_type == "vllm-server"
        assert config.model == "llama-3.1-8b"
        assert config.api_key == "test-key"
        assert config.base_url == "http://localhost:8000/v1"
        assert config.extra == {"timeout": 30}

    def test_to_engine_kwargs(self):
        config = ModelConfig(
            engine_type="gemini",
            model="gemini-pro",
            api_key="my-key",
            extra={"temperature": 0.5},
        )
        kwargs = config.to_engine_kwargs()
        assert kwargs["model"] == "gemini-pro"
        assert kwargs["api_key"] == "my-key"
        assert kwargs["temperature"] == 0.5

    def test_to_engine_kwargs_minimal(self):
        config = ModelConfig(
            engine_type="gemini",
            model="gemini-flash",
        )
        kwargs = config.to_engine_kwargs()
        assert kwargs == {"model": "gemini-flash"}

    def test_immutable(self):
        config = ModelConfig(
            engine_type="gemini",
            model="gemini-pro",
        )
        with pytest.raises(AttributeError):
            config.model = "different-model"


class TestEngineProviderInit:
    """Tests for EngineProvider initialization."""

    def test_keyword_only_parameters(self):
        """All parameters must be keyword-only."""
        # This should work (keyword-only)
        provider = EngineProvider(
            engine_type="gemini",
            generation_model="gemini-flash",
            validation_model="gemini-pro",
        )
        assert provider.generation_model == "gemini-flash"
        assert provider.validation_model == "gemini-pro"

        # This should fail (positional args)
        with pytest.raises(TypeError):
            EngineProvider("gemini", "gemini-flash", "gemini-pro")

    def test_stores_configuration(self):
        provider = EngineProvider(
            engine_type="gemini",
            generation_model="gemini-2.0-flash",
            validation_model="gemini-pro",
            api_key="test-key",
        )
        assert provider.engine_type == "gemini"
        assert provider.generation_model == "gemini-2.0-flash"
        assert provider.validation_model == "gemini-pro"
        assert provider._api_key == "test-key"

    def test_lazy_initialization(self):
        provider = EngineProvider(
            engine_type="gemini",
            generation_model="gemini-flash",
            validation_model="gemini-pro",
        )
        # Engines should not be created yet
        assert provider._generation_engine is None
        assert provider._validation_engine is None


class TestEngineProviderEngineAccess:
    """Tests for EngineProvider engine access."""

    @patch("course_builder.engine.provider.create_engine")
    def test_generation_engine_lazy_created(self, mock_create):
        mock_engine = MagicMock(spec=GenerationEngine)
        mock_create.return_value = mock_engine

        provider = EngineProvider(
            engine_type="gemini",
            generation_model="gemini-flash",
            validation_model="gemini-pro",
            api_key="test-key",
        )

        # Access generation engine
        engine = provider.generation_engine

        # Should have created the engine
        mock_create.assert_called_once_with(
            "gemini",
            model="gemini-flash",
            api_key="test-key",
        )
        assert engine == mock_engine

    @patch("course_builder.engine.provider.create_engine")
    def test_validation_engine_lazy_created(self, mock_create):
        mock_engine = MagicMock(spec=GenerationEngine)
        mock_create.return_value = mock_engine

        provider = EngineProvider(
            engine_type="gemini",
            generation_model="gemini-flash",
            validation_model="gemini-pro",
            api_key="test-key",
        )

        # Access validation engine
        engine = provider.validation_engine

        # Should have created the engine with validation model
        mock_create.assert_called_once_with(
            "gemini",
            model="gemini-pro",
            api_key="test-key",
        )
        assert engine == mock_engine

    @patch("course_builder.engine.provider.create_engine")
    def test_engines_cached(self, mock_create):
        mock_engine = MagicMock(spec=GenerationEngine)
        mock_create.return_value = mock_engine

        provider = EngineProvider(
            engine_type="gemini",
            generation_model="gemini-flash",
            validation_model="gemini-pro",
        )

        # Access generation engine twice
        engine1 = provider.generation_engine
        engine2 = provider.generation_engine

        # Should be the same instance (only created once)
        assert engine1 is engine2
        assert mock_create.call_count == 1


class TestEngineProviderFromConfigs:
    """Tests for EngineProvider.from_configs()."""

    @patch("course_builder.engine.provider.create_engine")
    def test_creates_from_separate_configs(self, mock_create):
        gen_engine = MagicMock(spec=GenerationEngine)
        val_engine = MagicMock(spec=GenerationEngine)
        mock_create.side_effect = [gen_engine, val_engine]

        gen_config = ModelConfig(
            engine_type="vllm-server",
            model="llama-8b",
            base_url="http://localhost:8000/v1",
        )
        val_config = ModelConfig(
            engine_type="gemini",
            model="gemini-pro",
            api_key="test-key",
        )

        provider = EngineProvider.from_configs(
            generation_config=gen_config,
            validation_config=val_config,
        )

        # Engines should be created immediately
        assert provider.generation_engine == gen_engine
        assert provider.validation_engine == val_engine

        # Verify calls
        assert mock_create.call_count == 2

    def test_keyword_only_from_configs(self):
        """from_configs must use keyword-only parameters."""
        gen_config = ModelConfig(engine_type="gemini", model="flash")
        val_config = ModelConfig(engine_type="gemini", model="pro")

        # This should fail (positional args)
        with pytest.raises(TypeError):
            EngineProvider.from_configs(gen_config, val_config)


class TestEngineProviderRepr:
    """Tests for EngineProvider string representation."""

    def test_repr(self):
        provider = EngineProvider(
            engine_type="gemini",
            generation_model="gemini-flash",
            validation_model="gemini-pro",
        )
        result = repr(provider)
        assert "EngineProvider" in result
        assert "gemini" in result
        assert "gemini-flash" in result
        assert "gemini-pro" in result


class TestConvenienceFunctions:
    """Tests for convenience provider creation functions."""

    @patch("course_builder.engine.provider.create_engine")
    def test_create_gemini_provider(self, mock_create):
        provider = create_gemini_provider()
        assert provider.engine_type == "gemini"
        assert provider.generation_model == "gemini-flash-latest"
        assert provider.validation_model == "gemini-pro"

    @patch("course_builder.engine.provider.create_engine")
    def test_create_gemini_provider_custom_models(self, mock_create):
        provider = create_gemini_provider(
            generation_model="gemini-2.0-flash",
            validation_model="gemini-2.0-pro",
            api_key="my-key",
        )
        assert provider.generation_model == "gemini-2.0-flash"
        assert provider.validation_model == "gemini-2.0-pro"
        assert provider._api_key == "my-key"

    @patch("course_builder.engine.provider.create_engine")
    def test_create_vllm_provider(self, mock_create):
        """Test offline vLLM provider (direct GPU inference)."""
        provider = create_vllm_provider(model="llama-3.1-8b")
        assert provider.engine_type == "vllm"
        assert provider.generation_model == "llama-3.1-8b"
        # Same model for both (single GPU)
        assert provider.validation_model == "llama-3.1-8b"

    @patch("course_builder.engine.provider.create_engine")
    def test_create_vllm_server_provider(self, mock_create):
        """Test vLLM server provider (OpenAI-compatible API)."""
        provider = create_vllm_server_provider(
            generation_model="llama-3.1-8b",
        )
        assert provider.engine_type == "vllm-server"
        assert provider.generation_model == "llama-3.1-8b"
        # Validation model defaults to generation model
        assert provider.validation_model == "llama-3.1-8b"

    @patch("course_builder.engine.provider.create_engine")
    def test_create_vllm_server_provider_separate_validation(self, mock_create):
        provider = create_vllm_server_provider(
            generation_model="llama-3.1-8b",
            validation_model="llama-3.1-70b",
            base_url="http://gpu-server:8000/v1",
        )
        assert provider.generation_model == "llama-3.1-8b"
        assert provider.validation_model == "llama-3.1-70b"
        assert provider._base_url == "http://gpu-server:8000/v1"

    @patch("course_builder.engine.provider.create_engine")
    def test_create_hybrid_provider(self, mock_create):
        gen_engine = MagicMock(spec=GenerationEngine)
        val_engine = MagicMock(spec=GenerationEngine)
        mock_create.side_effect = [gen_engine, val_engine]

        provider = create_hybrid_provider(
            generation_engine_type="vllm-server",
            generation_model="llama-8b",
            validation_engine_type="gemini",
            validation_model="gemini-pro",
            generation_base_url="http://localhost:8000/v1",
            validation_api_key="google-key",
        )

        assert provider.generation_model == "llama-8b"
        assert provider.validation_model == "gemini-pro"
        # Mixed engines means engine_type is None
        assert provider.engine_type is None


class TestPipelineIntegration:
    """Tests for engine integration with pipeline steps."""

    def test_engine_aware_step_accepts_engine(self):
        """EngineAwareStep should accept engine parameter."""
        from course_builder.pipeline.base import EngineAwareStep

        mock_engine = MagicMock(spec=GenerationEngine)

        class TestStep(EngineAwareStep):
            def run(self, context):
                return context

        step = TestStep(engine=mock_engine)
        assert step.get_engine() == mock_engine

    def test_engine_aware_step_keyword_only(self):
        """EngineAwareStep must use keyword-only parameters."""
        from course_builder.pipeline.base import EngineAwareStep

        mock_engine = MagicMock(spec=GenerationEngine)

        class TestStep(EngineAwareStep):
            def run(self, context):
                return context

        # This should fail (positional arg)
        with pytest.raises(TypeError):
            TestStep(mock_engine)

    def test_require_engine_raises_when_none(self):
        """require_engine should raise when no engine configured."""
        from course_builder.pipeline.base import EngineAwareStep

        class TestStep(EngineAwareStep):
            def run(self, context):
                return context

        step = TestStep()
        with pytest.raises(RuntimeError, match="requires a generation engine"):
            step.require_engine()

    def test_require_engine_returns_engine(self):
        """require_engine should return engine when configured."""
        from course_builder.pipeline.base import EngineAwareStep

        mock_engine = MagicMock(spec=GenerationEngine)

        class TestStep(EngineAwareStep):
            def run(self, context):
                return context

        step = TestStep(engine=mock_engine)
        assert step.require_engine() == mock_engine

    def test_item_content_step_accepts_engine(self):
        """ItemContentGenerationStep should accept engine parameter."""
        from course_builder.pipeline.content import ItemContentGenerationStep

        mock_engine = MagicMock(spec=GenerationEngine)
        mock_engine.engine_type = "mock"
        mock_engine.model_name = "mock-model"

        step = ItemContentGenerationStep(
            engine=mock_engine,
            max_workers=4,
        )
        assert step.get_engine() == mock_engine

    def test_validation_step_accepts_engine(self):
        """HierarchicalValidationStep should accept engine parameter."""
        from course_builder.pipeline.validation import HierarchicalValidationStep

        mock_engine = MagicMock(spec=GenerationEngine)
        mock_engine.engine_type = "mock"
        mock_engine.model_name = "mock-model"

        step = HierarchicalValidationStep(
            engine=mock_engine,
            max_workers=4,
            skip_llm_review=True,  # For testing without LLM
        )
        assert step.get_engine() == mock_engine

    def test_correction_step_accepts_engine(self):
        """CorrectionApplicationStep should accept engine parameter."""
        from course_builder.pipeline.validation import CorrectionApplicationStep

        mock_engine = MagicMock(spec=GenerationEngine)
        mock_engine.engine_type = "mock"
        mock_engine.model_name = "mock-model"

        step = CorrectionApplicationStep(
            engine=mock_engine,
            max_workers=4,
        )
        assert step.get_engine() == mock_engine


class TestEngineProviderWithRealSteps:
    """Integration tests with real pipeline step behavior."""

    @patch("course_builder.engine.provider.create_engine")
    def test_provider_engines_work_with_steps(self, mock_create):
        """Provider's engines should work with pipeline steps."""
        from course_builder.pipeline.content import ItemContentGenerationStep
        from course_builder.pipeline.validation import CorrectionApplicationStep
        from course_builder.pipeline.validation import HierarchicalValidationStep

        # Create mock engines
        gen_engine = MagicMock(spec=GenerationEngine)
        gen_engine.engine_type = "gemini"
        gen_engine.model_name = "gemini-flash"

        val_engine = MagicMock(spec=GenerationEngine)
        val_engine.engine_type = "gemini"
        val_engine.model_name = "gemini-pro"

        mock_create.side_effect = [gen_engine, val_engine]

        # Create provider
        provider = EngineProvider(
            engine_type="gemini",
            generation_model="gemini-flash",
            validation_model="gemini-pro",
        )

        # Create steps with provider's engines
        content_step = ItemContentGenerationStep(
            engine=provider.generation_engine,
            max_workers=4,
        )
        validation_step = HierarchicalValidationStep(
            engine=provider.validation_engine,
            max_workers=4,
        )
        correction_step = CorrectionApplicationStep(
            engine=provider.generation_engine,
            max_workers=4,
        )

        # Verify steps have correct engines
        assert content_step.get_engine() == gen_engine
        assert validation_step.get_engine() == val_engine
        assert correction_step.get_engine() == gen_engine
