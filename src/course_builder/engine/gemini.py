"""Gemini API generation engine.

This module provides a generation engine implementation for Google's Gemini API.
"""

import os
import time
from typing import Iterator

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


class GeminiEngine(GenerationEngine):
    """Generation engine using Google's Gemini API.

    Example:
        engine = GeminiEngine(api_key="your-api-key", model="gemini-flash-latest")
        result = engine.generate("What is the capital of France?")
        print(result.text)

    Environment Variables:
        GOOGLE_API_KEY: Default API key if not provided in constructor.
    """

    def __init__(
        self,
        model: str = "gemini-flash-latest",
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float | None = None,
    ):
        """Initialize the Gemini engine.

        Args:
            model: The Gemini model to use (e.g., "gemini-pro", "gemini-flash-latest").
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Base delay between retries (exponential backoff).
            timeout: Request timeout in seconds (optional).
        """
        self._model = model
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout = timeout
        self._client = None

        if not self._api_key:
            raise GenerationError(
                "No API key provided. Set GOOGLE_API_KEY environment variable or pass api_key.",
                engine="gemini",
                model=model,
            )

    def _get_client(self):
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self._api_key)
            except ImportError:
                raise GenerationError(
                    "google-genai package not installed. Run: pip install google-genai",
                    engine="gemini",
                    model=self._model,
                )
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def engine_type(self) -> str:
        return "gemini"

    def _build_config(self, config: GenerationConfig | None):
        """Build Gemini-specific generation config."""
        from google.genai import types

        if config is None:
            return None

        genai_config = {}

        if config.system_prompt:
            genai_config["system_instruction"] = config.system_prompt

        # Build generation config dict
        gen_params = {}
        if config.max_tokens is not None:
            gen_params["max_output_tokens"] = config.max_tokens
        if config.temperature is not None:
            gen_params["temperature"] = config.temperature
        if config.top_p is not None:
            gen_params["top_p"] = config.top_p
        if config.top_k is not None:
            gen_params["top_k"] = config.top_k
        if config.stop_sequences:
            gen_params["stop_sequences"] = config.stop_sequences

        if gen_params:
            genai_config.update(gen_params)

        # JSON mode
        if config.json_mode:
            genai_config["response_mime_type"] = "application/json"

        return types.GenerateContentConfig(**genai_config) if genai_config else None

    def _call_with_retry(self, fn):
        """Call function with exponential backoff retry."""
        last_error = None
        for attempt in range(self._max_retries):
            try:
                return fn()
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2**attempt)
                    time.sleep(delay)

        raise GenerationError(
            f"Generation failed after {self._max_retries} retries",
            engine="gemini",
            model=self._model,
            cause=last_error,
        )

    def _parse_stop_reason(self, response) -> StopReason:
        """Parse stop reason from Gemini response."""
        try:
            # Check if we have candidates
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    reason = str(candidate.finish_reason).upper()
                    if "STOP" in reason:
                        return StopReason.END_OF_SEQUENCE
                    elif "MAX" in reason or "LENGTH" in reason:
                        return StopReason.MAX_TOKENS
                    elif "SAFETY" in reason or "RECITATION" in reason:
                        return StopReason.ERROR
            return StopReason.END_OF_SEQUENCE
        except Exception:
            return StopReason.UNKNOWN

    def _parse_usage(self, response) -> TokenUsage:
        """Parse token usage from Gemini response."""
        try:
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                return TokenUsage(
                    prompt_tokens=getattr(usage, "prompt_token_count", 0),
                    completion_tokens=getattr(usage, "candidates_token_count", 0),
                    total_tokens=getattr(usage, "total_token_count", 0),
                )
        except Exception:
            pass
        return TokenUsage()

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt using Gemini API."""
        client = self._get_client()
        genai_config = self._build_config(config)

        def _call():
            return client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=genai_config,
            )

        response = self._call_with_retry(_call)

        return GenerationResult(
            text=response.text,
            stop_reason=self._parse_stop_reason(response),
            usage=self._parse_usage(response),
            model=self._model,
            raw_response=response,
        )

    def chat(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate a chat completion from conversation history."""
        client = self._get_client()
        from google.genai import types

        # Build config, extracting system prompt from messages if present
        effective_config = config or GenerationConfig()

        # Extract system message if present
        system_prompt = effective_config.system_prompt
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                chat_messages.append(msg)

        # Update config with system prompt
        if system_prompt and (config is None or config.system_prompt is None):
            effective_config = GenerationConfig(
                **{**effective_config.to_dict(), "system_prompt": system_prompt}
            )

        genai_config = self._build_config(effective_config)

        # Convert messages to Gemini format
        contents = []
        for msg in chat_messages:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

        def _call():
            return client.models.generate_content(
                model=self._model,
                contents=contents,
                config=genai_config,
            )

        response = self._call_with_retry(_call)

        return GenerationResult(
            text=response.text,
            stop_reason=self._parse_stop_reason(response),
            usage=self._parse_usage(response),
            model=self._model,
            raw_response=response,
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[StreamChunk]:
        """Generate text with streaming output."""
        client = self._get_client()
        genai_config = self._build_config(config)

        try:
            response_stream = client.models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=genai_config,
            )

            for chunk in response_stream:
                text = ""
                if hasattr(chunk, "text"):
                    text = chunk.text
                elif hasattr(chunk, "candidates") and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, "content") and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, "text"):
                                    text += part.text

                yield StreamChunk(text=text, is_final=False)

            # Final chunk
            yield StreamChunk(
                text="",
                is_final=True,
                stop_reason=StopReason.END_OF_SEQUENCE,
            )

        except Exception as e:
            yield StreamChunk(
                text="",
                is_final=True,
                stop_reason=StopReason.ERROR,
            )
            raise GenerationError(
                f"Streaming generation failed: {e}",
                engine="gemini",
                model=self._model,
                cause=e,
            )

    def is_available(self) -> bool:
        """Check if the Gemini API is available."""
        try:
            client = self._get_client()
            # Try a minimal request to verify connectivity
            response = client.models.generate_content(
                model=self._model,
                contents="Hi",
                config={"max_output_tokens": 1},
            )
            return response is not None
        except Exception:
            return False

    def generate_with_tools(
        self,
        prompt: str,
        tools: list,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate with tool/function calling support.

        This is a Gemini-specific feature for tool use.

        Args:
            prompt: The input prompt.
            tools: List of tool definitions (Gemini Tool objects).
            config: Generation configuration.

        Returns:
            GenerationResult with possible tool calls in raw_response.
        """
        client = self._get_client()
        from google.genai import types

        genai_config = self._build_config(config) or {}
        if isinstance(genai_config, types.GenerateContentConfig):
            genai_config = genai_config.to_dict() if hasattr(genai_config, "to_dict") else {}

        # Add tools to config
        genai_config["tools"] = tools

        config_obj = types.GenerateContentConfig(**genai_config)

        def _call():
            return client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config_obj,
            )

        response = self._call_with_retry(_call)

        return GenerationResult(
            text=response.text if hasattr(response, "text") else "",
            stop_reason=self._parse_stop_reason(response),
            usage=self._parse_usage(response),
            model=self._model,
            raw_response=response,
        )

    def generate_with_search(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate with Google Search grounding.

        This is a Gemini-specific feature that grounds responses in search results.

        Args:
            prompt: The input prompt.
            config: Generation configuration.

        Returns:
            GenerationResult with search-grounded response.
        """
        from google.genai import types

        tools = [types.Tool(google_search=types.GoogleSearch())]
        return self.generate_with_tools(prompt, tools, config)
