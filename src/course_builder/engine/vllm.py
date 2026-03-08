"""vLLM generation engine.

This module provides generation engine implementations for vLLM:
1. VLLMEngine - Direct Python integration (requires GPU, vllm package)
2. VLLMServerEngine - OpenAI-compatible API client (connects to vLLM server)

The server mode is recommended for production as it:
- Allows running the model on a separate GPU machine
- Supports multiple clients
- Provides continuous batching for efficiency
"""

import json
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


class VLLMEngine(GenerationEngine):
    """Generation engine using vLLM for direct local inference.

    Requires:
        - vLLM package installed (pip install vllm)
        - GPU with sufficient VRAM for the model

    Example:
        engine = VLLMEngine(model="meta-llama/Llama-3.1-8B-Instruct")
        result = engine.generate("What is the capital of France?")
        print(result.text)

    Note:
        Model loading happens lazily on first generation request.
        This can take several minutes for large models.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
        quantization: str | None = None,
        trust_remote_code: bool = True,
        download_dir: str | None = None,
    ):
        """Initialize the vLLM engine.

        Args:
            model: HuggingFace model name or local path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
            max_model_len: Maximum sequence length. If None, uses model default.
            dtype: Data type for model weights ("auto", "float16", "bfloat16").
            quantization: Quantization method (None, "awq", "gptq", "squeezellm").
            trust_remote_code: Whether to trust remote code in model repo.
            download_dir: Directory to download model files.
        """
        self._model = model
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._quantization = quantization
        self._trust_remote_code = trust_remote_code
        self._download_dir = download_dir
        self._llm = None
        self._tokenizer = None

    def _get_llm(self):
        """Lazy-initialize the vLLM model."""
        if self._llm is None:
            try:
                from vllm import LLM

                self._llm = LLM(
                    model=self._model,
                    tensor_parallel_size=self._tensor_parallel_size,
                    gpu_memory_utilization=self._gpu_memory_utilization,
                    max_model_len=self._max_model_len,
                    dtype=self._dtype,
                    quantization=self._quantization,
                    trust_remote_code=self._trust_remote_code,
                    download_dir=self._download_dir,
                )
                self._tokenizer = self._llm.get_tokenizer()
            except ImportError:
                raise GenerationError(
                    "vllm package not installed. Run: pip install vllm",
                    engine="vllm",
                    model=self._model,
                )
            except Exception as e:
                raise GenerationError(
                    f"Failed to load vLLM model: {e}",
                    engine="vllm",
                    model=self._model,
                    cause=e,
                )
        return self._llm

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def engine_type(self) -> str:
        return "vllm"

    def _build_sampling_params(self, config: GenerationConfig | None):
        """Build vLLM SamplingParams from GenerationConfig."""
        from vllm import SamplingParams

        if config is None:
            return SamplingParams()

        params = {}

        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.top_k is not None:
            params["top_k"] = config.top_k
        if config.stop_sequences:
            params["stop"] = config.stop_sequences
        if config.seed is not None:
            params["seed"] = config.seed

        return SamplingParams(**params)

    def _format_prompt(self, prompt: str, config: GenerationConfig | None) -> str:
        """Format prompt with system instruction if provided."""
        if config and config.system_prompt:
            # Try to use chat template if available
            if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": prompt},
                ]
                try:
                    return self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass
            # Fallback: simple concatenation
            return f"{config.system_prompt}\n\n{prompt}"
        return prompt

    def _format_messages(self, messages: list[Message]) -> str:
        """Format messages for the model."""
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            chat_messages = [{"role": m.role, "content": m.content} for m in messages]
            try:
                return self._tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback: simple text format
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt using vLLM."""
        llm = self._get_llm()
        sampling_params = self._build_sampling_params(config)
        formatted_prompt = self._format_prompt(prompt, config)

        try:
            outputs = llm.generate([formatted_prompt], sampling_params)
            output = outputs[0]

            # Get the generated text
            text = output.outputs[0].text

            # Parse stop reason
            finish_reason = output.outputs[0].finish_reason
            if finish_reason == "stop":
                stop_reason = StopReason.END_OF_SEQUENCE
            elif finish_reason == "length":
                stop_reason = StopReason.MAX_TOKENS
            else:
                stop_reason = StopReason.UNKNOWN

            # Token usage
            usage = TokenUsage(
                prompt_tokens=len(output.prompt_token_ids),
                completion_tokens=len(output.outputs[0].token_ids),
                total_tokens=len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            )

            return GenerationResult(
                text=text,
                stop_reason=stop_reason,
                usage=usage,
                model=self._model,
                raw_response=output,
            )

        except Exception as e:
            raise GenerationError(
                f"Generation failed: {e}",
                engine="vllm",
                model=self._model,
                cause=e,
            )

    def chat(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate a chat completion from conversation history."""
        formatted_prompt = self._format_messages(messages)
        return self.generate(formatted_prompt, config)

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationResult]:
        """Generate text for multiple prompts efficiently using vLLM batching."""
        llm = self._get_llm()
        sampling_params = self._build_sampling_params(config)

        formatted_prompts = [self._format_prompt(p, config) for p in prompts]

        try:
            outputs = llm.generate(formatted_prompts, sampling_params)

            results = []
            for output in outputs:
                text = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason

                if finish_reason == "stop":
                    stop_reason = StopReason.END_OF_SEQUENCE
                elif finish_reason == "length":
                    stop_reason = StopReason.MAX_TOKENS
                else:
                    stop_reason = StopReason.UNKNOWN

                usage = TokenUsage(
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(output.outputs[0].token_ids),
                    total_tokens=len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                )

                results.append(
                    GenerationResult(
                        text=text,
                        stop_reason=stop_reason,
                        usage=usage,
                        model=self._model,
                        raw_response=output,
                    )
                )

            return results

        except Exception as e:
            raise GenerationError(
                f"Batch generation failed: {e}",
                engine="vllm",
                model=self._model,
                cause=e,
            )

    def is_available(self) -> bool:
        """Check if vLLM is available and model can be loaded."""
        try:
            self._get_llm()
            return True
        except Exception:
            return False

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        if self._tokenizer is None:
            self._get_llm()  # This initializes tokenizer
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return super().count_tokens(text)


class VLLMServerEngine(GenerationEngine):
    """Generation engine connecting to a vLLM OpenAI-compatible server.

    This is the recommended approach for production:
    1. Start vLLM server: python -m vllm.entrypoints.openai.api_server --model <model>
    2. Connect with this engine: VLLMServerEngine(base_url="http://localhost:8000/v1")

    The server provides:
    - Continuous batching for high throughput
    - Multiple client support
    - Separation of model loading from inference

    Example:
        # Start vLLM server first:
        # python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct

        engine = VLLMServerEngine(
            base_url="http://localhost:8000/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
        )
        result = engine.generate("What is the capital of France?")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str | None = None,
        api_key: str = "EMPTY",  # vLLM doesn't require a real key
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
    ):
        """Initialize the vLLM server engine.

        Args:
            base_url: Base URL of the vLLM server (e.g., "http://localhost:8000/v1").
            model: Model name (must match the model loaded in vLLM server).
                   If None, will query the server for available models.
            api_key: API key (vLLM accepts any value, default "EMPTY").
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Base delay between retries (exponential backoff).
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        if self._model is None:
            # Try to get model from server
            try:
                models = self._list_models()
                if models:
                    self._model = models[0]
            except Exception:
                pass
        return self._model or "unknown"

    @property
    def engine_type(self) -> str:
        return "vllm-server"

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        json_data: dict | None = None,
        stream: bool = False,
    ):
        """Make HTTP request to vLLM server."""
        import urllib.error
        import urllib.request

        url = f"{self._base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        data = None
        if json_data:
            data = json.dumps(json_data).encode("utf-8")

        request = urllib.request.Request(url, data=data, headers=headers, method=method)

        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = urllib.request.urlopen(request, timeout=self._timeout)
                if stream:
                    return response  # Return response object for streaming
                return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else ""
                last_error = GenerationError(
                    f"HTTP {e.code}: {error_body}",
                    engine="vllm-server",
                    model=self._model,
                )
            except Exception as e:
                last_error = e

            if attempt < self._max_retries - 1:
                time.sleep(self._retry_delay * (2**attempt))

        raise GenerationError(
            f"Request failed after {self._max_retries} retries",
            engine="vllm-server",
            model=self._model,
            cause=last_error,
        )

    def _list_models(self) -> list[str]:
        """List available models on the server."""
        response = self._make_request("models")
        return [m["id"] for m in response.get("data", [])]

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text using vLLM server.

        Uses chat/completions endpoint for compatibility with instruct models.
        """
        config = config or GenerationConfig()

        # Build messages - wrap prompt as user message for chat endpoint
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.max_tokens:
            payload["max_tokens"] = config.max_tokens
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences
        if config.seed is not None:
            payload["seed"] = config.seed

        response = self._make_request("chat/completions", method="POST", json_data=payload)

        # Parse response (chat format)
        choice = response["choices"][0]
        text = choice["message"]["content"]

        finish_reason = choice.get("finish_reason", "unknown")
        if finish_reason == "stop":
            stop_reason = StopReason.END_OF_SEQUENCE
        elif finish_reason == "length":
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.UNKNOWN

        usage_data = response.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return GenerationResult(
            text=text,
            stop_reason=stop_reason,
            usage=usage,
            model=self.model_name,
            raw_response=response,
        )

    def chat(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate chat completion using vLLM server."""
        config = config or GenerationConfig()

        # Build messages in OpenAI format
        openai_messages = []

        if config.system_prompt:
            openai_messages.append({"role": "system", "content": config.system_prompt})

        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.max_tokens:
            payload["max_tokens"] = config.max_tokens
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences
        if config.seed is not None:
            payload["seed"] = config.seed

        response = self._make_request("chat/completions", method="POST", json_data=payload)

        # Parse response
        choice = response["choices"][0]
        text = choice["message"]["content"]

        finish_reason = choice.get("finish_reason", "unknown")
        if finish_reason == "stop":
            stop_reason = StopReason.END_OF_SEQUENCE
        elif finish_reason == "length":
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.UNKNOWN

        usage_data = response.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return GenerationResult(
            text=text,
            stop_reason=stop_reason,
            usage=usage,
            model=self.model_name,
            raw_response=response,
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[StreamChunk]:
        """Generate text with streaming output using chat/completions endpoint."""
        config = config or GenerationConfig()

        # Build messages for chat endpoint
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }

        if config.max_tokens:
            payload["max_tokens"] = config.max_tokens
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        try:
            response = self._make_request(
                "chat/completions", method="POST", json_data=payload, stream=True
            )

            for line in response:
                line = line.decode("utf-8").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    text = delta.get("content", "")
                    finish_reason = choice.get("finish_reason")

                    if finish_reason:
                        stop_reason = (
                            StopReason.END_OF_SEQUENCE
                            if finish_reason == "stop"
                            else StopReason.MAX_TOKENS
                        )
                        yield StreamChunk(text=text, is_final=True, stop_reason=stop_reason)
                    else:
                        yield StreamChunk(text=text, is_final=False)

        except Exception as e:
            yield StreamChunk(text="", is_final=True, stop_reason=StopReason.ERROR)
            raise GenerationError(
                f"Streaming failed: {e}",
                engine="vllm-server",
                model=self._model,
                cause=e,
            )

    def is_available(self) -> bool:
        """Check if the vLLM server is available."""
        try:
            models = self._list_models()
            return len(models) > 0
        except Exception:
            return False
