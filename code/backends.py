import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from models import BackendType, LLMResponse, Message, ModelArchitecture, ModelConfig
from reasoning_parser import parse_reasoning_output

logger = logging.getLogger(__name__)

class LLMBackend(ABC):
    OUTPUT_RESERVE_TOKENS = 256
    MIN_OUTPUT_TOKENS = 128

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, messages: List[Message]) -> List[LLMResponse]:
        pass

    @abstractmethod
    def get_details(self) -> Dict[str, Any]:
        pass


    def _prepend_system_prompt(self, messages: List[Message]) -> List[Message]:
        if self.config.system_prompt is not None and (not messages or messages[0].role != "system"):
            return [Message(role="system", content=self.config.system_prompt)] + list(messages)
        return list(messages)

    def _compute_max_new_tokens(self, prompt_tokens: int) -> int:
        if self.config.max_tokens is not None:
            return self.config.max_tokens
        max_model_len = self.config.max_model_len
        available = max_model_len - prompt_tokens - self.OUTPUT_RESERVE_TOKENS
        if available < self.MIN_OUTPUT_TOKENS:
            logger.warning(
                "Prompt uses %s tokens near max_model_len=%s; shrinking output budget.",
                prompt_tokens,
                max_model_len,
            )
            available = max(1, max_model_len - prompt_tokens)
        return max(1, available)

class VLLMBackend(LLMBackend):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        from vllm import LLM as _VLLM, SamplingParams, config as vllm_config
        self._SamplingParams = SamplingParams

        reasoning_config = None
        if ModelArchitecture.REASONING in config.architecture and config.reasoning_enabled:
            reasoning_config = vllm_config.ReasoningConfig()

        base_kwargs = {
            "model": config.model_id,
            "max_model_len": config.max_model_len,
        }
        
        if reasoning_config is not None:
            base_kwargs["reasoning_config"] = reasoning_config

        try:
            self._engine = _VLLM(
                dtype="auto",
                **base_kwargs
            )
        except Exception:
            if config.quantization == "awq":
                try:
                    self._engine = _VLLM(
                        quantization="awq_marlin",
                        dtype="float16",
                        **base_kwargs
                    )
                except Exception:
                    self._engine = _VLLM(
                        quantization="awq",
                        dtype="float16",
                        **base_kwargs
                    )
            else:
                quantization_param = None if config.quantization == "none" else config.quantization
                if quantization_param is None:
                    raise
                self._engine = _VLLM(
                    quantization=quantization_param,
                    dtype="auto",
                    **base_kwargs
                )

    def get_details(self) -> Dict[str, Any]:
        if not hasattr(self, "_engine"): return {"status": "Cleaned"}
        engine = self._engine.llm_engine
        m_cfg = getattr(engine, "model_config", None)
        return {
            "Dtype": getattr(m_cfg, "dtype", "N/A"), 
            "Max Len": getattr(m_cfg, "max_model_len", "N/A"),
            "Quantization": self.config.quantization or "None"
        }

    def generate(self, messages: List[Message]) -> List[LLMResponse]:
        if not messages:
            return []

        tokenizer = self._engine.get_tokenizer()
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        prompts: List[str] = []
        sampling_params_list: List[Any] = []

        for message in messages:
            prompt_messages = self._prepend_system_prompt([message])
            chat = [{"role": m.role, "content": m.content} for m in prompt_messages]
            prompt_tokens = self._count_prompt_tokens(tokenizer, chat)
            try:
                prompt = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.config.extra_template_kwargs
                )
            except ValueError:
                prompt = chat[-1].get("content", "") if chat else ""

            max_tokens = self._compute_max_new_tokens(prompt_tokens)
            sampling_params_list.append(
                self._SamplingParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=max_tokens,
                    stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
                )
            )
            prompts.append(prompt)


        outputs = self._engine.generate(prompts, sampling_params_list)
        responses: List[LLMResponse] = []
        for i, output in enumerate(outputs):
            result = output.outputs[0]
            reasoning_text, output_text = parse_reasoning_output(result.text)
            responses.append(
                LLMResponse(
                    text=result.text,
                    reasoning_text=reasoning_text,
                    output_text=output_text,
                    model_id=self.config.model_id,
                    prompt = prompts[i],
                    backend=BackendType.VLLM,
                    output_tokens=len(result.token_ids),
                    finish_reason=str(result.finish_reason),
                )
            )

        return responses


    def _count_prompt_tokens(self, tokenizer, chat: List[Dict[str, str]]) -> int:
        try:
            token_ids = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=True,
                **self.config.extra_template_kwargs,
            )
            return len(token_ids)
        except Exception:
            try:
                prompt = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.config.extra_template_kwargs,
                )
                return len(tokenizer.encode(prompt))
            except Exception:
                return 0

class GPTBackend(LLMBackend):
    def __init__(self, config: ModelConfig, api_key: Optional[str] = None) -> None:
        super().__init__(config)
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key: raise ValueError("OpenAI API key not found.")
        self._client = OpenAI(api_key=key)

    def get_details(self) -> Dict[str, Any]:
        return {"Provider": "OpenAI", "Reasoning": "Supported" if ModelArchitecture.REASONING in self.config.architecture else "No"}

    def generate(self, messages: List[Message]) -> List[LLMResponse]:
        responses: List[LLMResponse] = []
        for message in messages:
            prompt_messages = self._prepend_system_prompt([message])
            chat = [{"role": m.role, "content": m.content} for m in prompt_messages]
            prompt_tokens = self._estimate_prompt_tokens(prompt_messages)
            max_completion_tokens = self._compute_max_new_tokens(prompt_tokens)
            kwargs = dict(
                model=self.config.model_id,
                messages=chat,
                max_completion_tokens=max_completion_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            if ModelArchitecture.REASONING in self.config.architecture and self.config.reasoning_enabled:
                kwargs["reasoning_effort"] = self.config.extra_template_kwargs.get("reasoning_effort", "medium")

            response = self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning_text, output_text = parse_reasoning_output(content)
            responses.append(
                LLMResponse(
                    text=content,
                    reasoning_text=reasoning_text,
                    output_text=output_text,
                    model_id=self.config.model_id,
                    backend=BackendType.GPT,
                    finish_reason=choice.finish_reason,
                )
            )

        return responses

    def _estimate_prompt_tokens(self, messages: List[Message]) -> int:
        text = "\n".join(f"{m.role}: {m.content}" for m in messages)
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.config.model_id)
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)