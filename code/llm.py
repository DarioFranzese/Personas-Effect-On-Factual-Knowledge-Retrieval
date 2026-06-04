import logging
from typing import Dict, List, Optional, Any, Union
from models import BackendType, Message, ModelConfig, LLMResponse, ModelArchitecture
from registry import get_model_defaults
from backends import VLLMBackend, GPTBackend

logger = logging.getLogger(__name__)

class LLM:

    def __init__(
        self, 
        model_id: str, 
        api_keys: Optional[Dict[str, str]] = None,
        reasoning: bool = True,
        **kwargs: Any
    ) -> None:
        
        self._is_cleaned_up = False
        defaults = get_model_defaults(model_id)

        # Logica per impostare la quantizzazione basata sulla dimensione (> 20B)
        size_str = defaults.get("size", "")
        if size_str:
            try:
                # Estraiamo la prima parte (es. "379B" da "379B (17B)")
                first_part = size_str.split()[0]
                # Filtriamo solo i numeri e il punto decimale (per gestire 1.6T, 70B, etc)
                num_str = "".join([c for c in first_part if c.isdigit() or c == '.'])
                if num_str:
                    size_val = float(num_str)
                    # Se superiore a 20 o espresso in Tera (T), abilitiamo il supporto
                    # Se non è già specificato in defaults o kwargs, impostiamo un default
                    if (size_val > 20 or 'T' in first_part.upper()) and "quantization" not in kwargs and "quantization" not in defaults:
                        kwargs["quantization"] = "awq" # Valore di default per attivare il parametro
            except ValueError:
                pass

        max_model_len = kwargs.pop("max_model_len", None)
        if max_model_len is None:
            max_model_len = defaults.get("max_model_len", 16384)
        
        config_dict = {
            **defaults, 
            "model_id": model_id, 
            "max_tokens": None,
            "max_model_len": max_model_len,
            **kwargs
        }
        
        self.config = ModelConfig(**config_dict)

        self.reasoning = ModelArchitecture.REASONING in self.config.architecture
        
        self._apply_reasoning_setting(reasoning)
        
        self._backend = self._init_backend(api_keys)

        

    def _apply_reasoning_setting(self, reasoning: bool) -> None:
        model_id_lower = self.config.model_id.lower()
        supports_reasoning = ModelArchitecture.REASONING in self.config.architecture

        if "gpt-oss" in model_id_lower and not reasoning:
            raise ValueError(
                "gpt-oss models require reasoning enabled; use set_gpt_oss_reasoning_effort()."
            )

        if not supports_reasoning:
            if reasoning:
                logger.warning("Model does not support reasoning; ignoring reasoning=True.")

            self.config.reasoning_enabled = False
            self.config.reasoning_effort = None
            self.config.extra_template_kwargs.pop("reasoning_effort", None)
            self.config.extra_template_kwargs.pop("enable_thinking", None)
            return

        if not reasoning and "deepseek-r1" in model_id_lower:
            logger.warning("Model only supports reasoning mode; ignoring reasoning=False.")
            return

        self.config.reasoning_enabled = reasoning

        if "qwen" in model_id_lower:
            self.config.extra_template_kwargs["enable_thinking"] = reasoning

        if self.config.backend == BackendType.GPT:
            if reasoning:
                self.config.reasoning_effort = "medium"
                self.config.extra_template_kwargs["reasoning_effort"] = self.config.reasoning_effort
            else:
                self.config.reasoning_effort = None
                self.config.extra_template_kwargs.pop("reasoning_effort", None)

    def _check_status(self) -> None:
        if self._is_cleaned_up: raise RuntimeError("Istanza LLM disattivata.")

    def show_info(self) -> None:
        self._check_status()
        div = "=" * 60
        print(f"\n{div}\n{'LLM INTERFACE REPORT':^60}\n{div}")
        print(f"[GENERAL] ID: {self.config.model_id} | Backend: {self.config.backend.value.upper()}")
        max_tokens_display = self.config.max_tokens if self.config.max_tokens is not None else "AUTO"
        print(
            f"[CONFIG] Max Output: {max_tokens_display} | Max Model Len: {self.config.max_model_len} | "
            f"Quantization: {self.config.quantization or 'None'}"
        )
        reasoning_state = "ON" if self.config.reasoning_enabled else "OFF"
        effort = f" | Effort: {self.config.reasoning_effort}" if self.config.reasoning_effort else ""
        print(f"[REASONING] {reasoning_state}{effort}")
        try:
            details = self._backend.get_details()
            print(f"[BACKEND DETAILS]")
            for k, v in details.items(): print(f"  {k}: {v}")
        except: pass
        print(f"{div}\n")

    def set_reasoning(self, reasoning: bool) -> None:
        """Enable or disable reasoning at runtime when supported by the model."""
        self._check_status()
        self._apply_reasoning_setting(reasoning)

    def set_gpt_oss_reasoning_effort(self, effort: str) -> None:
        """Set reasoning effort for gpt-oss models only."""
        self._check_status()
        model_id_lower = self.config.model_id.lower()
        if "gpt-oss" not in model_id_lower:
            raise ValueError("reasoning_effort is only supported for gpt-oss models.")
        if effort not in {"low", "medium", "high"}:
            raise ValueError("reasoning_effort must be one of: low, medium, high.")
        self.config.reasoning_effort = effort
        self.config.extra_template_kwargs["reasoning_effort"] = effort

    def _init_backend(self, api_keys: Optional[Dict[str, str]]):
        if self.config.backend == BackendType.GPT:
            return GPTBackend(self.config, api_key=(api_keys or {}).get("OPENAI_API_KEY"))
        if self.config.backend == BackendType.VLLM:
            try:
                return VLLMBackend(self.config)
            except Exception as exc:
                logger.exception("VLLMBackend init failed")
                raise RuntimeError(f"VLLMBackend init failed: {exc}") from exc
        raise ValueError(f"Unsupported backend: {self.config.backend}")

    def generate(self, message: Message) -> LLMResponse:
        self._check_status()
        # Wrap single message in a list for backend compatibility
        return self._backend.generate([message])[0]

    def generate_batch(self, messages: List[Message] | List[List[Message]]) -> List[LLMResponse]:
        self._check_status()
        return self._backend.generate(messages)

    def cleanup(self) -> None:
        import gc, torch
        if self._is_cleaned_up: return
        if hasattr(self._backend, "_engine"): del self._backend._engine
        self._is_cleaned_up = True; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()