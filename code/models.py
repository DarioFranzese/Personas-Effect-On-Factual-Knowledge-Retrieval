from enum import Enum
from typing import Any, Dict, Literal, Optional, List
from pydantic import BaseModel, Field

class ModelArchitecture(str, Enum):
    BASE = "base"
    INSTRUCTION_TUNED = "instruction_tuned"
    MOE = "moe"
    REASONING = "reasoning"

class ModelAlignment(str, Enum):
    RLHF = "rlhf"
    DPO = "dpo"
    GRPO = "grpo"
    NONE = "none"
    OTHER = "other"

class BackendType(str, Enum):
    VLLM = "vllm"
    GPT = "gpt"

class QuantizationType(str, Enum):
    NONE = "none"
    GGUF = "gguf"
    FP8 = "fp8"
    AWQ = "awq"
    

class ModelConfig(BaseModel):
    model_id: str
    size: Optional[str] = None
    architecture: List[ModelArchitecture] = [ModelArchitecture.BASE]
    alignment: ModelAlignment = ModelAlignment.NONE
    backend: BackendType
    
    system_prompt: Optional[str] = None
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    max_model_len: int = Field(32768, gt=0)
    reasoning_enabled: bool = False
    reasoning_effort: Optional[str] = None
    
    # Parametri per la quantizzazione (es. "awq", "gptq", "fp8")
    quantization: Optional[str] = None
    
    extra_template_kwargs: Dict[str, Any] = Field(default_factory=dict)

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class LLMResponse(BaseModel):
    text: str
    reasoning_text: Optional[str] = None
    output_text: Optional[str] = None
    model_id: str
    backend: BackendType
    prompt: str
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    raw: Dict[str, Any] = None

    def get_details(self) -> None:
        """Stampa tutti gli attributi formattati a schermo."""
        def _preview(value: Optional[str], limit: int = 120) -> str:
            if not value:
                return "N/A"
            compact = " ".join(value.split())
            if len(compact) <= limit:
                return compact
            return compact[:limit] + "..."

        div = "-" * 40
        print(f"\n{div}")
        print(f"{'DETTAGLI RISPOSTA':^40}")
        print(div)
        print(f"{'Model ID:':<18} {self.model_id}")
        print(f"{'Prompt:':<18} {self.prompt}")
        print(f"{'Backend:':<18} {self.backend.value.upper()}")
        print(f"{'Finish Reason:':<18} {self.finish_reason or 'N/A'}")
        print(f"{'Output Tokens:':<18} {self.output_tokens if self.output_tokens is not None else 'N/A'}")
        print(f"{'Reasoning Tokens:':<18} {self.reasoning_tokens if self.reasoning_tokens is not None else 'N/A'}")
        print(f"{'Reasoning Text:':<18} {_preview(self.reasoning_text)}")
        print(f"{'Output Text:':<18} {_preview(self.output_text)}")
        print(div)
        print(f"Testo Generato:\n{self.text}")
        print(f"{div}\n")