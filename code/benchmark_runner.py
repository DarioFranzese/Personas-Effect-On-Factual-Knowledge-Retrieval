from __future__ import annotations

from pathlib import Path
import random

from datasets import load_dataset, get_dataset_config_names
from tqdm.auto import tqdm

from llm import LLM
from models import Message


class BenchmarkRunner:
    ANSWER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    FEW_SHOT_INTRO = "You remember that previously you solved the following problems:"
    NEW_PROBLEM_PREFIX = "Now answer this new question:"
    ZERO_SHOT_PREFIX = "Answer this question:"
    FINAL_ANSWER_INSTRUCTION = "Make sure to end with \"My Answer:\" followed by your answer."

    def __init__(
        self,
        model: LLM,
        personas: bool = True,
        shot_mode: str = "zero-shot",
        few_shot_seed: int = 42,
        num_shots: int = 3,
    ) -> None:
        self.model = model
        self.persona_root = Path("/home/dario/denv/experiments/personas")
        self.shot_mode = shot_mode
        self.few_shot_seed = few_shot_seed
        self.num_shots = num_shots
        self.use_personas = personas
        self._validate_shot_mode(self.shot_mode)

    def set_shot_mode(self, shot_mode: str) -> None:
        self._validate_shot_mode(shot_mode)
        self.shot_mode = shot_mode

    def run_benchmark(self, benchmark_label: str, shot_mode: str | None = None) -> list[dict]:
        shot_mode = shot_mode or self.shot_mode
        self._validate_shot_mode(shot_mode)
        label = benchmark_label.strip().lower()

        if label == "mmlu":
            return self.run_mmlu(shot_mode=shot_mode)
        if label == "mmlu_pro":
            return self.run_mmlu_pro(shot_mode=shot_mode)
        if label == "gpqa":
            return self.run_gpqa(shot_mode=shot_mode)
        if label == "aime":
            return self.run_aime_2026(shot_mode=shot_mode)

        raise ValueError(
            "Unsupported benchmark_label. Use one of: mmlu, mmlu-pro, gpqa, aime-2026."
        )
    

    def run_mmlu(self, shot_mode: str | None = None) -> list[dict]:
        shot_mode = shot_mode or self.shot_mode
        self._validate_shot_mode(shot_mode)
        results: list[dict] = []
        rng = random.Random(self.few_shot_seed)

        subsets = [
            name for name in get_dataset_config_names("cais/mmlu") if name != "all"
        ]
        for subset_index, subset in enumerate(subsets):
            ds = load_dataset("cais/mmlu", subset, split="test")
            self._set_persona("mmlu", subset)
            indices = list(range(len(ds)))
            batch_messages: list[Message] = []
            batch_records: list[dict] = []
            batch_messages: list[Message] = []
            batch_records: list[dict] = []

            for row_idx in tqdm(indices, desc=f"mmlu:{subset}"):
                row = ds[row_idx]
                few_shot_blocks = []
                if shot_mode == "few-shot":
                    sample_rows = self._sample_few_shots(
                        ds, indices, row_idx, rng, num_shots=self.num_shots
                    )
                    few_shot_blocks = [
                        self._build_example_block(
                            question=r["question"],
                            choices=r["choices"],
                            answer=r.get("answer"),
                            quote_question=True,
                        )
                        for r in sample_rows
                    ]

                question_block = self._build_question_block(
                    row["question"],
                    row["choices"],
                    quote_question=True,
                )
                prompt = self._build_prompt(question_block, few_shot_blocks, shot_mode=shot_mode)
                batch_messages.append(Message(role="user", content=prompt))
                batch_records.append(
                    {
                        "subject": row.get("subject", subset),
                        "question": row["question"],
                        "answer": row.get("answer"),
                        "index": subset_index,
                    }
                )

            self._append_batch_results(batch_messages, batch_records, results)
            return results

        return results

    def run_mmlu_pro(self, shot_mode: str | None = None) -> list[dict]:
        shot_mode = shot_mode or self.shot_mode
        self._validate_shot_mode(shot_mode)
        results: list[dict] = []
        rng = random.Random(self.few_shot_seed)

        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        category_to_indices: dict[str, list[int]] = {}
        for idx in range(len(ds)):
            category = ds[idx]["category"]
            category_to_indices.setdefault(category, []).append(idx)

        for category, indices in category_to_indices.items():
            self._set_persona("mmlu_pro", category)
            batch_messages: list[Message] = []
            batch_records: list[dict] = []
            for row_idx in tqdm(indices, desc=f"mmlu-pro:{category}"):
                row = ds[row_idx]
                answer = self._extract_answer(row, ["answer", "correct_answer", "label"])
                few_shot_blocks = []
                if shot_mode == "few-shot":
                    sample_rows = self._sample_few_shots(
                        ds, indices, row_idx, rng, num_shots=self.num_shots
                    )
                    few_shot_blocks = [
                        self._build_example_block(
                            question=r["question"],
                            choices=r["options"],
                            answer=self._extract_answer(
                                r, ["answer", "correct_answer", "label"]
                            ),
                            quote_question=True,
                        )
                        for r in sample_rows
                    ]

                question_block = self._build_question_block(
                    row["question"],
                    row["options"],
                    quote_question=True,
                )
                prompt = self._build_prompt(question_block, few_shot_blocks, shot_mode=shot_mode)
                batch_messages.append(Message(role="user", content=prompt))
                batch_records.append(
                    {
                        "subject": category,
                        "question": row["question"],
                        "answer": answer,
                        "index": row["question_id"],
                    }
                )

            self._append_batch_results(batch_messages, batch_records, results)
            return results

        return results
    

    def run_gpqa(self, shot_mode: str | None = None) -> list[dict]:
        shot_mode = shot_mode or self.shot_mode
        self._validate_shot_mode(shot_mode)
        results: list[dict] = []
        rng = random.Random(self.few_shot_seed)

        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        domain_to_indices: dict[str, list[int]] = {}
        for idx in range(len(ds)):
            domain = ds[idx]["High-level domain"]
            domain_to_indices.setdefault(domain, []).append(idx)

        for domain, indices in domain_to_indices.items():
            self._set_persona("gpqa", domain)
            batch_messages: list[Message] = []
            batch_records: list[dict] = []
            for row_idx in tqdm(indices, desc=f"gpqa:{domain}"):
                row = ds[row_idx]
                few_shot_blocks = []
                if shot_mode == "few-shot":
                    sample_rows = self._sample_few_shots(
                        ds, indices, row_idx, rng, num_shots=self.num_shots
                    )
                    few_shot_blocks = [
                        self._build_example_block(
                            question=r["Question"],
                            choices=None,
                            answer=r.get("Correct Answer"),
                            quote_question=False,
                        )
                        for r in sample_rows
                    ]

                question_block = self._build_question_block(
                    row["Question"],
                    choices=None,
                    quote_question=False,
                )
                prompt = self._build_prompt(question_block, few_shot_blocks, shot_mode=shot_mode)
                batch_messages.append(Message(role="user", content=prompt))
                batch_records.append(
                    {
                        "subject": domain,
                        "question": row["Question"],
                        "answer": row.get("Correct Answer"),
                        "index": row_idx,
                    }
                )

            self._append_batch_results(batch_messages, batch_records, results)
            return(results)

        return results
    

    def run_aime_2026(self, shot_mode: str | None = None) -> list[dict]:
        shot_mode = shot_mode or self.shot_mode
        self._validate_shot_mode(shot_mode)
        results: list[dict] = []
        rng = random.Random(self.few_shot_seed)

        ds = load_dataset("MathArena/aime_2026", split="train")
        self._set_persona("aime_2026", "math")
        indices = list(range(len(ds)))
        batch_messages: list[Message] = []
        batch_records: list[dict] = []

        for row_idx in tqdm(indices, desc="aime-2026"):
            row = ds[row_idx]
            few_shot_blocks = []
            if shot_mode == "few-shot":
                sample_rows = self._sample_few_shots(
                    ds, indices, row_idx, rng, num_shots=self.num_shots
                )
                few_shot_blocks = [
                    self._build_example_block(
                        question=r["problem"],
                        choices=None,
                        answer=r.get("answer"),
                        quote_question=False,
                    )
                    for r in sample_rows
                ]

            question_block = self._build_question_block(
                row["problem"],
                choices=None,
                quote_question=False,
            )
            prompt = self._build_prompt(question_block, few_shot_blocks, shot_mode=shot_mode)
            batch_messages.append(Message(role="user", content=prompt))
            batch_records.append(
                {
                    "subject": "math",
                    "question": row["problem"],
                    "answer": row.get("answer"),
                    "index": row.get("problem_idx", row_idx),
                }
            )

        self._append_batch_results(batch_messages, batch_records, results)

        return results

    def _validate_shot_mode(self, shot_mode: str) -> None:
        if shot_mode not in {"zero-shot", "few-shot"}:
            raise ValueError("shot_mode must be 'zero-shot' or 'few-shot'.")

    def _append_batch_results(
        self,
        batch_messages: list[Message],
        batch_records: list[dict],
        results: list[dict],
    ) -> None:
        if not batch_messages:
            return
        responses = self.model.generate_batch(batch_messages)
        for response, record in zip(responses, batch_records):
            record["output"] = response
            results.append(record)

    def _set_persona(self, dataset_name: str, subject: str) -> None:
        if not self.use_personas:
            self.model.config.system_prompt = None
            return
        self.model.config.system_prompt = self._load_persona(dataset_name, subject)

    def _load_persona(self, dataset_name: str, subject: str) -> str:
        persona_path = self.persona_root / dataset_name / f"{subject}.md"
        print(persona_path)
        if not persona_path.exists():
            raise FileNotFoundError(f"Persona file not found: {persona_path}")
        return persona_path.read_text(encoding="utf-8").strip()

    def _format_choices(self, choices) -> str:
        if not choices:
            return ""
        return " ".join(
            f"{self.ANSWER_LABELS[i]}) {str(choice)}"
            for i, choice in enumerate(choices)
        )

    def _resolve_choice_label(self, answer, choices) -> str | None:
        if answer is None or not choices:
            return None
        if isinstance(answer, str):
            normalized = answer.strip()
            if normalized in self.ANSWER_LABELS:
                return normalized
            if normalized.isdigit():
                answer = int(normalized)
            else:
                try:
                    idx = choices.index(answer)
                    return self.ANSWER_LABELS[idx]
                except ValueError:
                    return None
        try:
            idx = int(answer)
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(choices):
            return self.ANSWER_LABELS[idx]
        return None

    def _format_answer_text(self, answer, choices) -> str:
        if choices:
            label = self._resolve_choice_label(answer, choices)
            if label is None:
                raise ValueError("Could not resolve answer label for few-shot example.")
            return label
        if answer is None:
            raise ValueError("Answer missing for few-shot example.")
        return str(answer)

    def _build_question_block(
        self,
        question: str,
        choices=None,
        quote_question: bool = True,
    ) -> str:
        q_text = f"\"{question}\"" if quote_question else str(question)
        if choices:
            q_text = (
                f"{q_text}\nWhich one of these available options is correct answer? "
                f"{self._format_choices(choices)}"
            )
        return q_text 

    def _build_example_block(
        self,
        question: str,
        choices,
        answer,
        quote_question: bool = True,
    ) -> str:
        q_text = f"\"{question}\"" if quote_question else str(question)
        if choices:
            q_text = (
                f"{q_text}\nWhich one of these available options is correct answer? "
                f"{self._format_choices(choices)}"
            )
        answer_text = self._format_answer_text(answer, choices)
        return f"Question: {q_text}\nYour Answer: {answer_text}"
    

    def _build_prompt(
        self,
        question_block: str,
        few_shot_blocks: list[str] | None = None,
        shot_mode: str = "zero-shot",
    ) -> str:
        parts = []
        if few_shot_blocks:
            parts.append(self.FEW_SHOT_INTRO)
            parts.extend(few_shot_blocks)
            parts.append(self.NEW_PROBLEM_PREFIX)
        else:
            parts.append(self.ZERO_SHOT_PREFIX)
        
        parts.append(f"Question: {question_block}")
        parts.append(self.FINAL_ANSWER_INSTRUCTION)
        return "\n\n".join(parts)

    def _sample_few_shots(
        self,
        ds,
        indices: list[int],
        current_index: int,
        rng: random.Random,
        num_shots: int = 3,
    ):
        candidates = [i for i in indices if i != current_index]
        if not candidates:
            return []
        if len(candidates) <= num_shots:
            sample_indices = candidates
        else:
            sample_indices = rng.sample(candidates, num_shots)
        return [ds[i] for i in sample_indices]

    def _extract_answer(self, row: dict, keys: list[str]):
        for key in keys:
            if key in row:
                return row[key]
        return None


def run_benchmark(
    model: LLM,
    benchmark_label: str,
    shot_mode: str = "zero-shot",
    personas: bool = True,
    few_shot_seed: int = 42,
    num_shots: int = 3,
) -> list[dict]:
    runner = BenchmarkRunner(
        model,
        personas=personas,
        shot_mode=shot_mode,
        few_shot_seed=few_shot_seed,
        num_shots=num_shots,
    )
    return runner.run_benchmark(benchmark_label, shot_mode=shot_mode)

