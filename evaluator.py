"""Evaluation module for benchmark accuracy and confidence metrics.

This module provides tools to compute accuracy metrics and confidence scores
(via logprobs) from benchmark evaluation results. It handles multiple choice
label formats, dataset-specific answer extraction, and 10-run aggregation.
"""

from datasets import load_dataset

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except ImportError:
    # Fallback stopwords if NLTK not available
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'i', 'me', 'my', 'we', 'you', 'your'
    }

import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

ds_collection = {
    "mmlu": load_dataset("cais/mmlu", "all", split="dev"),
    "mmlu_pro":  load_dataset("TIGER-Lab/MMLU-Pro", split="validation"),
    "gpqa": load_dataset("Idavidrein/gpqa", "gpqa_main", split="train"),
}


@dataclass
class QuestionMetrics:
    """Metrics for a single question evaluation."""
    extracted_answer: str
    correct: bool
    answer_logprob_mean: Optional[float] = None
    content_logprob_mean: Optional[float] = None


@dataclass
class RunMetrics:
    """Aggregated metrics for a single run."""
    subject_accuracy: Dict[str, float] = field(default_factory=dict)
    global_accuracy: float = 0.0
    answer_logprob_mean: Optional[float] = None
    content_logprob_mean: Optional[float] = None
    num_questions: int = 0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple runs."""
    accuracy: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, Any] = field(default_factory=dict)


class AnswerExtractor:
    """Extract final answers from model outputs with fallback chain."""

    ANSWER_LABELS = "ABCDEFGHIJ"
    GENERIC_ANSWER_PATTERNS = (
        r"\\boxed\s*{\s*([^{}]+?)\s*}",
        r"\\\(\s*([^()\n]+?)\s*\\\)",
        r"\$\s*([^$\n]+?)\s*\$",
        r"^\s*\*{1,2}\s*([^\n*]+?)\s*\*{0,2}\s*$",
        r"^\s*([^\s\"'`]+?)\s*[\*\.\"]+\s*$",
        r"^\s*([^\s]+)\s*$",
    )

    def __init__(self):
        """Initialize the answer extractor."""
        pass

    def extract(
        self,
        output_text: str,
        golden: str,
        question: str,
        dataset: str = "mmlu",
        finish_reason: Optional[str] = None,
    ) -> tuple[str, str]:
        """Extract answer from model output with fallback chain.

        Parameters
        ----------
        output_text : str
            The model's output text (full text or output_text field).
        golden_answer : Any
            The golden/correct answer for reference.
        choices : list of str, optional
            Available choices for multiple-choice datasets.
        dataset : str
            Dataset name for dataset-specific extraction ("mmlu", "gpqa", "aime", etc.).
        finish_reason : str, optional
            Reason generation stopped ("stop", "length", etc.).

        Returns
        -------
        str
            Extracted answer or fallback value ("UNFINISHED_OUTPUT", "N.D.").
        """

        if dataset == 'aime':
            answer = self._extract_aime_answer(output_text)
        else:
            answer = self._extract_generic_answer(output_text)
    
        if answer:
            if (dataset == 'aime' or len(answer) == 1) and answer.isalnum(): # check if label was found (except for aime)
                return ("generic", answer.strip())

        answer = self._extract_golden_answer(dataset, question, golden, output_text) # we only check golden answer because if it's notthe given one then is a mistake

        if answer:
            return ("golden", answer)
        else:
            return ("fallback", self._fallback(finish_reason))
    

    def get_logprobs(self, answer: str, logprobs: List) -> Tuple:
        """Extract the logprobs of the extracted answer token and return the list of logprobs of generated text without stopwords and punctuation"""

        stop_words = set(stopwords.words('english'))
        punctuation_set = set(string.punctuation)
        cleaned_logprobs = [c for c in logprobs if c[0] not in punctuation_set and c[0].lower().strip() not in stop_words]
        
        target_answer = answer.strip()

        if target_answer.startswith('N.D.'):
            answer_logprob = logprobs[-1][1] # Fallback
        else:
            answer_logprob = next((c[1] for c in reversed(logprobs) if c[0].strip() == target_answer), None)

        # Multi-token reconstruction
        if answer_logprob is None and len(target_answer) > 1:
            
            # Helper function to normalize strings for robust comparison
            def normalize(text: str) -> str:
                text = text.replace('\\', '')
                text = re.sub(r'\s+', ' ', text)
                return text.strip()

            # NEW: Super-normalization that strips ALL whitespace for strict length/match checks
            def global_strip(text: str) -> str:
                text = text.replace('\\', '')
                return re.sub(r'\s+', '', text)

            normalized_target = normalize(target_answer)
            flat_target = global_strip(target_answer)

            # Step backwards through the logprobs to find the STARTing token of the match
            for start_idx in range(len(logprobs) - 1, -1, -1):
                accumulated_text = ""
                accumulated_text_stripped = ""
                accumulated_logprobs = []
                
                # Scan FORWARD from this start position to piece the tokens together
                for current_idx in range(start_idx, len(logprobs)):
                    token, lp = logprobs[current_idx]
                    
                    accumulated_text_stripped += token.strip()
                    accumulated_text += token
                    accumulated_logprobs.append(lp)
                    
                    # Standard normalized versions
                    norm_accumulated = normalize(accumulated_text)
                    norm_accumulated_stripped = normalize(accumulated_text_stripped)
                    
                    # NEW: Compare using standard normalization OR global whitespace removal
                    if (normalized_target in [norm_accumulated, norm_accumulated_stripped] or 
                        flat_target in [global_strip(accumulated_text), global_strip(accumulated_text_stripped)]):
                        
                        answer_logprob = sum(accumulated_logprobs) / len(accumulated_logprobs)
                        break
                    
                    # Check lengths using the space-free version to prevent premature truncation
                    if len(global_strip(accumulated_text_stripped)) > len(flat_target):
                        break
                
                # If a match was found in the forward scan, break the outer loop
                if answer_logprob is not None:
                    break

        if answer_logprob is None:
            print(f"Couldn't extract the logprob for answer {answer} and logprobs: {logprobs}")
            raise Exception
        else:
            return answer_logprob, cleaned_logprobs
        
    def _extract_golden_answer(self, dataset: str, question: str, answer: str, text: str) -> Optional[str]:
        """If no label is found, look for actual content of the answer. 
        If it's not found, it's either not present or wrongly answered.
        "answer" here is the golden answer provided by the dataset
        """
        
        answer = str(answer)

        # If the answer given is a label, convert it to the full string answer
        if dataset not in ['aime', 'gpqa']:
            ds = ds_collection[dataset]

            row_index = ds['question'].index(question) 

            if dataset == 'mmlu_pro': 
                choice_index = ord(answer.lower()) - ord('a') 
                answer = ds['options'][row_index][choice_index]
            else:
                # Assumes answer is already a 0-indexed integer or integer-string
                choice_index = int(answer)
                answer = ds['choices'][row_index][choice_index]

        # --- REGEX SEARCH FOR THE STRINGS ---
        
        # Escape the answer string so regex doesn't choke on special characters (like ?, (, ), etc.)
        escaped_answer = re.escape(answer)
        
        # Define patterns looking specifically for your answer string
        # We use \b for word boundaries so we don't accidentally match substrings
        patterns = [
            # 1. Trova "pick" dentro o dopo una parola (es. Cyclepick{answer})
            rf"pick({escaped_answer})\b",
            
            # 2. **Answer**: {answer}
            rf"\*\*(?:my\s+)?answer\s*[:\-]?\s*\*\*\s*({escaped_answer})\b",
            
            # 3. **Answer: {answer}**
            rf"\*\*(?:my\s+)?answer\s*[:\-]?\s*({escaped_answer})\*\*",
            
            # 4. Answer: **{answer}**
            rf"(?:my\s+)?answer\s*[:\-]?\s*\*({escaped_answer})\*\*",
            
            # 5. Answer: {answer}
            rf"(?:my\s+)?answer\s*[:\-]\s*({escaped_answer})\b",
            
            # 6. Answer {answer}
            rf"\b(?:my\s+)?answer\b\s*({escaped_answer})\b",
            
            # 7. er: {answer} (gestione troncamenti)
            rf"\ber\s*:\s*({escaped_answer})\b",
            
            # 8. Riga che inizia con : {answer}
            rf"^\s*:\s*({escaped_answer})\b",
            
            # 9. \boxed{{answer}}
            rf"\\boxed\{{({escaped_answer})\}}",
            
            # 10. correct answer: {answer}
            rf"\bcorrect\s+answer\s*[:\-]?\s*({escaped_answer})\b",
            
            # 11. correct choice: {answer}
            rf"\bcorrect\s+choice\s*[:\-]?\s*({escaped_answer})\b",
            
            # 12. correct answer is {answer}
            rf"\bcorrect\s+answer\s+is\s+({escaped_answer})\b",
            
            # 13. choose {answer}
            rf"\bchoose\s+({escaped_answer})\b",

            # 14. FALLBACK: check if escaped_answer is present within the last 50+len(escaped_answer) characters 
            rf"\b({escaped_answer})\b(?=.{{0,50}}(?:\n|$))",
        ]

        # Clean text to remove basic formatting/line break artifacts
        cleaned_text = re.sub(r'\\', '', text)
        
        # To find the LAST match in the document, we scan line-by-line from bottom to top
        candidates = [*reversed(cleaned_text.splitlines()), cleaned_text]
        
        for candidate_text in candidates:
            if not candidate_text or not candidate_text.strip():
                continue

            for pattern in patterns:
                # Case-insensitive search
                match = re.search(pattern, candidate_text, re.IGNORECASE)
                if match:
                    # Successfully found the exact string in a valid context!
                    extracted = match.group(1).strip()
                    return self._clean_answer_candidate(extracted)

        return None


    def _extract_aime_answer(self, text: str) -> Optional[str]:
            """Extract a likely final numerical answer from common wrapper and punctuation patterns."""

            # Pre-process text to remove source artifacts and repair split lines
            cleaned_text = re.sub(r'\\', '', text)
            cleaned_text = re.sub(r'\bmy\s*\n\s*answer\b', 'my answer', cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'\b(my\s+)?answer\s*\n\s*', r'\1answer', cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'(\banswer\s*[:\-]\s*)\n\s*', r'\1', cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'(\banswer\s*\*\*\s*[:\-]\s*)\n\s*', r'\1', cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'(\banswer\s*[:\-]\s*\*\*\s*)\n\s*', r'\1', cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'(\banswer\s*\*\*\s*[:\-]\s*\*\*\s*)\n\s*', r'\1', cleaned_text, flags=re.IGNORECASE)

            patterns = [
                # 1. Trova "pick" seguito da un numero intero (es. pick42 o pick 100)
                r"pick\s*\b([0-9]+)\b",
                
                # 2. **Answer**: 42
                r"\*\*(?:my\s+)?answer\s*[:\-]?\s*\*\*\s*\b([0-9]+)\b",
                
                # 3. **Answer: 42**
                r"\*\*(?:my\s+)?answer\s*[:\-]?\s*\b([0-9]+)\*\*",
                
                # 4. Answer: **42**
                r"(?:my\s+)?answer\s*[:\-]?\s*\*\b([0-9]+)\*\*",
                
                # 5. Answer: 42
                r"(?:my\s+)?answer\s*[:\-]\s*\b([0-9]+)\b",
                
                # 6. Answer 42
                r"\b(?:my\s+)?answer\b\s*\b([0-9]+)\b",
                
                # 7. er: 42 (gestione troncamenti)
                r"\ber\s*:\s*\b([0-9]+)\b",
                
                # 8. Riga che inizia con : 42
                r"^\s*:\s*\b([0-9]+)\b",
                
                # 9. \boxed{42}
                r"boxed\{([0-9]+)\}",
                
                # 10. correct answer: 42
                r"\bcorrect\s+answer\s*[:\-]?\s*\b([0-9]+)\b",
                
                # 11. correct choice: 42
                r"\bcorrect\s+choice\s*[:\-]?\s*\b([0-9]+)\b",
                
                # 12. correct answer is 42
                r"\bcorrect\s+answer\s+is\s+\b([0-9]+)\b",
                
                # 13. choose 42
                r"\bchoose\s+\b([0-9]+)\b",
            ]
            
            # Prioritize checking line-by-line from the bottom to get the final answer,
            # then fall back to scanning the entire text block.
            candidates = [*reversed(cleaned_text.splitlines()), cleaned_text]
            
            for candidate_text in candidates:
                if not candidate_text or not candidate_text.strip():
                    continue

                for pattern in patterns:
                    match = re.search(pattern, candidate_text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        candidate = match.group(1).strip()
                        # Strip any lingering markdown formatting characters or trailing punctuation
                        candidate = candidate.strip("*").strip("_").strip().strip(".")
                        candidate = self._clean_answer_candidate(candidate)
                        if candidate:
                            return candidate

            return None
        

    def _extract_generic_answer(self, text: str) -> Optional[str]:
        """Extract a likely final answer from common wrapper and punctuation patterns."""

        # Pre-process text to remove source artifacts and repair split lines
        cleaned_text = re.sub(r'\\', '', text)
        cleaned_text = re.sub(r'\bmy\s*\n\s*answer\b', 'my answer', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'\b(my\s+)?answer\s*\n\s*', r'\1answer', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'(\banswer\s*[:\-]\s*)\n\s*', r'\1', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'(\banswer\s*\*\*\s*[:\-]\s*)\n\s*', r'\1', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'(\banswer\s*[:\-]\s*\*\*\s*)\n\s*', r'\1', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'(\banswer\s*\*\*\s*[:\-]\s*\*\*\s*)\n\s*', r'\1', cleaned_text, flags=re.IGNORECASE)

        patterns = [
            # 1. Trova "pick" seguito direttamente da UN SOLO carattere alfanumerico (es. CyclepickA)
            # Usiamo \b alla fine per assicurarci che dopo la lettera/cifra non ci siano altre lettere/cifre
            r"pick([A-La-l0-9])\b",
            
            # 2. **Answer**: A
            r"\*\*(?:my\s+)?answer\s*[:\-]?\s*\*\*\s*([A-La-l0-9])\b",
            
            # 3. **Answer: A** (Qui togliamo il + in modo che dentro gli asterischi ci sia solo il singolo carattere)
            r"\*\*(?:my\s+)?answer\s*[:\-]?\s*([A-La-l0-9])\*\*",
            
            # 4. Answer: **A**
            r"(?:my\s+)?answer\s*[:\-]?\s*\*([A-La-l0-9])\*\*",
            
            # 5. Answer: A
            r"(?:my\s+)?answer\s*[:\-]\s*([A-La-l0-9])\b",
            
            # 6. Answer A
            r"\b(?:my\s+)?answer\b\s*([A-La-l0-9])\b",
            
            # 7. er: A (gestione troncamenti)
            r"\ber\s*:\s*([A-La-l0-9])\b",
            
            # 8. Riga che inizia con : A
            r"^\s*:\s*([A-La-l0-9])\b",
            
            # 9. \boxed{A} (Rimosso il + in modo che dentro le graffe ci sia solo una lettera o una cifra)
            r"\\boxed\{([A-La-l0-9])\}",
            
            # 10. correct answer: A
            r"\bcorrect\s+answer\s*[:\-]?\s*([A-La-l0-9])\b",
            
            # 11. correct choice: A
            r"\bcorrect\s+choice\s*[:\-]?\s*([A-La-l0-9])\b",
            
            # 12. correct answer is A
            r"\bcorrect\s+answer\s+is\s+([A-La-l0-9])\b",
            
            # 13. choose A
            r"\bchoose\s+([A-La-l0-9])\b",
        ]
        
        # Prioritize checking line-by-line from the bottom to get the final answer,
        # then fall back to scanning the entire text block.
        candidates = [*reversed(cleaned_text.splitlines()), cleaned_text]
        
        for candidate_text in candidates:
            if not candidate_text or not candidate_text.strip():
                continue

            for pattern in patterns:
                match = re.search(pattern, candidate_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    candidate = match.group(1).strip()
                    # Strip any lingering markdown formatting characters or trailing punctuation
                    candidate = candidate.strip("*").strip("_").strip().strip(".")
                    candidate = self._clean_answer_candidate(candidate)
                    if candidate:
                        return candidate

        return None

    def _clean_answer_candidate(self, candidate: str) -> str:
        """Remove leftover wrappers from a captured answer candidate."""
        cleaned = candidate.strip()
        cleaned = re.sub(r"^[\s\*\$`'\"\(\)\[\]\{\}\\]+", "", cleaned)
        cleaned = re.sub(r"[\s\*\$`'\"\(\)\[\]\{\}\\]+$", "", cleaned)
        return cleaned.strip()

    def _fallback(self, finish_reason: Optional[str]) -> str:
        """Return fallback value based on finish_reason."""
        if finish_reason and finish_reason != "stop":
            return "N.D. LENGHT"
        return "N.D."

 
class AccuracyComputer:
    """Compute accuracy metrics per run and aggregated across runs."""

    def __init__(self, extractor: Optional[AnswerExtractor] = None):
        """Initialize the accuracy computer.

        Parameters
        ----------
        extractor : AnswerExtractor, optional
            Answer extractor instance. If None, creates a new one.
        """
        self.extractor = extractor or AnswerExtractor()

    def compute_run_accuracy(
        self,
        run_results: List[Dict[str, Any]],
        dataset: str = "mmlu",
    ) -> Tuple[RunMetrics, List[QuestionMetrics]]:
        """Compute accuracy metrics for a single run.

        Parameters
        ----------
        run_results : list of dict
            Results from one benchmark run with keys: subject, question, answer, output, index.
        dataset : str
            Dataset name for answer extraction.

        Returns
        -------
        run_metrics : RunMetrics
            Aggregated metrics for the run.
        question_metrics : list of QuestionMetrics
            Per-question metrics.
        """
        question_metrics_list = []
        subject_counts = defaultdict(lambda: {"correct": 0, "total": 0})
        total_correct = 0
        total_questions = 0

        for item in run_results:
            subject = item.get("subject", "unknown")
            golden_answer = item.get("answer")
            output_dict = item.get("output", {})

            # Use output_text if available (from reasoning), else use full text
            output_text = output_dict.get("output_text") or output_dict.get("text", "")
            finish_reason = output_dict.get("finish_reason")

            # Get choices if available
            choices = item.get("choices") or item.get("options")

            # Extract answer
            extracted_answer = self.extractor.extract(
                output_text=output_text,
                golden_answer=golden_answer,
                choices=choices,
                dataset=dataset,
                finish_reason=finish_reason,
            )

            # Check correctness
            is_correct = self._check_correctness(extracted_answer, golden_answer, choices, dataset)

            # Update metrics
            question_metrics = QuestionMetrics(
                extracted_answer=extracted_answer,
                correct=is_correct,
            )
            question_metrics_list.append(question_metrics)

            subject_counts[subject]["total"] += 1
            if is_correct:
                subject_counts[subject]["correct"] += 1

            total_questions += 1
            if is_correct:
                total_correct += 1

        # Compute subject accuracies
        subject_accuracy = {}
        for subject, counts in subject_counts.items():
            subject_accuracy[subject] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0

        global_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

        run_metrics = RunMetrics(
            subject_accuracy=subject_accuracy,
            global_accuracy=global_accuracy,
            num_questions=total_questions,
        )

        return run_metrics, question_metrics_list

    def _check_correctness(
        self,
        extracted_answer: str,
        golden_answer: Any,
        choices: Optional[List[str]],
        dataset: str,
    ) -> bool:
        """Check if extracted answer matches golden answer."""
        if extracted_answer in ("N.D.", "UNFINISHED_OUTPUT"):
            return False

        if dataset.lower() == "aime":
            # For AIME, compare as integers
            try:
                extracted_int = int(extracted_answer)
                golden_int = int(golden_answer) if isinstance(golden_answer, (int, str)) else None
                return extracted_int == golden_int
            except (ValueError, TypeError):
                return False

        # For multiple choice datasets
        if choices:
            extracted_lower = str(extracted_answer).lower().strip()

            # Try matching against choice label
            for i, choice in enumerate(choices):
                choice_str = str(choice).strip()
                label = AnswerExtractor.ANSWER_LABELS[i] if i < len(AnswerExtractor.ANSWER_LABELS) else str(i + 1)

                # Check if extracted matches label or content
                if extracted_lower == label.lower() or extracted_lower == choice_str.lower():
                    # Now check if this matches golden
                    golden_lower = str(golden_answer).lower().strip()
                    if golden_lower == label.lower() or golden_lower == choice_str.lower():
                        return True
            return False

        # Fallback: string comparison
        return str(extracted_answer).lower() == str(golden_answer).lower()

    def aggregate_runs(self, runs_metrics: List[RunMetrics]) -> Dict[str, Any]:
        """Aggregate metrics across multiple runs.

        Parameters
        ----------
        runs_metrics : list of RunMetrics
            Metrics from each of the 10 runs.

        Returns
        -------
        dict
            Aggregated metrics with per_run lists and mean values.
        """
        if not runs_metrics:
            return {}

        # Collect per-run accuracies
        per_run_global = [m.global_accuracy for m in runs_metrics]
        per_run_subject = [m.subject_accuracy for m in runs_metrics]

        # Compute mean accuracy
        mean_accuracy = sum(per_run_global) / len(per_run_global) if per_run_global else 0.0

        # Compute per-subject mean accuracy
        all_subjects = set()
        for subject_dict in per_run_subject:
            all_subjects.update(subject_dict.keys())

        per_subject_mean = {}
        for subject in all_subjects:
            values = [
                m.subject_accuracy.get(subject, 0.0)
                for m in runs_metrics
                if subject in m.subject_accuracy
            ]
            if values:
                per_subject_mean[subject] = sum(values) / len(values)

        return {
            "per_run": per_run_global,
            "per_run_subject": per_run_subject,
            "mean_accuracy": mean_accuracy,
            "per_subject": per_subject_mean,
        }



    def compute_run_accuracy_json(self, run_results: List[Dict[str, Any]], dataset: str = "mmlu") -> Dict[str, Any]:
        """Compute accuracy metrics for a single run using the custom JSON structure and dataset-specific logic."""
        total_correct = 0
        total_questions = 0
        nd_count = 0
        nd_length_count = 0
        subject_counts = defaultdict(lambda: {"correct": 0, "total": 0})
        
        dataset_lower = dataset.lower().strip()
        
        for item in run_results:
            subject = item.get("subject", "unknown")
            golden_raw = item.get("golden_label")
            extracted_raw = item.get("extracted_answer")
            
            # Ensure safe string formatting for text analysis
            extracted_str = str(extracted_raw).strip() if extracted_raw is not None else ""
            extracted_lower = extracted_str.lower()
            
            # 1. Primi controlli per i fallimenti di estrazione (Sempre errati)
            if extracted_lower == "n.d.":
                nd_count += 1
                is_correct = False
            elif extracted_lower == "n.d. lenght":
                nd_length_count += 1
                is_correct = False
            else:
                # 2. Logica condizionale in base al dataset fornito
                if dataset_lower == "aime":
                    # Confronto tra stringa ed intero
                    try:
                        is_correct = int(float(extracted_str)) == int(golden_raw)
                    except (ValueError, TypeError):
                        is_correct = extracted_str.strip() == str(golden_raw).strip()
                        
                elif dataset_lower == "gpqa":
                    # Corretto se estratto è "A" oppure se ha lunghezza maggiore di 1
                    is_correct = (extracted_str == "A") or (len(extracted_str) > 1)
                    
                elif dataset_lower == "mmlu":
                    # Mappatura indici numerici (0->A, 1->B, ...)
                    try:
                        golden_index = int(golden_raw)
                        # Genera lettere dinamiche a seconda dell'indice
                        expected_letter = chr(65 + golden_index) if 0 <= golden_index < 26 else ""
                    except (ValueError, TypeError):
                        expected_letter = str(golden_raw).strip().upper()
                        
                    if len(extracted_str) == 1 and extracted_str.isalpha():
                        is_correct = extracted_str.upper() == expected_letter
                    else:
                        # Se non è una singola lettera, allora è corretto per fallback
                        is_correct = True
                        
                elif dataset_lower == "mmlu_pro":
                    # Entrambe le label sono lettere, se l'estratto non è una singola lettera è corretto
                    if len(extracted_str) == 1 and extracted_str.isalpha():
                        is_correct = extracted_str.upper() == str(golden_raw).strip().upper()
                    else:
                        is_correct = True
                else:
                    # Fallback generico per altri dataset non mappati
                    is_correct = extracted_lower == str(golden_raw).strip().lower()
            
            total_questions += 1
            subject_counts[subject]["total"] += 1
            if is_correct:
                total_correct += 1
                subject_counts[subject]["correct"] += 1
                
        global_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        nd_percentage = nd_count / total_questions if total_questions > 0 else 0.0
        nd_length_percentage = nd_length_count / total_questions if total_questions > 0 else 0.0
        
        subject_accuracy = {
            subj: (counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0)
            for subj, counts in subject_counts.items()
        }
        
        return {
            "global_accuracy": global_accuracy,
            "subject_accuracy": subject_accuracy,
            "nd_percentage": nd_percentage,
            "nd_length_percentage": nd_length_percentage
        }

class ConfidenceComputer:
    """Compute confidence metrics from logprobs."""

    def __init__(self, extractor: Optional[AnswerExtractor] = None):
        """Initialize the confidence computer.

        Parameters
        ----------
        extractor : AnswerExtractor, optional
            Answer extractor instance for extracting answer tokens.
        """
        self.extractor = extractor or AnswerExtractor()

    def compute_question_confidence(
        self,
        logprobs: Optional[List[Dict[str, Any]]],
        extracted_answer: str,
        output_text: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute confidence metrics for a single question.

        Parameters
        ----------
        logprobs : list of dict, optional
            List of {"token_id": int, "token": str, "logprob": float}.
        extracted_answer : str
            The extracted answer text.
        output_text : str
            Full output text.

        Returns
        -------
        answer_logprob_mean : float or None
            Mean logprob for answer tokens.
        content_logprob_mean : float or None
            Mean logprob for content tokens (excluding stopwords).
        """
        if not logprobs:
            return None, None

        # Extract answer tokens
        answer_logprobs = self._get_answer_logprobs(logprobs, extracted_answer)
        answer_logprob_mean = self._compute_mean_logprob(answer_logprobs) if answer_logprobs else None

        # Extract content tokens (excluding stopwords)
        content_logprobs = self._get_content_logprobs(logprobs)
        content_logprob_mean = self._compute_mean_logprob(content_logprobs) if content_logprobs else None

        return answer_logprob_mean, content_logprob_mean

    def _get_answer_logprobs(
        self,
        logprobs: List[Dict[str, Any]],
        answer: str,
    ) -> List[float]:
        """Extract logprobs for tokens that match the answer."""
        if not answer or answer in ("N.D.", "UNFINISHED_OUTPUT"):
            return []

        answer_tokens = answer.lower().split()
        answer_logprobs = []

        for logprob_entry in logprobs:
            token = logprob_entry.get("token", "").lower().strip()
            logprob = logprob_entry.get("logprob")

            # Match token against answer tokens
            for answer_token in answer_tokens:
                if token == answer_token or answer_token in token or token in answer_token:
                    if logprob is not None:
                        answer_logprobs.append(logprob)
                    break

        return answer_logprobs

    def _get_content_logprobs(self, logprobs: List[Dict[str, Any]]) -> List[float]:
        """Extract logprobs for content tokens (excluding stopwords)."""
        content_logprobs = []

        for logprob_entry in logprobs:
            token = logprob_entry.get("token", "").lower().strip()
            logprob = logprob_entry.get("logprob")

            # Skip stopwords and special tokens
            if token and token not in STOPWORDS and not token.startswith("<") and logprob is not None:
                content_logprobs.append(logprob)

        return content_logprobs

    def _compute_mean_logprob(self, logprobs: List[float]) -> Optional[float]:
        """Compute mean of logprobs."""
        if not logprobs:
            return None
        return sum(logprobs) / len(logprobs)

    def compute_run_confidence(
        self,
        run_results: List[Dict[str, Any]],
        question_metrics_list: List[QuestionMetrics],
    ) -> Tuple[List[Tuple[Optional[float], Optional[float]]], Optional[float], Optional[float]]:
        """Compute confidence metrics for a run.

        Parameters
        ----------
        run_results : list of dict
            Results from one benchmark run.
        question_metrics_list : list of QuestionMetrics
            Per-question metrics (with extracted_answer).

        Returns
        -------
        per_question_confidence : list of (answer_logprob_mean, content_logprob_mean)
            Confidence metrics for each question.
        run_answer_logprob_mean : float or None
            Mean logprob for answer tokens across all questions.
        run_content_logprob_mean : float or None
            Mean logprob for content tokens across all questions.
        """
        per_question_confidence = []
        all_answer_logprobs = []
        all_content_logprobs = []

        for item, metrics in zip(run_results, question_metrics_list):
            output_dict = item.get("output", {})
            logprobs = output_dict.get("logprobs")
            output_text = output_dict.get("output_text") or output_dict.get("text", "")

            answer_logprob_mean, content_logprob_mean = self.compute_question_confidence(
                logprobs=logprobs,
                extracted_answer=metrics.extracted_answer,
                output_text=output_text,
            )

            metrics.answer_logprob_mean = answer_logprob_mean
            metrics.content_logprob_mean = content_logprob_mean

            per_question_confidence.append((answer_logprob_mean, content_logprob_mean))

            # Collect for per-run mean
            if answer_logprob_mean is not None:
                all_answer_logprobs.append(answer_logprob_mean)
            if content_logprob_mean is not None:
                all_content_logprobs.append(content_logprob_mean)

        # Compute per-run means
        run_answer_mean = (
            sum(all_answer_logprobs) / len(all_answer_logprobs)
            if all_answer_logprobs
            else None
        )
        run_content_mean = (
            sum(all_content_logprobs) / len(all_content_logprobs)
            if all_content_logprobs
            else None
        )

        return per_question_confidence, run_answer_mean, run_content_mean

    def aggregate_runs(self, runs_metrics: List[RunMetrics]) -> Dict[str, Any]:
        """Aggregate confidence metrics across multiple runs.

        Parameters
        ----------
        runs_metrics : list of RunMetrics
            Metrics from each of the 10 runs.

        Returns
        -------
        dict
            Aggregated confidence metrics with per_run values and means.
        """
        if not runs_metrics:
            return {}

        # Collect per-run values
        per_run_answer = [
            m.answer_logprob_mean for m in runs_metrics if m.answer_logprob_mean is not None
        ]
        per_run_content = [
            m.content_logprob_mean for m in runs_metrics if m.content_logprob_mean is not None
        ]

        # Compute aggregate means
        answer_logprob_mean = (
            sum(per_run_answer) / len(per_run_answer) if per_run_answer else None
        )
        content_logprob_mean = (
            sum(per_run_content) / len(per_run_content) if per_run_content else None
        )

        return {
            "answer_logprob_mean_per_run": per_run_answer,
            "content_logprob_mean_per_run": per_run_content,
            "answer_logprob_mean": answer_logprob_mean,
            "content_logprob_mean": content_logprob_mean,
        }



    def compute_run_confidence_json(self, run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute confidence metrics for a single run using the custom JSON structure."""
        extracted_logprobs = []
        complete_logprob_means = []
        subject_data = defaultdict(lambda: {"extracted_logprobs": [], "complete_logprob_means": []})
        
        for item in run_results:
            subject = item.get("subject", "unknown")
            
            ext_lp = item.get("extracted_logprob")
            if ext_lp is not None:
                extracted_logprobs.append(ext_lp)
                subject_data[subject]["extracted_logprobs"].append(ext_lp)
                
            lps = item.get("logprobs", [])
            if lps:
                lp_values = [pair[1] for pair in lps if len(pair) > 1 and isinstance(pair[1], (int, float))]
                if lp_values:
                    mean_lp = sum(lp_values) / len(lp_values)
                    complete_logprob_means.append(mean_lp)
                    subject_data[subject]["complete_logprob_means"].append(mean_lp)
                    
        mean_extracted_logprob = sum(extracted_logprobs) / len(extracted_logprobs) if extracted_logprobs else None
        
        subject_confidence = {}
        for subj, data in subject_data.items():
            subj_mean_ext_lp = sum(data["extracted_logprobs"]) / len(data["extracted_logprobs"]) if data["extracted_logprobs"] else None
            subject_confidence[subj] = {
                "extracted_logprob_mean": subj_mean_ext_lp,
                "complete_logprob_means": data["complete_logprob_means"]
            }
            
        return {
            "extracted_logprob_mean": mean_extracted_logprob,
            "complete_logprob_means": complete_logprob_means,
            "subject_confidence": subject_confidence
        }

class BenchmarkEvaluator:
    """Main evaluator orchestrating accuracy and confidence computation."""

    def __init__(
        self,
        dataset: str = "mmlu",
        extractor: Optional[AnswerExtractor] = None,
    ):
        """Initialize the evaluator.

        Parameters
        ----------
        dataset : str
            Dataset name for answer extraction ("mmlu", "gpqa", "aime", etc.).
        extractor : AnswerExtractor, optional
            Custom answer extractor. If None, creates new instance.
        """
        self.dataset = dataset
        self.extractor = extractor or AnswerExtractor()
        self.accuracy_computer = AccuracyComputer(self.extractor)
        self.confidence_computer = ConfidenceComputer(self.extractor)

    def evaluate_results(
        self,
        results: List[List[Dict[str, Any]]],
        compute_confidence: bool = True,
    ) -> AggregatedMetrics:
        """Evaluate benchmark results across all runs.

        Parameters
        ----------
        results : list of list of dict
            Results structure: list of 10 runs, each run is list of question results.
        compute_confidence : bool
            If True, compute confidence metrics. If False, only compute accuracy.

        Returns
        -------
        AggregatedMetrics
            Aggregated accuracy and confidence metrics.
        """
        all_runs_metrics = []
        all_question_metrics_per_run = []

        for run_results in results:
            # Compute accuracy
            run_metrics, question_metrics_list = self.accuracy_computer.compute_run_accuracy(
                run_results, dataset=self.dataset
            )
            all_runs_metrics.append(run_metrics)
            all_question_metrics_per_run.append(question_metrics_list)

            # Compute confidence if requested
            if compute_confidence:
                per_question_conf, answer_mean, content_mean = self.confidence_computer.compute_run_confidence(
                    run_results, question_metrics_list
                )
                run_metrics.answer_logprob_mean = answer_mean
                run_metrics.content_logprob_mean = content_mean

        # Aggregate metrics across runs
        accuracy_metrics = self.accuracy_computer.aggregate_runs(all_runs_metrics)
        confidence_metrics = (
            self.confidence_computer.aggregate_runs(all_runs_metrics)
            if compute_confidence
            else {}
        )

        return AggregatedMetrics(
            accuracy=accuracy_metrics,
            confidence=confidence_metrics,
        )



    def evaluate_json_results(self, all_runs_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Evaluate custom JSON benchmark results across all 5 runs."""
        output_metrics = []
        for run_results in all_runs_results:
            acc_metrics = self.accuracy_computer.compute_run_accuracy_json(run_results, dataset=self.dataset)
            conf_metrics = self.confidence_computer.compute_run_confidence_json(run_results)
            
            run_aggregator = {
                "accuracy": {
                    "mean_accuracy": acc_metrics["global_accuracy"],
                    "accuracy_per_subject": acc_metrics["subject_accuracy"],
                    "nd_percentage": acc_metrics["nd_percentage"],
                    "nd_length_percentage": acc_metrics["nd_length_percentage"]
                },
                "confidence_score": {
                    "mean_extracted_logprob": conf_metrics["extracted_logprob_mean"],
                    "complete_logprob_means": conf_metrics["complete_logprob_means"],
                    "per_subject": conf_metrics["subject_confidence"]
                }
            }
            output_metrics.append(run_aggregator)
        return output_metrics

def should_skip_evaluation(
    evaluation_file: Path,
    has_logprobs: bool = False,
) -> bool:
    """Check if evaluation file should be skipped.

    File is skipped only if:
    - File exists
    - Accuracy metrics are populated
    - If has_logprobs=True, confidence metrics are also populated

    Parameters
    ----------
    evaluation_file : Path
        Path to evaluation file.
    has_logprobs : bool
        If True, also check that confidence metrics are populated.

    Returns
    -------
    bool
        True if file should be skipped, False otherwise.
    """
    if not evaluation_file.exists():
        return False

    try:
        import json

        with open(evaluation_file) as f:
            data = json.load(f)

        # Check if metrics structure exists
        if "metrics" not in data or "accuracy" not in data["metrics"]:
            return False

        accuracy = data["metrics"].get("accuracy", {})

        # Check if accuracy is populated
        if not accuracy.get("per_run") or not isinstance(accuracy.get("per_run"), list):
            return False

        # If has_logprobs, also check confidence
        if has_logprobs:
            confidence = data["metrics"].get("confidence", {})
            if not confidence.get("answer_logprob_mean_per_run"):
                return False

        return True
    except Exception:
        return False