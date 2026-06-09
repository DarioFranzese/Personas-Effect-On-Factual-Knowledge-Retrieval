"""Evaluation module for benchmark accuracy and confidence metrics.

This module provides tools to compute accuracy metrics and confidence scores
(via logprobs) from benchmark evaluation results. It handles multiple choice
label formats, dataset-specific answer extraction, and 10-run aggregation.
"""

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

    ANSWER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
        golden_answer: Any,
        choices: Optional[List[str]] = None,
        dataset: str = "mmlu",
        finish_reason: Optional[str] = None,
    ) -> str:
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
            if answer:
                return answer.strip()

        # Step 1.2: Look for My Answer: {answer}
        answer = self._extract_my_answer_pattern(output_text)
        if answer:
            if dataset in ['aime', 'gpqa'] or (len(answer) == 1 and answer.isalnum()): # the extracted answer must refer to the multiple choises
                return answer.strip()


        # Step 1.2: Choice matching for multiple-choice datasets
        if choices:
            answer = self._extract_choice_answer(output_text, choices)
            if answer and len(answer) == 1 and answer.isalnum():
                return answer.strip()

        # Step 1.3: Generic regex fallbacks for wrapped / punctuated answers
        answer = self._extract_generic_answer(output_text)
        if answer:
            if dataset in ['aime', 'gpqa'] or (len(answer) == 1 and answer.isalnum()): # the extracted answer must refer to the multiple choises
                return answer.strip()
            else:
                print(f"\n\n\n### *OUTPUT*: {output_text[(len(output_text)//2):]}")

        # Step 1.4: Fallback based on finish_reason
        return self._fallback(finish_reason)
    
    def _extract_aime_answer(self, text: str) -> Optional[str]:
        """Extract integer answer from AIME output (integers only, no decimals)."""
        # Look for integers in the output, searching from the end
        integers = re.findall(r'(?<!\d)(?<!\w)\d+(?!\w)(?!\d)', text)
        if integers:
            # Return the last integer found (most likely the answer)
            return integers[-1]
        
    def _extract_my_answer_pattern(self, text: str) -> Optional[str]:
        """Extract answer from 'My Answer: {answer}' pattern."""
        pattern = r"my\s+answer\s*:\s*([A-Z])\)?"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_choice_answer(self, text: str, choices: Optional[List[str]]) -> Optional[str]:
        """Extract choice label from text by matching against available choices."""
        if not choices:
            return None

        text_lower = text.lower()
        choices_list = [str(c).strip() for c in choices]

        # Try matching choice labels (A, B, C, etc.)
        for i, choice in enumerate(choices_list):
            label = self.ANSWER_LABELS[i] if i < len(self.ANSWER_LABELS) else str(i + 1)

            # Search for label in output (backward to prioritize end of text)
            for search_pos in range(len(text_lower) - 1, max(len(text_lower) - 500, 0), -1):
                if text_lower[search_pos : search_pos + len(label)].lower() == label.lower():
                    return label
                if text_lower[search_pos : search_pos + len(choice)].lower() == choice.lower():
                    return label

        # Try matching choice content anywhere in output
        for i, choice in enumerate(choices_list):
            label = self.ANSWER_LABELS[i] if i < len(self.ANSWER_LABELS) else str(i + 1)
            if choice.lower() in text_lower:
                return label

        return None

    def _extract_generic_answer(self, text: str) -> Optional[str]:
        """Extract a likely final answer from common wrapper and punctuation patterns."""
        for candidate_text in [text, *reversed(text.splitlines())]:
            if not candidate_text or not candidate_text.strip():
                continue

            for pattern in self.GENERIC_ANSWER_PATTERNS:
                match = re.search(pattern, candidate_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    candidate = match.group(1).strip()
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
            return "UNFINISHED_OUTPUT"
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