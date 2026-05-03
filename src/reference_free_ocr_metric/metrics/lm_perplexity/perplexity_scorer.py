"""LM Perplexity-based OCR quality metric using character n-grams and GPT-2."""

import math
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reference_free_ocr_metric.metrics.base import BaseMetric


class LMPerplexityMetric(BaseMetric):
    """OCR quality metric based on language model perplexity.

    Lower perplexity indicates more linguistically plausible text,
    which suggests better OCR quality. Combines character n-gram
    self-consistency with GPT-2 perplexity scoring.
    """

    def __init__(self, transformer_model: str = "gpt2") -> None:
        """Initialize with a HuggingFace causal-LM model name (default: gpt2)."""
        self._model_name = transformer_model
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        """Return the metric identifier string."""
        return "lm_perplexity"

    def _load_model(self) -> None:
        """Lazy-load the transformer model and tokenizer."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model.eval()
        self._model.to(device)

    def _ngram_score(self, text: str, n: int = 4) -> float:
        """Compute character n-gram self-consistency score.

        More natural text has repeated n-gram patterns; garbled text
        tends to have abnormally high unique-to-total ratio.

        Returns:
            Score in [0, 1] where higher = more natural.
        """
        if len(text) < n:
            return 0.5

        ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
        total_ngrams = len(ngrams)
        unique_ngrams = len(set(ngrams))

        return 1.0 - (unique_ngrams / total_ngrams)

    def _transformer_perplexity(self, text: str) -> float:
        """Compute GPT-2 perplexity for the given text.

        For texts longer than the model's max sequence length (1024 tokens),
        uses a sliding window with stride to average negative log-likelihoods
        across chunks.

        Returns:
            Perplexity value (lower = better). Returns inf for empty text.
        """
        if not text:
            return float("inf")

        if self._model is None:
            self._load_model()

        device = next(self._model.parameters()).device
        encodings = self._tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)
        max_len = self._model.config.n_positions  # 1024 for GPT-2

        if seq_len <= max_len:
            with torch.no_grad():
                outputs = self._model(input_ids, labels=input_ids)
                return torch.exp(outputs.loss).item()

        # Sliding window for long texts
        stride = max_len // 2
        nlls = []
        n_tokens = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_len, seq_len)
            chunk_ids = input_ids[:, begin:end]
            target_ids = chunk_ids.clone()
            # Mask tokens in the overlap region to avoid double-counting
            if begin > 0:
                target_ids[:, : end - begin - stride] = -100
            with torch.no_grad():
                outputs = self._model(chunk_ids, labels=target_ids)
            valid_tokens = (target_ids != -100).sum().item()
            nlls.append(outputs.loss.item() * valid_tokens)
            n_tokens += valid_tokens
            if end == seq_len:
                break

        avg_nll = sum(nlls) / n_tokens
        return math.exp(avg_nll)

    def compute(self, ocr_output: str, **kwargs: Any) -> float:
        """Compute composite quality score for OCR output.

        Returns:
            Float in [0, 1] where higher = better quality.
        """
        result = self.compute_detailed(ocr_output)
        return result["composite"]

    def compute_detailed(self, text: str) -> Dict[str, float]:
        """Compute detailed quality breakdown.

        Returns:
            Dict with ngram_score, transformer_score, perplexity, and composite.
        """
        ngram_score = self._ngram_score(text)
        perplexity = self._transformer_perplexity(text)
        transformer_score = 1.0 / (1.0 + math.log(max(perplexity, 1.0)))
        composite = 0.3 * ngram_score + 0.7 * transformer_score

        return {
            "ngram_score": ngram_score,
            "transformer_score": transformer_score,
            "perplexity": perplexity,
            "composite": composite,
        }
