"""Surprisal extraction via text-level concatenation of left context with target text.

This implements the standard (buggy) approach described in Pimentel & Meister (2024),
where the left context is concatenated as a string prefix before tokenization.
"""

from typing import List, Tuple

import numpy as np
import torch

from text_metrics.surprisal.base import BaseSurprisalExtractor
from text_metrics.text_processing import trim_left_context


class ConcatenatedSurprisalExtractor(BaseSurprisalExtractor):
    """Compute surprisal by concatenating left context and target text at the text level.

    The left context string is prepended to the target text with a space separator,
    then the full string is tokenized and passed through the model. Surprisal values
    are extracted only for the target text portion.
    """

    def __init__(
        self,
        model_name: str,
        extractor_type_name: str,
        model_target_device: str = "cpu",
        pythia_checkpoint: str | None = "step143000",
        hf_access_token: str | None = None,
    ):
        super().__init__(
            model_name,
            extractor_type_name,
            model_target_device,
            pythia_checkpoint,
            hf_access_token,
        )

    def _extract_target_from_full_context(
        self,
        target_text_char_onset: int,
        all_log_probs: torch.Tensor,
        offset_mapping: List[Tuple[int]],
    ):
        """Remove left context log probs, keeping only target text results.

        Finds where the target text begins in the offset mapping and slices
        both log probs and offsets to cover only the target text.
        """
        offset_mapping_first_index = [
            i
            for i, (start, end) in enumerate(offset_mapping)
            if start
            == target_text_char_onset
            - 1  # -1 because the first token includes the space
        ][0]

        target_text_log_probs = all_log_probs[offset_mapping_first_index:]

        target_text_offset_mapping = offset_mapping[offset_mapping_first_index:]
        target_text_offset_mapping = [
            (i - target_text_char_onset, j - target_text_char_onset)
            for i, j in target_text_offset_mapping
        ]

        assert target_text_log_probs.shape[0] == len(target_text_offset_mapping)

        return target_text_log_probs, target_text_offset_mapping

    def compute_surprisal(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = True,
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        """Compute surprisal for target text, optionally conditioned on left context.

        If left_context_text is provided, it is concatenated before the target text
        and passed through the model together. Surprisal values are returned only
        for the target text portion.

        :param target_text: the text to compute surprisal for.
        :param left_context_text: optional prefix context string.
        :param overlap_size: token overlap for chunking long texts.
        :param allow_overlap: whether to allow chunking.
        :return: (log_probs, offset_mapping) for the target text only.
        """
        if allow_overlap:
            assert overlap_size is not None, "overlap_size must be specified"

        if left_context_text in [None, ""]:
            return self.compute_surprisal_no_context(target_text, allow_overlap, overlap_size)

        with torch.no_grad():
            try:
                max_ctx = self.model.config.max_position_embeddings
            except AttributeError:
                max_ctx = int(1e6)

            left_context_text = trim_left_context(
                self.tokenizer,
                left_context_text=left_context_text,
                max_tokens=max_ctx,
            )

            full_context = left_context_text + " " + target_text
            target_text_char_onset = len(full_context) - len(target_text)

            assert overlap_size < max_ctx, (
                f"Stride size {overlap_size} is larger than the maximum context size {max_ctx}"
            )

            (
                all_log_probs,
                offset_mapping,
                accumulated_tokenized_text,
            ) = self._compute_log_probs_with_chunking(
                full_context, overlap_size, allow_overlap, max_ctx
            )

        assert (
            accumulated_tokenized_text
            == self.tokenizer(full_context, add_special_tokens=False)["input_ids"]
        )

        all_log_probs = np.asarray(all_log_probs.cpu())

        target_text_log_probs, target_text_offset_mapping = (
            self._extract_target_from_full_context(
                target_text_char_onset, all_log_probs, offset_mapping
            )
        )

        return target_text_log_probs, target_text_offset_mapping
