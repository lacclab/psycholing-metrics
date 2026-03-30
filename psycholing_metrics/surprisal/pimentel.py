"""Corrected surprisal extraction per Pimentel & Meister (2024).

Uses a Bag-of-Words language model approach to properly compute word probabilities,
fixing the issues with the standard concatenation approach.
"""

from typing import List, Tuple

import pandas as pd
import torch
from torch._tensor import Tensor

from psycholing_metrics.pimentel_word_prob.wordsprobability.main import agg_surprisal_per_word
from psycholing_metrics.pimentel_word_prob.wordsprobability.models import (
    MODELS,
    get_model,
)
from psycholing_metrics.pimentel_word_prob.wordsprobability.models.bow_lm import BaseBOWModel
from psycholing_metrics.surprisal.concatenated import ConcatenatedSurprisalExtractor
from psycholing_metrics.text_processing import trim_left_context


class PimentelSurprisalExtractor(ConcatenatedSurprisalExtractor):
    """Corrected surprisal computation per Pimentel & Meister (2024).

    Unlike other extractors, compute_surprisal() returns a DataFrame directly
    because the word-level aggregation process differs from the standard approach.
    Only supports models registered in the pimentel_word_prob MODELS registry.
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
        if model_name not in MODELS:
            raise ValueError(
                f"""Model name {model_name} is currently not supported for PimentelSurprisalExtractor,
                The supported models are: {list(MODELS.keys())}
                """
            )
        self.bow_model: BaseBOWModel = get_model(
            model_name=model_name, model=self.model, tokenizer=self.tokenizer
        )

    def _compute_log_probs_with_chunking(
        self, full_context: str, overlap_size: int, allow_overlap: bool, max_ctx: int
    ) -> Tuple[Tensor, List[Tuple[int]], List[Tensor]]:
        if allow_overlap:
            assert overlap_size is not None, "overlap_size must be specified"

        results, offsets = self.bow_model.get_predictions(
            sentence=full_context,
            use_bos_symbol=True,
            overlap_size=overlap_size,
        )

        res_df = pd.DataFrame(results)
        res_df["text_id"] = 0
        res_df["offsets"] = offsets

        surp_df = agg_surprisal_per_word(
            res_df, self.model_name, return_buggy_surprisals=False
        )

        return surp_df

    def _compute_surprisal_full_text(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = False,
    ) -> pd.DataFrame:
        """Compute surprisal for the full text (context + target) and return as DataFrame."""
        if allow_overlap:
            assert overlap_size is not None, "overlap_size must be specified"

        with torch.no_grad():
            try:
                max_ctx = self.model.config.max_position_embeddings
            except AttributeError:
                max_ctx = int(1e6)

            if left_context_text is not None and len(left_context_text) > 0:
                left_context_text = trim_left_context(
                    self.tokenizer,
                    left_context_text=left_context_text,
                    max_tokens=max_ctx,
                )

                full_context = left_context_text + " " + target_text
            else:
                full_context = target_text

            assert overlap_size < max_ctx, (
                f"Stride size {overlap_size} is larger than the maximum context size {max_ctx}"
            )

            dataframe_surps = self._compute_log_probs_with_chunking(
                full_context=full_context,
                overlap_size=overlap_size,
                allow_overlap=allow_overlap,
                max_ctx=512,
            ).reset_index(drop=True)

        return dataframe_surps

    def compute_surprisal(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = False,
    ) -> pd.DataFrame:
        """Compute corrected word-level surprisal values.

        :param target_text: the text to compute surprisal for.
        :param left_context_text: optional prefix context.
        :param overlap_size: token overlap for chunking.
        :param allow_overlap: whether to allow chunking.
        :return: DataFrame with 'Word' and 'Surprisal' columns for the target text.
        """
        dataframe_surps = self._compute_surprisal_full_text(
            target_text=target_text,
            left_context_text=left_context_text,
            overlap_size=overlap_size,
            allow_overlap=allow_overlap,
        )
        dataframe_surps.rename(
            columns={"word": "Word", "surprisal": "Surprisal"}, inplace=True
        )

        if left_context_text is not None and len(left_context_text) > 0:
            left_ctx_len_in_words = len(left_context_text.split())
            dataframe_surps = dataframe_surps.iloc[left_ctx_len_in_words:]

        return dataframe_surps.reset_index(drop=True)
