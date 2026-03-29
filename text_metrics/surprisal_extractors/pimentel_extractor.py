from typing import List, Tuple

import pandas as pd
import torch
from torch._tensor import Tensor

from text_metrics.pimentel_word_prob.wordsprobability.main import agg_surprisal_per_word
from text_metrics.pimentel_word_prob.wordsprobability.models import (
    MODELS,
    get_model,
)
from text_metrics.pimentel_word_prob.wordsprobability.models.bow_lm import BaseBOWModel
from text_metrics.surprisal_extractors.text_cat_extractor import CatCtxLeftSurpExtractor
from text_metrics.utils import remove_redundant_left_context


class PimentelSurpExtractor(CatCtxLeftSurpExtractor):
    """Unlike for the other extractors, here `surprise` doesnt return logits and offsetts,
    because the aggregation process is different. Thus, surprise returns the final surp-per-word dataframe.

    Args:
        CatCtxLeftSurpExtractor (_type_): _description_
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
                f"""Model name {model_name} is currently not supported for PimentelSurpExtractor,
                The supported models are: {list(MODELS.keys())}
                """
            )
        self.bow_model: BaseBOWModel = get_model(
            model_name=model_name, model=self.model, tokenizer=self.tokenizer
        )

    def _full_context_to_probs_and_offsets(
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

    def surprise_full_text(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = False,
    ) -> pd.DataFrame:
        """Get the surprisal values for each word in the target text.

        Args:
            target_text (str): the text to get surprisal values for.
            left_context_text (str | None, optional): the left context to consider. Defaults to None.
            overlap_size (int | None, optional): the overlap size to use when considering the left context. Defaults to None.
            allow_overlap (bool, optional): whether to allow overlap between the left context and the target text. Defaults to False.

        Returns:
            pd.DataFrame: the surprisal values for each word in the target text.
        """
        if allow_overlap:
            assert overlap_size is not None, "overlap_size must be specified"

        with torch.no_grad():
            try:
                max_ctx = self.model.config.max_position_embeddings
            except AttributeError:
                max_ctx = int(1e6)

            if left_context_text is not None and len(left_context_text) > 0:
                # than the max context, we remove the redundant left context tokens that cannot be used in-context with the target text
                left_context_text = remove_redundant_left_context(
                    self.tokenizer,
                    left_context_text=left_context_text,
                    max_ctx_in_tokens=max_ctx,
                )

                full_context = left_context_text + " " + target_text
            else:
                full_context = target_text

            assert overlap_size < max_ctx, (
                f"Stride size {overlap_size} is larger than the maximum context size {max_ctx}"
            )

            dataframe_surps = self._full_context_to_probs_and_offsets(
                full_context=full_context,
                overlap_size=overlap_size,
                allow_overlap=allow_overlap,
                max_ctx=512,
            ).reset_index(drop=True)

        return dataframe_surps

    def surprise(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = False,
    ) -> pd.DataFrame:
        dataframe_surps = self.surprise_full_text(
            target_text=target_text,
            left_context_text=left_context_text,
            overlap_size=overlap_size,
            allow_overlap=allow_overlap,
        )
        dataframe_surps.rename(
            columns={"word": "Word", "surprisal": "Surprisal"}, inplace=True
        )

        if left_context_text is not None and len(left_context_text) > 0:
            # remove the records that are not part of the target text
            left_ctx_len_in_words = len(left_context_text.split())
            dataframe_surps = dataframe_surps.iloc[left_ctx_len_in_words:]

        return dataframe_surps.reset_index(drop=True)
