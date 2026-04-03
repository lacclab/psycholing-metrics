"""Inverse effect extractor: measures how much context reduces surprisal.

Computes the difference between surprisal without context and surprisal with context,
keeping only cases where context actually reduced surprisal (positive effect).
"""

import pandas as pd

from psycholing_metrics.surprisal.base import BaseSurprisalExtractor
from psycholing_metrics.surprisal.types import SurprisalExtractorType
from psycholing_metrics.text_processing import aggregate_token_log_probs


class InverseEffectExtractor(BaseSurprisalExtractor):
    """Compute the inverse effect of context on surprisal.

    For each word, computes: max(0, surp(word|no_context) - surp(word|context)).
    This measures how much the context helps predict each word.
    """

    def __init__(
        self,
        model_name: str,
        extractor_type_name: str,
        target_extractor_type: SurprisalExtractorType,
        model_target_device: str = "cpu",
        pythia_checkpoint: str | None = "step143000",
        hf_access_token: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            extractor_type_name=extractor_type_name,
            model_target_device=model_target_device,
            pythia_checkpoint=pythia_checkpoint,
            hf_access_token=hf_access_token,
        )

        from psycholing_metrics.surprisal.factory import create_surprisal_extractor

        self.target_extractor = create_surprisal_extractor(
            model_name=model_name,
            extractor_type=target_extractor_type,
            model_target_device=model_target_device,
            pythia_checkpoint=pythia_checkpoint,
            hf_access_token=hf_access_token,
        )

    def compute_surprisal(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = False,
    ):
        baseline_surp = self.target_extractor.compute_surprisal(
            target_text=target_text,
            left_context_text=None,
            overlap_size=overlap_size,
            allow_overlap=allow_overlap,
        )
        other_surp = self.target_extractor.compute_surprisal(
            target_text=target_text,
            left_context_text=left_context_text,
            overlap_size=overlap_size,
            allow_overlap=allow_overlap,
        )

        dataframe_probs_baseline = pd.DataFrame(
            aggregate_token_log_probs(target_text, baseline_surp[0], baseline_surp[1])[
                1
            ],
            columns=["Word", "Surprisal"],
        )

        dataframe_probs_other = pd.DataFrame(
            aggregate_token_log_probs(target_text, other_surp[0], other_surp[1])[1],
            columns=["Word", "Surprisal"],
        )

        other_surp_col = dataframe_probs_other["Surprisal"]
        baseline_surp_col = dataframe_probs_baseline["Surprisal"]

        surp_diff = other_surp_col - baseline_surp_col
        surp_diff[surp_diff > 0] = 0
        surp_diff = -surp_diff  # the negative diffs are now positive
        baseline_surp_col += surp_diff
        dataframe_probs_baseline["Surprisal"] = baseline_surp_col

        return dataframe_probs_baseline
