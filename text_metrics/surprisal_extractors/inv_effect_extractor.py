import pandas as pd

from text_metrics.surprisal_extractors.base_extractor import BaseSurprisalExtractor
from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType
from text_metrics.utils import string_to_log_probs


class InvEffectExtractor(BaseSurprisalExtractor):
    def __init__(
        self,
        model_name: str,
        extractor_type_name: str,
        target_extractor_type: SurpExtractorType,
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

        from text_metrics.surprisal_extractors.extractor_switch import (
            get_surp_extractor,
        )

        self.target_extractor = get_surp_extractor(
            model_name=model_name,
            extractor_type=target_extractor_type,
            model_target_device=model_target_device,
            pythia_checkpoint=pythia_checkpoint,
            hf_access_token=hf_access_token,
        )

    def surprise(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = False,
    ):
        baseline_surp = self.target_extractor.surprise(
            target_text=target_text,
            left_context_text=None,
            overlap_size=overlap_size,
            allow_overlap=allow_overlap,
        )
        other_surp = self.target_extractor.surprise(
            target_text=target_text,
            left_context_text=left_context_text,
            overlap_size=overlap_size,
            allow_overlap=allow_overlap,
        )

        dataframe_probs_baseline = pd.DataFrame(
            string_to_log_probs(target_text, baseline_surp[0], baseline_surp[1])[1],
            columns=["Word", "Surprisal"],
        )

        dataframe_probs_other = pd.DataFrame(
            string_to_log_probs(target_text, other_surp[0], other_surp[1])[1],
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
