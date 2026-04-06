"""Metric extraction for tabular text data (one word per row)."""

from typing import List

import pandas as pd
import tqdm

from psycholing_metrics.metrics import get_metrics
from psycholing_metrics.surprisal.factory import create_surprisal_extractor
from psycholing_metrics.surprisal.types import SurprisalExtractorType


def add_metrics_to_tabular_text(
    tabular_text: pd.DataFrame,
    surprisal_extraction_model_names: List[str],
    surp_extractor_types: (
        SurprisalExtractorType | List[SurprisalExtractorType]
    ) = SurprisalExtractorType.CAT_CTX_LEFT,
    model_target_device: str = "cpu",
) -> pd.DataFrame:
    """Add word-level metrics to a tabular text DataFrame.

    Each row in the input represents a word from a trial. Words are grouped by 'item'
    and reassembled into sentences for metric extraction, then merged back.

    :param tabular_text: DataFrame with columns ['item', 'wordnum', 'word'].
    :param surprisal_extraction_model_names: HuggingFace model names for surprisal.
    :param surp_extractor_types: extraction strategy or list of strategies to use.
    :param model_target_device: device for the models.
    :return: the input DataFrame enriched with surprisal, frequency, and word length columns.
    """
    if isinstance(surp_extractor_types, SurprisalExtractorType):
        surp_extractor_types = [surp_extractor_types]

    grouped_text = tabular_text.groupby(["item"])["word"].apply(list)
    grouped_text = grouped_text.apply(lambda text: " ".join(text))

    metric_dfs = []
    for model_name in surprisal_extraction_model_names:
        for surp_extractor_type in surp_extractor_types:
            surp_extractor = create_surprisal_extractor(
                extractor_type=surp_extractor_type,
                model_name=model_name,
                model_target_device=model_target_device,
            )
            for index, sentence in tqdm.tqdm(
                grouped_text.items(),
                total=len(grouped_text),
                desc=f"Extracting metrics ({model_name}, {surp_extractor_type.name})",
            ):
                merged_df = get_metrics(
                    target_text=sentence,
                    surp_extractor=surp_extractor,
                    parsing_model=None,
                    parsing_mode=None,
                    add_parsing_features=False,
                )
                merged_df["item"] = index
                merged_df.reset_index(inplace=True)
                merged_df["index"] += 1
                merged_df = merged_df.rename({"index": "wordnum"}, axis=1)
                metric_dfs.append(merged_df)

    metric_df = pd.concat(metric_dfs, axis=0)

    tabular_text_enriched = tabular_text.merge(
        metric_df,
        how="left",
        suffixes=("", "_metric"),
        on=["item", "wordnum"],
        validate="many_to_one",
    )

    tabular_text_enriched.drop(["Word"], axis=1, inplace=True)
    tabular_text_enriched["subtlex_Frequency"] = tabular_text_enriched[
        "subtlex_Frequency"
    ].replace(0, "NA")
    return tabular_text_enriched


if __name__ == "__main__":
    stim = pd.read_csv("stim.csv", keep_default_na=False)
    tabular_text_enriched = add_metrics_to_tabular_text(stim, ["gpt2"])
    tabular_text_enriched.to_csv("stim_with_surprisal.csv", index=False)
