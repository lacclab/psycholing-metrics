from typing import List

import pandas as pd
import tqdm

from text_metrics.ling_metrics_funcs import get_metrics
from text_metrics.surprisal_extractors.extractor_switch import get_surp_extractor
from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType


def add_metrics_tabular_text(
    tabular_text: pd.DataFrame,
    surprisal_extraction_model_names: List[str],
    surp_extractor_type: SurpExtractorType = SurpExtractorType.CAT_CTX_LEFT,
    model_target_device: str = "cpu",
) -> pd.DataFrame:
    """
    Adds metrics to each row in the tabular_text DataFrame.

    :param tabular_text: The input DataFrame with tabular text data, where each row represents
        a word that was read in a given trial. Should have columns - ['item', 'wordnum', 'word']
    :param surprisal_extraction_model_names: The names of the models to extract surprisal values from.
    :param surp_extractor_type: The type of surprisal extractor to use. Defaults to CAT_CTX_LEFT.
    :param model_target_device: The device to run the model on. Defaults to "cpu".
    :return: The tabular_text DataFrame with added columns for surprisal, frequency, and word length metrics.
    """

    # Group by item and join all words
    grouped_text = tabular_text.groupby(["item"])["word"].apply(list)
    grouped_text = grouped_text.apply(lambda text: " ".join(text))

    metric_dfs = []
    for model_name in surprisal_extraction_model_names:
        surp_extractor = get_surp_extractor(
            extractor_type=surp_extractor_type,
            model_name=model_name,
            model_target_device=model_target_device,
        )
        for index, sentence in tqdm.tqdm(
            grouped_text.items(), total=len(grouped_text),
            desc=f"Extracting metrics ({model_name})",
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
    tabular_text_enriched = add_metrics_tabular_text(stim, ["gpt2"])
    tabular_text_enriched.to_csv("stim_with_surprisal.csv", index=False)
