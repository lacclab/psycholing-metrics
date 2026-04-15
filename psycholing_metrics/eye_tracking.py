"""Integration of word-level metrics with eye-tracking data.

Functions for extracting linguistic metrics (surprisal, frequency, word length, parsing)
and merging them with eye-tracking interest area reports.
"""

import gc
from functools import partial
from typing import Dict, List, Literal, Tuple

import pandas as pd
import spacy
import torch
import tqdm
from spacy.language import Language

from psycholing_metrics.metrics import get_metrics
from psycholing_metrics.surprisal import base, factory, types


def create_text_input(
    row: pd.Series,
    text_col_name: str,
    prefix_col_names: List[str],
    suffix_col_names: List[str],
) -> Tuple[
    str, Tuple[int, int], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]
]:
    """Construct a single string from text, prefix, and suffix columns in a DataFrame row.

    :param row: a row from a DataFrame containing text columns.
    :param text_col_name: column name for the main text.
    :param prefix_col_names: column names for prefixes, added in order.
    :param suffix_col_names: column names for suffixes, added in order.
    :return: tuple of (full_text, main_text, prefix_text, suffix_text,
             main_text_word_indices, prefix_word_indices, suffix_word_indices)
    """
    text_input = ""
    curr_w_index = 0

    prefix_text = ""
    prefixes_word_indices_ranges = {}
    for prefix_col in prefix_col_names:
        curr_prefix = getattr(row, prefix_col)
        if (curr_prefix is not None) and (len(curr_prefix) > 0):
            addition_to_acc_text = curr_prefix + " "
            text_input += addition_to_acc_text
            prefix_text += addition_to_acc_text
        curr_prefix_len = len(curr_prefix.split())
        next_w_index = curr_w_index + curr_prefix_len
        prefixes_word_indices_ranges[prefix_col] = (
            curr_w_index,
            next_w_index - 1,
        )
        curr_w_index = next_w_index

    prefix_text = prefix_text[:-1]  # remove the last space

    row_main_text = getattr(row, text_col_name).strip()
    row_main_text_len = len(row_main_text.split())
    text_input += row_main_text
    main_text_word_indices = (
        curr_w_index,
        curr_w_index + row_main_text_len - 1,
    )
    curr_w_index += row_main_text_len - 1

    suffix_text = ""
    suffixes_word_indices_ranges = {}
    if len(suffix_col_names) > 0:
        text_input += " "
        for i, suffix_col in enumerate(suffix_col_names):
            curr_suffix_text = getattr(row, suffix_col)
            addition_to_acc_text = (
                curr_suffix_text + " "
                if i < len(suffix_col_names) - 1
                else curr_suffix_text
            )
            text_input += addition_to_acc_text
            suffix_text += addition_to_acc_text

            curr_suffix_len = len(curr_suffix_text.split())
            suffixes_word_indices_ranges[suffix_col] = (
                curr_w_index,
                curr_w_index + curr_suffix_len - 1,
            )
            curr_w_index += curr_suffix_len

    return (
        text_input,
        row_main_text,
        prefix_text,
        suffix_text,
        main_text_word_indices,
        prefixes_word_indices_ranges,
        suffixes_word_indices_ranges,
    )


def extract_metrics_for_text_df(
    text_df: pd.DataFrame,
    text_col_name: str,
    text_key_cols: List[str],
    surp_extractor: base.BaseSurprisalExtractor,
    ordered_prefix_col_names: List[str] = [],
    ordered_suffix_col_names: List[str] = [],
    get_metrics_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Extract word-level metrics for each text in a DataFrame.

    :param text_df: DataFrame where each row has text and identifying columns.
    :param text_col_name: column containing the main text string.
    :param text_key_cols: columns that uniquely identify each text.
    :param surp_extractor: the surprisal extractor to use.
    :param ordered_prefix_col_names: columns with prefix text, prepended in order.
    :param ordered_suffix_col_names: columns with suffix text, appended in order.
    :param get_metrics_kwargs: additional keyword arguments for get_metrics().
    :return: DataFrame where each row is a word with its metrics and text identifiers.
    """
    get_metrics_kwargs = {} if get_metrics_kwargs is None else get_metrics_kwargs.copy()
    metric_dfs = []
    for row in tqdm.tqdm(
        text_df.reset_index().itertuples(),
        total=len(text_df),
        desc="Extracting metrics",
    ):
        if len(ordered_prefix_col_names) > 0 or len(ordered_suffix_col_names) > 0:
            (
                _text_input,
                main_text,
                prefix_text,
                _suffix_text,
                _main_text_word_indices,
                _prefixes_word_indices_ranges,
                _suffixes_word_indices_ranges,
            ) = create_text_input(
                row, text_col_name, ordered_prefix_col_names, ordered_suffix_col_names
            )
        else:
            main_text = getattr(row, text_col_name).strip()
            prefix_text = ""

        merged_df = get_metrics(
            target_text=main_text.strip(),
            left_context_text=prefix_text,
            surp_extractor=surp_extractor,
            **get_metrics_kwargs,
        )
        merged_df.reset_index(inplace=True)

        merged_df[text_key_cols] = [getattr(row, key_col) for key_col in text_key_cols]

        metric_dfs.append(merged_df)

    return pd.concat(metric_dfs, axis=0)


def extract_metrics_for_multiple_models(
    text_df: pd.DataFrame,
    text_col_name: str,
    text_key_cols: List[str],
    surprisal_extraction_model_names: List[str],
    surp_extractor_types: (
        types.SurprisalExtractorType | List[types.SurprisalExtractorType]
    ) = types.SurprisalExtractorType.CAT_CTX_LEFT,
    add_parsing_features: bool = True,
    parsing_mode: (
        Literal["keep-first", "keep-all", "re-tokenize"] | None
    ) = "re-tokenize",
    spacy_model: Language | None = spacy.load("en_core_web_sm"),
    model_target_device: str = "cpu",
    hf_access_token: str | None = None,
    extract_metrics_kwargs: dict | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Extract word-level metrics using multiple HuggingFace language models and extractor types.

    Iterates over all (model, extractor type) combinations, creating a surprisal extractor
    for each and extracting metrics. Each combination produces a uniquely named surprisal
    column (e.g., ``gpt2_cat_Surprisal``, ``gpt2_pimentel_Surprisal``). Non-surprisal
    metrics (frequency, length, parsing) are only computed once with the first combination.

    :param text_df: DataFrame where each row has text and identifying columns.
    :param text_col_name: column containing the main text string.
    :param text_key_cols: columns that uniquely identify each text.
    :param surprisal_extraction_model_names: HuggingFace model names to extract surprisal from.
    :param surp_extractor_types: extraction strategy or list of strategies to use.
        When a list is provided, surprisal is extracted for every (model, type) combination.
    :param add_parsing_features: whether to include spaCy parsing features.
    :param parsing_mode: tokenization alignment strategy for spaCy.
    :param spacy_model: the spaCy model instance.
    :param model_target_device: device for the models.
    :param hf_access_token: HuggingFace access token for gated models.
    :param extract_metrics_kwargs: additional kwargs for extract_metrics_for_text_df().
    :param save_path: if provided, save intermediate results to this CSV path.
    :return: DataFrame with word-level metrics from all models and extractor types.
    """
    assert not (
        add_parsing_features is True and (parsing_mode is None or spacy_model is None)
    ), (
        "If add_parsing_features is True, both parsing_mode and spacy_model must be provided"
    )

    if isinstance(surp_extractor_types, types.SurprisalExtractorType):
        surp_extractor_types = [surp_extractor_types]

    if extract_metrics_kwargs is None:
        extract_metrics_kwargs = {}

    get_metrics_kwargs = {}
    if "get_metrics_kwargs" in extract_metrics_kwargs:
        get_metrics_kwargs = extract_metrics_kwargs["get_metrics_kwargs"]
        del extract_metrics_kwargs["get_metrics_kwargs"]
    get_metrics_kwargs["parsing_model"] = spacy_model
    get_metrics_kwargs["parsing_mode"] = parsing_mode

    metric_df = None
    is_first = True
    for model_name in surprisal_extraction_model_names:
        for surp_extractor_type in surp_extractor_types:
            try:
                if is_first:
                    print("Extracting Frequency, Length")
                print(
                    f"Extracting surprisal using model: {model_name}, "
                    f"type: {surp_extractor_type.name}"
                )

                surp_extractor = factory.create_surprisal_extractor(
                    extractor_type=surp_extractor_type,
                    model_name=model_name,
                    model_target_device=model_target_device,
                    hf_access_token=hf_access_token,
                )

                get_metrics_kwargs["add_parsing_features"] = (
                    True if is_first and add_parsing_features else False
                )
                metric_dfs = extract_metrics_for_text_df(
                    text_df=text_df,
                    text_col_name=text_col_name,
                    text_key_cols=text_key_cols,
                    surp_extractor=surp_extractor,
                    get_metrics_kwargs=get_metrics_kwargs,
                    **extract_metrics_kwargs,
                )

                if metric_df is None:
                    metric_df = metric_dfs.copy()
                else:
                    concatenated_metric_dfs = metric_dfs.copy()
                    cols_to_merge = concatenated_metric_dfs.columns.difference(
                        metric_df.columns
                    ).tolist()
                    cols_to_merge += text_key_cols + ["index"]

                    metric_df = metric_df.merge(
                        concatenated_metric_dfs[cols_to_merge],
                        how="left",
                        on=text_key_cols + ["index"],
                        validate="one_to_one",
                    )

                if save_path is not None:
                    _save_aggregated(metric_df, save_path, text_key_cols)

                del surp_extractor
                gc.collect()
                torch.cuda.empty_cache()
                is_first = False

            except Exception as e:
                print(f"Error for {model_name} with {surp_extractor_type.name}: {e}")

    return metric_df


def _save_aggregated(df: pd.DataFrame, save_path: str, groupby_cols: List[str]):
    """Save aggregated statistics to CSV."""
    text_df_w_metrics = df.drop(
        columns=["Word", "Length", "Wordfreq_Frequency", "subtlex_Frequency"]
    )
    mean_surp_df = (
        text_df_w_metrics.groupby(groupby_cols)
        .agg(
            {
                col: ["mean", "max", "min", "std", "count", "median"]
                for col in text_df_w_metrics.columns
                if col not in groupby_cols
            }
        )
        .reset_index()
    )
    mean_surp_df.columns = [
        "_".join(col).strip("_") for col in mean_surp_df.columns.values
    ]
    mean_surp_df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")


def add_metrics_to_eye_tracking_report(
    eye_tracking_data: pd.DataFrame,
    surprisal_extraction_model_names: List[str],
    textual_item_key_cols: List[str],
    spacy_model_name: str,
    surp_extractor_types: (
        types.SurprisalExtractorType | List[types.SurprisalExtractorType]
    ),
    parsing_mode: Literal["keep-first", "keep-all", "re-tokenize"],
    model_target_device: str = "cpu",
    hf_access_token: str | None = None,
) -> pd.DataFrame:
    """Add word-level metrics to an eye-tracking interest area report.

    Extracts unique textual items from the eye-tracking data, computes metrics
    for each word, and merges them back into the original DataFrame.

    :param eye_tracking_data: DataFrame with eye-tracking data (must have 'IA_ID' and 'IA_LABEL' columns).
    :param surprisal_extraction_model_names: HuggingFace model names for surprisal extraction.
    :param textual_item_key_cols: columns that identify unique textual items.
    :param spacy_model_name: name of the spaCy model to load.
    :param surp_extractor_types: extraction strategy or list of strategies to use.
    :param parsing_mode: tokenization alignment strategy for spaCy.
    :param model_target_device: device for the models.
    :param hf_access_token: HuggingFace access token for gated models.
    :return: the eye_tracking_data DataFrame enriched with word-level metrics.
    """

    without_duplicates = eye_tracking_data[
        textual_item_key_cols
        + [
            "IA_ID",
            "IA_LABEL",
        ]
    ].drop_duplicates()

    text_from_et = without_duplicates.groupby(textual_item_key_cols)["IA_LABEL"].apply(
        list
    )

    text_from_et = text_from_et.apply(lambda text: " ".join(text))

    spacy_model = spacy.load(spacy_model_name)

    extract_metrics_partial = partial(
        extract_metrics_for_multiple_models,
        text_col_name="IA_LABEL",
        text_key_cols=textual_item_key_cols,
        surprisal_extraction_model_names=surprisal_extraction_model_names,
        surp_extractor_types=surp_extractor_types,
        parsing_mode=parsing_mode,
        spacy_model=spacy_model,
        model_target_device=model_target_device,
        hf_access_token=hf_access_token,
    )
    metric_df = extract_metrics_partial(
        text_df=text_from_et,
        extract_metrics_kwargs=dict(
            ordered_prefix_col_names=[],
        ),
    )

    metric_df = metric_df.rename({"index": "IA_ID", "Word": "IA_LABEL"}, axis=1)

    et_data_enriched = eye_tracking_data.merge(
        metric_df,
        how="left",
        on=textual_item_key_cols
        + [
            "IA_ID",
        ],
        validate="many_to_one",
    )

    return et_data_enriched
