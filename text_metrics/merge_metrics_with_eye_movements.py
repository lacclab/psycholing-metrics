import gc
from functools import partial
from typing import Tuple, Dict, List, Literal
import pandas as pd
import tqdm
import spacy
from text_metrics.utils import break_down_p_id
from spacy.language import Language
import torch
from text_metrics.ling_metrics_funcs import get_metrics
from text_metrics.surprisal_extractors import (
    base_extractor,
    extractors_constants,
    extractor_switch,
)


def create_text_input(
    row: pd.Series,
    text_col_name: str,
    prefix_col_names: List[str],
    suffix_col_names: List[str],
) -> Tuple[
    str, Tuple[int, int], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]
]:
    """This function creates a single string from the text columns in the row

    Args:
        row (pd.Series): A row from a dataframe that contains text columns
        text_col_name (str): The name of the column in row that contains the main text
        prefix_col_names (List[str]): Column names in row that contain prefixes to be added.
            Text will be added according to the order of the columns in the list
        suffix_col_names (List[str]): Column names in row that contain suffixes to be added.
            Text will be added according to the order of the columns in the list

    Returns:
        Tuple[ str, Tuple[int, int], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]] ]:
            A tuple containing:
            - The concatenated text from the columns in row
            - The word indices of the main text in the concatenated text
            - A dictionary with the word indices ranges of the prefixes in the concatenated text.
            - A dictionary with the word indices ranges of the suffixes in the concatenated text.
    """
    text_input = ""
    curr_w_index = 0

    # add prefixes and document their word indices
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

    # add suffixes and document their word indices
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
        text_input,  # full text containing prefixes, main text and suffixes
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
    surp_extractor: base_extractor.BaseSurprisalExtractor,
    ordered_prefix_col_names: List[str] = [],
    ordered_suffix_col_names: List[str] = [],
    get_metrics_kwargs: dict | None = None,
) -> pd.DataFrame:
    """This function extracts word level characteristics
    (Length, frequency, surprisal) from a text_df.
    This function allows adding prefixes and suffixes to
    the text and extract metrics for them as well (or omit them).

    Args:
        text_df (pd.DataFrame): A dataframe where each row has text identifying columns,
            main text column (string) and possibly prefix and suffix columns (strings)
        text_col_name (str): The name of the column in text_df that contains the main text
        text_key_cols (List[str]): The columns in text_df that identify the text
        surp_extractor (BaseSurprisalExtractor): The surprisal extractor to use
        ordered_prefix_col_names (List[str], optional): A list of column names in text_df
            that contain prefixes. The order in which they appear in the list is the order
            in which they will be added to the text. Defaults to [].
        ordered_suffix_col_names (List[str], optional): A list of column names in text_df
            that contain suffixes. The order in which they appear in the list is the order
            in which they will be added to the text. Defaults to [].
        get_metrics_kwargs (dict | None, optional): A dict of additional keyword arguments for
            the get_metrics function. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with the extracted metrics, now instead of each row being a
            whole text, each row is a word in the text. The columns are the main text
            identifier, word_index, word and extracted metrics.
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
                text_input,  # full text containing prefixes, main text and suffixes
                main_text,
                prefix_text,
                suffix_text,
                main_text_word_indices,
                prefixes_word_indices_ranges,
                suffixes_word_indices_ranges,
            ) = create_text_input(
                row, text_col_name, ordered_prefix_col_names, ordered_suffix_col_names
            )
        else:
            main_text = getattr(row, text_col_name).strip()
            prefix_text = ""

        # add here new metrics
        #! Note: for now, the get metrics function can accept only left context
        # Note: merged df contains only the main text metrics
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


def extract_metrics_for_text_df_multiple_hf_models(
    text_df: pd.DataFrame,
    text_col_name: str,
    text_key_cols: List[str],
    surprisal_extraction_model_names: List[str],
    surp_extractor_type: extractors_constants.SurpExtractorType = extractors_constants.SurpExtractorType.CAT_CTX_LEFT,
    add_parsing_features: bool = True,
    parsing_mode: (
        Literal["keep-first", "keep-all", "re-tokenize"] | None
    ) = "re-tokenize",
    spacy_model: Language | None = spacy.load("en_core_web_sm"),
    model_target_device: str = "cpu",
    hf_access_token: str | None = None,
    extract_metrics_for_text_df_kwargs: dict | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """This function extracts word level characteristics while extracting surprisal
        estimates from multiple huggingface models

    Args:
        text_df (pd.DataFrame): A dataframe where each row has text identifying columns,
            main text column (string) and possibly prefix and suffix columns (strings)
        text_col_name (str): The name of the column in text_df that contains the main text
        text_key_cols (List[str]): The columns in text_df that identify the text
        surprisal_extraction_model_names (List[str]): The name of the models to extract surprisal
            values from
        surp_extractor_type (extractors_constants.SurpExtractorType): The type of surprisal
            extractor to use (e.g. CAT_CTX_LEFT, SOFT_CAT_SENTENCES)
        add_parsing_features (bool, optional): If True, parsing features will be added to the
            extracted metrics. Defaults to True.
        parsing_mode (Literal[&quot;keep): Type of parsing to use. one of
            ['keep-first','keep-all','re-tokenize']. Defaults to "re-tokenize".
        spacy_model (Language): The spacy model to use for parsing the text.
            Defaults to spacy.load("en_core_web_sm").
        model_target_device (str, optional): The device to move the model to. Defaults to "cpu".
        hf_access_token (str, optional): A huggingface access token. Defaults to None.
        extract_metrics_for_text_df_kwargs (dict | None, optional): A dict of additional keyword
            arguments for the extract_metrics_for_text_df function. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with the extracted metrics, now instead of each row being a
            whole text, each row is a word in the text. The columns are the main text
            identifier, word_index, word and extracted metrics.
    """
    assert not (
        add_parsing_features is True and (parsing_mode is None or spacy_model is None)
    ), "If add_parsing_features is True, both parsing_mode and spacy_model must be provided"

    if extract_metrics_for_text_df_kwargs is None:
        extract_metrics_for_text_df_kwargs = {}

    # Check if get_metrics_kwargs is in extract_metrics_for_text_df_kwargs
    get_metrics_kwargs = {}
    if "get_metrics_kwargs" in extract_metrics_for_text_df_kwargs:
        get_metrics_kwargs = extract_metrics_for_text_df_kwargs["get_metrics_kwargs"]
        del extract_metrics_for_text_df_kwargs["get_metrics_kwargs"]
    get_metrics_kwargs["parsing_model"] = spacy_model
    get_metrics_kwargs["parsing_mode"] = parsing_mode

    metric_df = None
    for i, model_name in enumerate(surprisal_extraction_model_names):
        try:
            # log the model name
            if i == 0:
                print("Extracting Frequency, Length")
            print(f"Extracting surprisal using model: {model_name}")

            surp_extractor = extractor_switch.get_surp_extractor(
                extractor_type=surp_extractor_type,
                model_name=model_name,
                model_target_device=model_target_device,
                hf_access_token=hf_access_token,
            )

            get_metrics_kwargs["add_parsing_features"] = (
                True if metric_df is None and add_parsing_features else False
            )
            metric_dfs = extract_metrics_for_text_df(
                text_df=text_df,
                text_col_name=text_col_name,  # this is after turning all the words into a single string
                text_key_cols=text_key_cols,
                surp_extractor=surp_extractor,
                get_metrics_kwargs=get_metrics_kwargs,
                **extract_metrics_for_text_df_kwargs,
            )

            """Here we incrementally add the metrics from all models
            In order to avoid duplicates, we merge only on the columns that were added in
            the current iteration (surprisal estimates from the current model)"""
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
                
            def save_temp(df: pd.DataFrame, save_path, groupby_cols) -> pd.DataFrame:
                text_df_w_metrics = df.drop(
                    columns=[
                        "Word",
                        "Length",
                        "Wordfreq_Frequency",
                        "subtlex_Frequency",
                    ]
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
            if save_path is not None:
                save_temp(metric_df, save_path, text_key_cols)
            # move the model back to the cpu and delete it to free up space
            del surp_extractor
            gc.collect()
            torch.cuda.empty_cache()
            

        except Exception as e:
            print(f"Error for {model_name}: {e}")
        
    return metric_df


def add_metrics_to_word_level_eye_tracking_report(
    eye_tracking_data: pd.DataFrame,
    surprisal_extraction_model_names: List[str],
    textual_item_key_cols: List[str],
    spacy_model_name: str,
    surp_extractor_type: extractors_constants.SurpExtractorType,
    parsing_mode: Literal["keep-first", "keep-all", "re-tokenize"],
    model_target_device: str = "cpu",
    hf_access_token: str | None = None,
) -> pd.DataFrame:
    """This function adds metrics to the eye_tracking_data dataframe.

    Args:
        eye_tracking_data (pd.DataFrame): A dataframe with eye tracking data
        surprisal_extraction_model_names (List[str]): The names of the models to extract surprisal
            values from (huggingface models)
        textual_item_key_cols (List[str]): The columns that identify textual items
        spacy_model_name (str): The name of the spacy model to use for parsing the text
        surp_extractor_type (SurpExtractorType): The type of surprisal extractor to use
        parsing_mode (str): Type of parsing to use. one of
            ['keep-first','keep-all','re-tokenize']
        model_target_device (str, optional): The device to move the model to. Defaults to "cpu".
        hf_access_token (str, optional): A huggingface access token. Defaults to None.

    Returns:
        pd.DataFrame: The eye_tracking_data dataframe with the added word-level metrics
    """

    # Remove duplicates
    without_duplicates = eye_tracking_data[
        textual_item_key_cols
        + [
            "IA_ID",
            "IA_LABEL",
        ]
    ].drop_duplicates()

    # Group by paragraph_id, batch, article_id, level and join all IA_LABELs (words)
    text_from_et = without_duplicates.groupby(textual_item_key_cols)["IA_LABEL"].apply(
        list
    )

    text_from_et = text_from_et.apply(lambda text: " ".join(text))

    spacy_model = spacy.load(spacy_model_name)

    extract_metrics_partial = partial(
        extract_metrics_for_text_df_multiple_hf_models,
        text_col_name="IA_LABEL",
        text_key_cols=textual_item_key_cols,
        surprisal_extraction_model_names=surprisal_extraction_model_names,
        surp_extractor_type=surp_extractor_type,
        parsing_mode=parsing_mode,
        spacy_model=spacy_model,
        model_target_device=model_target_device,
        hf_access_token=hf_access_token,
    )
    metric_df = extract_metrics_partial(
        text_df=text_from_et,
        extract_metrics_for_text_df_kwargs=dict(
            ordered_prefix_col_names=[],
        ),
    )

    metric_df = metric_df.rename({"index": "IA_ID", "Word": "IA_LABEL"}, axis=1)

    # Join metrics with eye_tracking_data
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


if __name__ == "__main__":
    df = pd.read_csv(
        "ln_shared_data/onestop/processed/ia_data_enriched_360_05052024.csv",
    ).drop(
        columns=["gpt2_Surprisal", "Length", "Wordfreq_Frequency", "subtlex_Frequency"]
    )
    df = break_down_p_id(df)

    textual_item_key_cols = [
        "paragraph_id",
        "batch",
        "article_id",
        "level",
        "has_preview",
        "question",
    ]

    surprisal_extraction_model_names = ["gpt2"]
    df = add_metrics_to_word_level_eye_tracking_report(
        eye_tracking_data=df,
        surprisal_extraction_model_names=surprisal_extraction_model_names,
        textual_item_key_cols=textual_item_key_cols,
        spacy_model_name="en_core_web_sm",
        surp_extractor_type=extractor_switch.SurpExtractorType.PIMENTEL_CTX_LEFT,
        parsing_mode="re-tokenize",
        model_target_device="cuda:0",
        hf_access_token="blablabla",
    )

    df.rename(columns={"IA_LABEL_x": "IA_LABEL"}, inplace=True)

    group_columns = ["subject_id", "unique_paragraph_id"]
    columns_to_shift = [
        "Wordfreq_Frequency",
        "subtlex_Frequency",
        "Length",
    ]
    columns_to_shift += [
        f"{model}_Surprisal" for model in surprisal_extraction_model_names
    ]
    for column in columns_to_shift:
        df[f"prev_{column}"] = df.groupby(group_columns)[column].shift(1)
