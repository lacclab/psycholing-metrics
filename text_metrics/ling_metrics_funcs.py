import string
from typing import Literal

import numpy as np
import pandas as pd
import spacy
from text_metrics.surprisal_extractors.base_extractor import BaseSurprisalExtractor
from importlib.resources import files

from text_metrics.utils import get_parsing_features, string_to_log_probs, clean_text
from wordfreq import word_frequency, tokenize
from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType
from text_metrics.surprisal_extractors.extractor_switch import (
    get_surp_extractor,
)
import matplotlib
from text_metrics.viz_text_heatmap import generate_html_for_texts


# Credits: https://github.com/byungdoh/llm_surprisal/blob/eacl24/get_llm_surprisal.py
# https://github.com/rycolab/revisiting-uid/blob/0b60df7e8f474d9c7ac938e7d8a02fda6fc8787a/src/language_modeling.py#L136
def get_surprisal(
    target_text: str,
    surp_extractor: BaseSurprisalExtractor,
    overlap_size: int = 512,
    left_context_text: str | None = None,
) -> pd.DataFrame:
    """
    Get surprisal values for each word in text.

    Words are split by white space, and include adjacent punctuation.
    A surprisal of a word is the sum of the surprisal of the subwords
    (as split by the tokenizer) that make up the word.

    :param target_text: str, the text to get surprisal values for.
    :param surp_extractor: the surprisal extractor to use.
    :param overlap_size: int, the number of tokens to overlap between chunks if text is too long.
    :param left_context_text: str or None, optional left context to prepend.
    :return: pd.DataFrame, each row represents a word and its surprisal.
    """

    if surp_extractor.extractor_type_name not in [
        SurpExtractorType.PIMENTEL_CTX_LEFT.value,
        SurpExtractorType.INV_EFFECT_EXTRACTOR.value,
    ]:
        probs, offset_mapping = surp_extractor.surprise(
            target_text=target_text,
            left_context_text=left_context_text,
            overlap_size=overlap_size,
        )

        dataframe_probs = pd.DataFrame(
            string_to_log_probs(target_text, probs, offset_mapping)[1],
            columns=["Word", "Surprisal"],
        )
    else:
        dataframe_probs = surp_extractor.surprise(
            target_text=target_text,
            left_context_text=left_context_text,
            overlap_size=overlap_size,
        )
    # assert there are no NaN values
    assert (
        not dataframe_probs.isnull().values.any()
    ), "There are NaN values in the dataframe."
    assert (
        len(dataframe_probs) == len(target_text.split())
    ), "The number of words in the surprisal dataframe does not match the number of words in the text."
    return dataframe_probs


def get_frequency(text: str, language: str) -> pd.DataFrame:
    """
    Get (negative log2) frequencies for each word in text.

    Words are split by white space.
    A frequency of a word does not include adjacent punctuation.
    Half harmonic mean is applied for complex words.
    E.g. freq(top-level) = 1/(1/freq(top) + 1/freq(level))

    :param text: str, the text to get frequencies for.
    :param language: str, the language of the text.
    :return: pd.DataFrame, each row represents a word and its frequency.

    >>> text = "hello, how are you?"
    >>> frequencies = get_frequency(text=text, language=language)
    >>> frequencies
         Word  Wordfreq_Frequency  subtlex_Frequency
    0  hello,           14.217323          10.701528
    1     how            9.166697           8.317353
    2     are            7.506353           7.548023
    3    you?            6.710284           4.541699
    """
    words = text.split()
    frequencies = {
        "Word": words,
        "Wordfreq_Frequency": [
            -np.log2(word_frequency(word, lang=language, minimum=1e-11)) for word in words
        ],  # minimum equal to ~36.5
    }
    data_path = files("text_metrics").joinpath("data/SUBTLEXus74286wordstextversion_lower.tsv")
    subtlex = pd.read_csv(
        data_path,
        sep="\t",
        index_col=0,
    )
    subtlex["Frequency"] = -np.log2(subtlex["Count"] / subtlex.sum().iloc[0])

    subtlex_freqs = []
    for word in words:
        tokens = tokenize(word, lang=language)
        one_over_result = 0.0
        try:
            for token in tokens:
                one_over_result += 1.0 / subtlex.loc[token, "Frequency"]
        except KeyError:
            subtlex_freq = float("inf")
        else:
            subtlex_freq = 1.0 / one_over_result if one_over_result != 0 else float("inf")
        subtlex_freqs.append(subtlex_freq)
    frequencies["subtlex_Frequency"] = subtlex_freqs

    return pd.DataFrame(frequencies)


def get_word_length(text: str, disregard_punctuation: bool = True) -> pd.DataFrame:
    """
    Get the length of each word in text.

    Words are split by white space.

    :param text: str, the text to get lengths for.
    :param disregard_punctuation: bool, to include adjacent punctuation (False) or not (True).
    :return: pd.DataFrame, each row represents a word and its length.

    Examples
    --------
    >>> text = "hello, how are you?"
    >>> lengths = get_word_length(text=text, disregard_punctuation=True)
    >>> lengths
         Word  Length
    0  hello,       5
    1     how       3
    2     are       3
    3    you?       3

    >>> text = "hello, how are you?"
    >>> lengths = get_word_length(text=text, disregard_punctuation=False)
    >>> lengths
         Word  Length
    0  hello,       6
    1     how       3
    2     are       3
    3    you?       4


    """
    word_lengths = {
        "Word": text.split(),
    }
    if disregard_punctuation:
        #     text = text.translate(str.maketrans('', '', string.punctuation))
        word_lengths["Length"] = [
            len(word.translate(str.maketrans("", "", string.punctuation)))
            for word in text.split()
        ]
    else:
        word_lengths["Length"] = [len(word) for word in text.split()]

    return pd.DataFrame(word_lengths)


def get_metrics(
    target_text: str,
    surp_extractor: BaseSurprisalExtractor,
    parsing_model: spacy.Language | None,
    parsing_mode: (
        Literal["keep-first", "keep-all", "re-tokenize"] | None
    ) = "re-tokenize",
    left_context_text: str | None = None,
    add_parsing_features: bool = True,
    overlap_size: int = 512,
    language: str = 'en',
    disregard_punctuation: bool = True,
) -> pd.DataFrame:
    """
    Wrapper function to get the surprisal and frequency values and length of each word in the text.

    :param target_text: str, the text to get metrics for.
    :param surp_extractor: the surprisal extractor to use.
    :param parsing_model: the spacy model to use for parsing the text.
    :param parsing_mode: type of parsing to use. one of ['keep-first','keep-all','re-tokenize']
    :param left_context_text: str or None, optional left context to prepend for surprisal extraction.
    :param add_parsing_features: whether to add parsing features to the output.
    :param overlap_size: int, the number of tokens to overlap between chunks if text is too long.
    :param language: language of the text.
    :param disregard_punctuation: whether to disregard punctuation in word length computation.
    :return: pd.DataFrame, each row represents a word, its length, surprisal and frequency.
    """

    target_text_reformatted = clean_text(target_text)
    left_context_text_reformatted = (
        clean_text(left_context_text) if left_context_text is not None else None
    )
    surprisal = get_surprisal(
        target_text=target_text_reformatted,
        left_context_text=left_context_text_reformatted,
        surp_extractor=surp_extractor,
        overlap_size=overlap_size,
    )
    surprisal.rename(
        columns={"Surprisal": f"{surp_extractor.model_name}_Surprisal"}, inplace=True
    )

    frequency = get_frequency(text=target_text_reformatted, language=language)
    word_length = get_word_length(
        text=target_text_reformatted, disregard_punctuation=disregard_punctuation
    )

    merged_df = word_length.join(frequency.drop("Word", axis=1))
    merged_df = merged_df.join(surprisal.drop("Word", axis=1))

    if add_parsing_features:
        assert (
            parsing_model is not None
        ), "Please provide a parsing model to extract parsing features."
        assert (
            parsing_mode is not None
        ), "Please provide a parsing mode to extract parsing features."

        parsing_features = get_parsing_features(
            target_text_reformatted, parsing_model, parsing_mode
        )
        merged_df = merged_df.join(parsing_features)

    return merged_df


if __name__ == "__main__":
    text = """Many of us know we don't get enough sleep, but imagine if there was a simple solution:
    getting up later. In a speech at the British Science Festival, Dr. Paul Kelley from Oxford University
    said schools should stagger their starting times to work with the natural rhythms of their students.
    This would improve exam results and students' health (lack of sleep can cause diabetes, depression,
    obesity and other health problems).""".replace("\n", " ").replace("    ", "")
    question = "Which university is Dr. Paul Kelley from?"

    model_name = "gpt2"
    surp_extractor = get_surp_extractor(
        extractor_type=SurpExtractorType.CAT_CTX_LEFT,
        model_name=model_name,
        hf_access_token="",
    )
    parsing_model = spacy.load("en_core_web_sm")

    metrics = get_metrics(
        target_text=text,
        surp_extractor=surp_extractor,
        parsing_model=parsing_model,
        parsing_mode="re-tokenize",
        left_context_text=question,
        add_parsing_features=True,
    )

    print(metrics.head(5).to_markdown())

    extractor_type_lst = [
        SurpExtractorType.CAT_CTX_LEFT,
        SurpExtractorType.CAT_CTX_LEFT,
    ]
    metrics_df = None
    surp_lsts = []
    surp_name_lsts = []
    for i, extractor_type in enumerate(extractor_type_lst):
        surp_extractor = get_surp_extractor(
            model_name=model_name, extractor_type=extractor_type
        )
        q = question if i > 0 else None
        metrics = get_metrics(
            target_text=text,
            surp_extractor=surp_extractor,
            parsing_model=None,
            parsing_mode=None,
            left_context_text=q,
            add_parsing_features=False,
            overlap_size=512,
        )
        surp_col_name = f"{extractor_type.value}_{'Q_P' if i > 0 else 'P'}_Surprisal"
        metrics.rename(
            columns={f"{model_name}_Surprisal": surp_col_name},
            inplace=True,
        )
        surp_lsts.append(metrics[surp_col_name].values.tolist())
        surp_name_lsts.append(surp_col_name)

        if metrics_df is None:
            metrics_df = metrics
        else:
            columns_to_use = metrics.columns.difference(metrics_df.columns)
            metrics_df = metrics_df.merge(
                metrics[columns_to_use], left_index=True, right_index=True
            )

    generate_html_for_texts(
        titles=surp_name_lsts,
        texts=[text] * len(surp_name_lsts),
        weights_list=surp_lsts,
        output_file_name="colorized_texts_w_surp.html",
        color_normalizing_factor=10,
        cmap=matplotlib.colormaps.get_cmap("Blues"),
        additional_note="Question: " + question,
    )

    metrics_df.to_csv("metrics_df.csv", index=False)
