"""Core functions for extracting word-level metrics: surprisal, frequency, and word length."""

import string
from importlib.resources import files
from typing import Literal

import numpy as np
import pandas as pd
import spacy
from wordfreq import tokenize, word_frequency

from psycholing_metrics.surprisal.base import BaseSurprisalExtractor
from psycholing_metrics.surprisal.types import SurprisalExtractorType
from psycholing_metrics.text_processing import (
    aggregate_token_log_probs,
    clean_text,
    get_parsing_features,
)


def get_surprisal(
    target_text: str,
    surp_extractor: BaseSurprisalExtractor,
    overlap_size: int = 512,
    left_context_text: str | None = None,
) -> pd.DataFrame:
    """Compute surprisal values for each word in text.

    Words are split by whitespace and include adjacent punctuation.
    A word's surprisal is the sum of its sub-word token surprisals.

    :param target_text: the text to get surprisal values for.
    :param surp_extractor: the surprisal extractor to use.
    :param overlap_size: token overlap between chunks for long texts.
    :param left_context_text: optional left context to condition on.
    :return: DataFrame with 'Word' and 'Surprisal' columns.
    """

    if surp_extractor.extractor_type_name not in [
        SurprisalExtractorType.PIMENTEL_CTX_LEFT.value,
        SurprisalExtractorType.INV_EFFECT_EXTRACTOR.value,
    ]:
        probs, offset_mapping = surp_extractor.compute_surprisal(
            target_text=target_text,
            left_context_text=left_context_text,
            overlap_size=overlap_size,
        )

        dataframe_probs = pd.DataFrame(
            aggregate_token_log_probs(target_text, probs, offset_mapping)[1],
            columns=["Word", "Surprisal"],
        )
    else:
        dataframe_probs = surp_extractor.compute_surprisal(
            target_text=target_text,
            left_context_text=left_context_text,
            overlap_size=overlap_size,
        )
    assert not dataframe_probs.isnull().values.any(), (
        "There are NaN values in the dataframe."
    )
    assert len(dataframe_probs) == len(target_text.split()), (
        "The number of words in the surprisal dataframe does not match the number of words in the text."
    )
    return dataframe_probs


def get_frequency(text: str, language: str) -> pd.DataFrame:
    """Compute word frequency (negative log2) for each word in text.

    Primary source is the ``wordfreq`` library (Speer 2022), the frequency
    source used by Wilcox et al. (2023, TACL) for cross-lingual reading
    studies. For English only, the SUBTLEX-US corpus (Brysbaert & New 2009)
    is also provided as ``subtlex_Frequency`` for comparability with prior
    psycholinguistic work. Words are split by whitespace; for multi-token
    lookups the harmonic mean of token frequencies is used.

    :param text: the text to get frequencies for.
    :param language: language code (e.g. 'en'). SUBTLEX column is populated
        only when ``language == 'en'`` (SUBTLEX-US is English-only); other
        languages get ``float('inf')`` for that column.
    :return: DataFrame with 'Word', 'Wordfreq_Frequency', and 'subtlex_Frequency' columns.

    >>> text = "hello, how are you?"
    >>> frequencies = get_frequency(text=text, language="en")
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
            -np.log2(word_frequency(word, lang=language, minimum=1e-11))
            for word in words
        ],  # minimum equal to ~36.5
    }

    if language == "en":
        data_path = files("psycholing_metrics").joinpath(
            "data/SUBTLEXus74286wordstextversion_lower.tsv"
        )
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
                subtlex_freq = (
                    1.0 / one_over_result if one_over_result != 0 else float("inf")
                )
            subtlex_freqs.append(subtlex_freq)
        frequencies["subtlex_Frequency"] = subtlex_freqs
    else:
        frequencies["subtlex_Frequency"] = [float("inf")] * len(words)

    return pd.DataFrame(frequencies)


def get_word_length(text: str, disregard_punctuation: bool = True) -> pd.DataFrame:
    """Compute the length of each word in text.

    :param text: the text to get lengths for.
    :param disregard_punctuation: if True, strip punctuation before counting characters.
    :return: DataFrame with 'Word' and 'Length' columns.

    >>> get_word_length("hello, how are you?", disregard_punctuation=True)
         Word  Length
    0  hello,       5
    1     how       3
    2     are       3
    3    you?       3

    >>> get_word_length("hello, how are you?", disregard_punctuation=False)
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
    language: str = "en",
    disregard_punctuation: bool = True,
) -> pd.DataFrame:
    """Extract all word-level metrics: surprisal, frequency, length, and optionally parsing features.

    This is the main entry point for computing word-level attributes for a piece of text.

    :param target_text: the text to extract metrics for.
    :param surp_extractor: the surprisal extractor to use.
    :param parsing_model: spaCy model for linguistic features (POS, NER, dependencies, etc.).
    :param parsing_mode: tokenization alignment strategy for spaCy parsing.
    :param left_context_text: optional left context for surprisal conditioning.
    :param add_parsing_features: whether to include spaCy parsing features.
    :param overlap_size: token overlap between chunks for long texts.
    :param language: language code for frequency lookup.
    :param disregard_punctuation: whether to strip punctuation when computing word length.
    :return: DataFrame where each row is a word with its metrics.
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
    surp_col_suffix = surp_extractor.extractor_type.column_suffix
    surprisal.rename(
        columns={
            "Surprisal": f"{surp_extractor.model_name}_{surp_col_suffix}_Surprisal"
        },
        inplace=True,
    )

    frequency = get_frequency(text=target_text_reformatted, language=language)
    word_length = get_word_length(
        text=target_text_reformatted, disregard_punctuation=disregard_punctuation
    )

    merged_df = word_length.join(frequency.drop("Word", axis=1))
    merged_df = merged_df.join(surprisal.drop("Word", axis=1))

    if add_parsing_features:
        assert parsing_model is not None, (
            "Please provide a parsing model to extract parsing features."
        )
        assert parsing_mode is not None, (
            "Please provide a parsing mode to extract parsing features."
        )

        parsing_features = get_parsing_features(
            target_text_reformatted, parsing_model, parsing_mode
        )
        merged_df = merged_df.join(parsing_features)

    return merged_df


if __name__ == "__main__":
    from psycholing_metrics.surprisal.factory import create_surprisal_extractor

    text = """Many of us know we don't get enough sleep, but imagine if there was a simple solution:
    getting up later. In a speech at the British Science Festival, Dr. Paul Kelley from Oxford University
    said schools should stagger their starting times to work with the natural rhythms of their students.
    This would improve exam results and students' health (lack of sleep can cause diabetes, depression,
    obesity and other health problems).""".replace("\n", " ").replace("    ", "")
    question = "Which university is Dr. Paul Kelley from?"

    model_name = "gpt2"
    surp_extractor = create_surprisal_extractor(
        extractor_type=SurprisalExtractorType.CAT_CTX_LEFT,
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
