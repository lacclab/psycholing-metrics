"""Text processing utilities: cleaning, parsing, and word-level feature extraction."""

import re
from collections import defaultdict
from typing import List, Literal

import numpy as np
import pandas as pd
from spacy.language import Language

CONTENT_WORDS = {
    "PUNCT": False,
    "PROPN": True,
    "NOUN": True,
    "PRON": False,
    "VERB": True,
    "SCONJ": False,
    "NUM": False,
    "DET": False,
    "CCONJ": False,
    "ADP": False,
    "AUX": False,
    "ADV": True,
    "ADJ": True,
    "INTJ": False,
    "X": False,
    "PART": False,
}

REDUCED_POS = {
    "PUNCT": "FUNC",
    "PROPN": "NOUN",
    "NOUN": "NOUN",
    "PRON": "FUNC",
    "VERB": "VERB",
    "SCONJ": "FUNC",
    "NUM": "FUNC",
    "DET": "FUNC",
    "CCONJ": "FUNC",
    "ADP": "FUNC",
    "AUX": "FUNC",
    "ADV": "ADJ",
    "ADJ": "ADJ",
    "INTJ": "FUNC",
    "X": "FUNC",
    "PART": "FUNC",
}


def clean_text(raw_text: str) -> str:
    """Replace problematic Unicode characters with ASCII equivalents.

    Made for OnestopQA corpus text normalization.
    E.g., \u201c -> ", \u00eb -> e
    """
    return (
        raw_text.replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2026", "...")
        .replace("\u2018", "'")
        .replace("\u00e9", "e")
        .replace("\u00eb", "e")
        .replace("\ufb01", "fi")
        .replace("\u00ef", "i")
    )


def is_content_word(pos: str) -> bool:
    """Check if a POS tag corresponds to a content word (noun, verb, adj, adv)."""
    return CONTENT_WORDS.get(pos, False)


def get_reduced_pos(pos: str) -> str:
    """Map a fine-grained POS tag to a reduced set: NOUN, VERB, ADJ, or FUNC."""
    return REDUCED_POS.get(pos, "UNKNOWN")


def get_dependency_direction(head_idx: int, word_idx: int) -> str:
    """Return the direction from a word to its dependency head: LEFT, RIGHT, or SELF."""
    if head_idx > word_idx:
        return "RIGHT"
    elif head_idx < word_idx:
        return "LEFT"
    else:
        return "SELF"


def get_parsing_features(
    text: str,
    spacy_model: Language,
    mode: Literal["keep-first", "keep-all", "re-tokenize"] = "re-tokenize",
) -> pd.DataFrame:
    """Extract word-level linguistic features (POS, NER, morphology, dependencies) using spaCy.

    :param text: the text to extract features from.
    :param spacy_model: the spaCy model to use.
    :param mode: tokenization alignment strategy:
        - 'keep-first': keep the first sub-token's features for compressed words
        - 'keep-all': keep all sub-token features as lists
        - 're-tokenize': merge sub-tokens into single tokens via spaCy retokenization
    :return: DataFrame where each row is a word with its linguistic features.
    """
    features = {}
    doc = spacy_model(text)
    token_idx = 0
    word_idx = 1
    token_idx2word_idx = {}
    spans_to_merge = []
    while token_idx < len(doc):
        token = doc[token_idx]
        accumulated_tokens = []
        while not bool(token.whitespace_) and token_idx < len(doc):
            accumulated_tokens.append((token.i, token))
            token_idx += 1
            if token_idx < len(doc):
                token = doc[token_idx]

        if token_idx < len(doc):
            accumulated_tokens.append((token.i, token))
        token_idx += 1

        if len(accumulated_tokens) > 1:
            start_idx = accumulated_tokens[0][0]
            end_idx = accumulated_tokens[-1][0] + 1
            spans_to_merge.append(doc[start_idx:end_idx])

        if mode in ("keep-first", "keep-all"):
            features[word_idx] = accumulated_tokens
            for token in accumulated_tokens:
                token_idx2word_idx[token[0]] = word_idx
            word_idx += 1

    if mode == "re-tokenize":
        with doc.retokenize() as retokenizer:
            for span in spans_to_merge:
                retokenizer.merge(span)
        for word_idx, token in enumerate(doc):
            features[word_idx + 1] = [(token.i, token)]

    res = []
    for word_idx, word in features.items():
        word_features: dict[str, list[list[str] | int | str | None]] = defaultdict(list)
        word_features["Word_idx"] = [word_idx]
        for ind, token in word:
            word_features["Token"].append(token.text)
            word_features["POS"].append(token.pos_)
            word_features["TAG"].append(token.tag_)
            word_features["Token_idx"].append(ind)
            word_features["Relationship"].append(token.dep_)
            word_features["Morph"].append([f for f in token.morph])
            word_features["Entity"].append(
                token.ent_type_ if token.ent_type_ != "" else None
            )
            word_features["Is_Content_Word"].append(is_content_word(token.pos_))
            word_features["Reduced_POS"].append(get_reduced_pos(token.pos_))
            if mode in ("keep-first", "keep-all"):
                word_features["Head_word_idx"].append(
                    token_idx2word_idx[token.head.i]
                    if token.head.i in token_idx2word_idx
                    else -1
                )
                word_features["n_Lefts"].append(
                    len([d for d in token.lefts if d.i in token_idx2word_idx])
                )
                word_features["n_Rights"].append(
                    len([d for d in token.rights if d.i in token_idx2word_idx])
                )
                word_features["AbsDistance2Head"].append(
                    abs(token_idx2word_idx[ind] - token_idx2word_idx[token.head.i])
                    if token.head.i in token_idx2word_idx
                    else -1
                )
                word_features["Distance2Head"].append(
                    token_idx2word_idx[ind] - token_idx2word_idx[token.head.i]
                    if token.head.i in token_idx2word_idx
                    else -1
                )
                word_features["Head_Direction"].append(
                    get_dependency_direction(
                        token_idx2word_idx[token.head.i], token_idx2word_idx[ind]
                    )
                    if token.head.i in token_idx2word_idx
                    else "UNKNOWN"
                )
            else:
                word_features["Head_word_idx"].append(token.head.i + 1)
                word_features["n_Lefts"].append(token.n_lefts)
                word_features["n_Rights"].append(token.n_rights)
                word_features["AbsDistance2Head"].append(abs(token.head.i - token.i))
                word_features["Distance2Head"].append(token.head.i - token.i)
                word_features["Head_Direction"].append(
                    get_dependency_direction(token.head.i, token.i)
                )

        res.append(word_features)

    final_res = pd.DataFrame(res)
    assert pd.__version__ > "2.1.0", f"""Your pandas version is {pd.__version__}
            Please upgrade pandas to version 2.1.0 or higher to use mode={mode}.
            (requires pd.DataFrame.map)""".replace("\n", "")
    if mode in ("keep-first", "re-tokenize"):
        final_res = final_res.map(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
        )

    return final_res


def add_col_not_num_or_punc(df: pd.DataFrame):
    df["not_num_or_punc"] = df["IA_LABEL"].apply(
        lambda x: bool(re.match("^[a-zA-Z ]*$", x))
    )
    return df


def break_down_paragraph_id(et_data_enriched: pd.DataFrame):
    """Split 'unique_paragraph_id' into component columns: batch, article_id, level, paragraph_id."""
    col_names = ["batch", "article_id", "level", "paragraph_id"]
    for i, name in enumerate(col_names):
        et_data_enriched[name] = et_data_enriched["unique_paragraph_id"].apply(
            lambda x: x.split("_")[i]
        )
        if name != "level":
            et_data_enriched[name] = et_data_enriched[name].astype(int)

    return et_data_enriched


def get_word_offsets(words: List[str]):
    """Compute character offset ranges for each word in a space-separated text.

    :param words: list of words (as from str.split())
    :return: list of (start, end) character offset tuples
    """
    offsets = []
    pos = 0
    for w in words:
        offsets.append((pos, pos + len(w)))
        pos += len(w) + 1
    return offsets


def aggregate_token_log_probs(text: str, token_log_probs: np.ndarray, token_offsets: list):
    """Aggregate token-level log probabilities to word-level.

    Sums log probabilities of sub-word tokens that belong to the same whitespace-delimited word.

    :param text: the input text (no leading/trailing whitespace).
    :param token_log_probs: log probability for each token.
    :param token_offsets: character offset (start, end) for each token.
    :return: tuple of (word_log_probs, list of (word, log_prob) pairs)
    """
    words = text.split()
    agg_log_probs = []
    word_offsets = get_word_offsets(words)
    cur_prob = 0
    cur_word_ind = 0
    for i, (lp, ind) in enumerate(zip(token_log_probs, token_offsets)):
        cur_prob += lp
        if ind[1] == word_offsets[cur_word_ind][1]:
            agg_log_probs.append(cur_prob)
            cur_prob = 0
            cur_word_ind += 1

    zipped_surp = list(zip(words, agg_log_probs))
    return agg_log_probs, zipped_surp


def trim_left_context(tokenizer, left_context_text: str, max_tokens: int) -> str:
    """Trim left context from the beginning so it fits within the model's max context window.

    Removes words from the start of the context until the token count is within the limit.

    :param tokenizer: the tokenizer to use for encoding.
    :param left_context_text: the left context text to trim.
    :param max_tokens: the maximum number of tokens allowed.
    """
    words = left_context_text.split()
    while len(tokenizer.encode(" ".join(words))) > max_tokens:
        words.pop(0)
    return " ".join(words)
