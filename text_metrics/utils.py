from collections import defaultdict
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from spacy.language import Language
from torch.nn.functional import log_softmax
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
    LlamaForCausalLM,
    MambaForCausalLM,
)
import re

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
    """
    Replaces the problematic characters in the raw_text, made for OnestopQA.
    E.g., "ë" -> "e"
    """
    return (
        raw_text.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("…", "...")
        .replace("‘", "'")
        .replace("é", "e")
        .replace("ë", "e")
        .replace("ﬁ", "fi")
        .replace("ï", "i")
    )


def is_content_word(pos: str) -> bool:
    """
    Checks if the pos is a content word.
    """
    return CONTENT_WORDS.get(pos, False)


def get_reduced_pos(pos: str) -> str:
    """
    Returns the reduced pos tag of the pos tag.
    """
    return REDUCED_POS.get(
        pos, "UNKNOWN"
    )  # TODO Why should there be UNKNOWN? Why not map to a reduced pos?


def get_direction(head_idx: int, word_idx: int) -> str:
    """
    Returns the direction of the word from the head word.
    :param head_idx: int, the head index.
    :param word_idx: int, the word index.
    :return: str, the direction of the word from the head.
    """
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
    """
    Extracts the parsing features from the text using spacy.
    :param text: str, the text to extract features from.
    :param spacy_model: the spacy model to use.
    :param mode: type of parsing to use. one of ['keep-first','keep-all','re-tokenize']
    :return: pd.DataFrame, each row represents a word and its parsing features.
            for compressed words (e.g., "don't"),
     each feature has a list of all the sub-words' features.
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

        if mode == "keep-first" or mode == "keep-all":
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
            if mode == "keep-first" or mode == "keep-all":
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
                    get_direction(
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
                    get_direction(token.head.i, token.i)
                )

        res.append(word_features)

    final_res = pd.DataFrame(res)
    if mode == "keep-all":
        pass

    assert pd.__version__ > "2.1.0", f"""Your pandas version is {pd.__version__}
            Please upgrade pandas to version 2.1.0 or higher to use mode={mode}.
            (requires pd.DataFrame.map)""".replace("\n", "")
    if mode == "keep-first":
        final_res = final_res.map(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
        )
    elif mode == "re-tokenize":
        final_res = final_res.map(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
        )

    return final_res


def _get_surp(text: str, tokenizer, model) -> list[tuple[str, float]]:
    """
    Extract surprisal values from model for text tokenized by tokenizer.

    :param text: the  text to get surprisal values for.
    :param model: model used to get surprisal values.
    :param tokenizer: should be compatible with model.
    :return: list of tuples of (subword, surprisal values).
    """
    # text = text  # + tokenizer.eos_token  # add beginning of sentence token
    ids = torch.tensor(tokenizer.encode(text))
    toks = tokenizer.tokenize(text)

    with torch.no_grad():
        outputs = model(ids)

    # log softmax converted to base 2.
    # More numerically stable than -log2(softmax(outputs[0], dim=1))
    log_probs = -(1 / torch.log(torch.tensor(2.0))) * log_softmax(outputs[0], dim=1)

    out = []
    for ind, word_id in enumerate(ids, 0):
        word_log_prob = float(log_probs[ind - 1, word_id])
        out.append((toks[ind], word_log_prob))
    return out


def _join_surp(words: list[str], tok_surps: list[tuple[str, float]]):
    """
    Add up the subword surprisals of each word.

    :param words: list of actual words
    :param tok_surps: list of tuples of (subword, subword surprisal value)
    :return: list of tuples of (word, word surprisal value)
    """
    out = []
    word_surp, word_ind, within_word_position = 0, 0, 0
    word_till_now = ""
    for tok, tok_surp in tok_surps:
        tok_str = tok[1:] if tok.startswith("Ġ") else tok
        tok_str = tok_str.replace("Â", "").replace(
            "âĤ¬", "€"
        )  # Converts back euro and gbp sign
        assert (
            words[word_ind][within_word_position : within_word_position + len(tok_str)]
            == tok_str
        ), (
            words[word_ind][within_word_position : within_word_position + len(tok_str)]
            + "!="
            + tok_str
        )
        word_surp += tok_surp
        within_word_position += len(tok_str)
        word_till_now += tok_str

        if word_till_now == words[word_ind]:
            out.append((words[word_ind], word_surp))
            word_ind += 1
            word_surp, within_word_position = 0, 0
            word_till_now = ""
    assert word_ind == len(words)
    assert len(out) == len(words)
    return out


def add_col_not_num_or_punc(df: pd.DataFrame):
    df["not_num_or_punc"] = df["IA_LABEL"].apply(
        lambda x: bool(re.match("^[a-zA-Z ]*$", x))
    )
    return df


def break_down_p_id(et_data_enriched: pd.DataFrame):
    # "unique_paragraph_id" -> "batch", 'article_id', 'level', 'paragraph_id' (sepated by "_")
    col_names = ["batch", "article_id", "level", "paragraph_id"]
    for i, name in enumerate(col_names):
        et_data_enriched[name] = et_data_enriched["unique_paragraph_id"].apply(
            lambda x: x.split("_")[i]
        )
        if name != "level":
            et_data_enriched[name] = et_data_enriched[name].astype(int)

    return et_data_enriched


def init_tok_n_model(
    model_name: str,
    device: str = "cpu",
    pythia_checkpoint: str | None = "step143000",
    hf_access_token: str | None = None,
):
    """This function initializes the tokenizer and model for the specified LLM variant.

    Args:
        model_name (str): the model name. Supported models:
            GPT-2 family:
            "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

            GPT-Neo family:
            "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
            "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

            OPT family:
            "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
            "facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"

            Pythia family:
            "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
            "step1", "step2", "step4", ..., "step142000", "step143000"
            Llama family:
            "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf"

            Gemma family (also `*-it` versions):
            "google/gemma-2b", "google/gemma-7b", "google/recurrentgemma-2b", "google/recurrentgemma-9b"

            Mamba family: (install (pip): mamba-ssm, causal-conv1d>=1.2.0)
            "state-spaces/mamba-130m-hf", "state-spaces/mamba-370m-hf",
            "state-spaces/mamba-790m-hf", "state-spaces/mamba-1.4b-hf",
            "state-spaces/mamba-2.8b-hf"

            Mistral family:
            "mistralai/Mistral-7B-Instruct-v0.*" where * is the version number

        device (str, optional): Defaults to 'cpu'.
        pythia_checkpoint (str, optional): The checkpoint for Pythia models. Defaults to 'step143000'.

    Raises:
        ValueError: Unsupported LLM variant

    Returns:
        Tuple[Union[AutoTokenizer, GPTNeoXTokenizerFast],
              Union[AutoModelForCausalLM, GPTNeoXForCausalLM]]: tokenizer, model
    """

    # TODO merge AutoTokenizer/ModelForCausalLM with/without hf_access_token?
    model_variant = model_name.split("/")[-1]
    model_variant_lower = model_variant.lower()

    if any(
        variant in model_variant_lower
        for variant in ["gpt-neo", "gpt", "opt", "mamba", "rwkv"]
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    elif "gpt-neox" in model_variant_lower:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)

    elif "Eagle" in model_variant:  # RWKV V5
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    elif any(variant in model_variant_lower for variant in ["llama", "mistral", "gemma"]):
        assert (
            hf_access_token is not None
        ), f"Please provide the HuggingFace access token to load {model_name}"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, token=hf_access_token
        )

    elif "pythia" in model_variant_lower:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=pythia_checkpoint, use_fast=True
        )

    else:
        raise ValueError("Unsupported LLM variant")

    if any(variant in model_variant_lower for variant in ["gpt-neo", "gpt", "opt", "rwkv"]):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    elif "Eagle" in model_variant:  # RWKV
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        ).to(torch.float32)

    elif "pythia" in model_variant_lower:
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=pythia_checkpoint, device_map="auto"
        )

    elif "mamba" in model_variant_lower:
        # print('loaded with bfloat16')
        model = MambaForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            #  torch_dtype=torch.bfloat16
        )

    elif "llama" in model_variant_lower:
        model = LlamaForCausalLM.from_pretrained(
            model_name, token=hf_access_token, device_map="auto"
        )

    elif any(variant in model_variant_lower for variant in ["gemma-2"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_access_token,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    elif any(variant in model_variant_lower for variant in ["mistral", "gemma"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_access_token, device_map="auto"
        )

    else:
        raise ValueError("Unsupported LLM variant")

    # model = model.to(device)

    return tokenizer, model


def get_word_mapping(words: List[str]):
    """Given a list of words, return the offset mapping for each word.

    Args:
        words (List[str]): The list of words

    Returns:
        Tuple[str]: The offset mapping for each word
    """
    offsets = []
    pos = 0
    for w in words:
        offsets.append((pos, pos + len(w)))
        pos += len(w) + 1
    return offsets


def string_to_log_probs(string: str, probs: np.ndarray, offsets: list):
    """Given text and token-level log probabilities, aggregate the log probabilities to word-level log probabilities.
    Note: Assumes there is no whitespace in the end and the beginning of the string.

    Handles byte-level BPE tokenizers (e.g. for Chinese) where multiple
    sub-tokens may share the same character offset.  Once the current
    word's end position is matched, remaining tokens with the same end
    position are still accumulated into that word's probability before
    moving on.

    Args:
        string (str): The input text
        probs (np.ndarray): Log probabilities for each token
        offsets (list): The offset mapping for each token

    Returns:
        Tuple[List[float], List[Tuple[float]]]: The aggregated log probabilities for each word
    """
    words = string.split()
    agg_log_probs = []
    word_mapping = get_word_mapping(words)
    cur_prob = 0
    cur_word_ind = 0
    matched = False
    for i, (lp, ind) in enumerate(zip(probs, offsets)):
        if cur_word_ind >= len(word_mapping):
            # All words already matched; ignore trailing tokens
            break
        cur_prob += lp
        if ind[1] == word_mapping[cur_word_ind][1]:
            matched = True
        elif matched:
            # We matched on a previous token but this token has a different
            # end offset -> commit the accumulated prob and start a new word
            agg_log_probs.append(cur_prob - lp)
            cur_prob = lp
            cur_word_ind += 1
            matched = False
            if cur_word_ind >= len(word_mapping):
                break
            if ind[1] == word_mapping[cur_word_ind][1]:
                matched = True

    # Commit the last word if it was matched
    if matched and cur_word_ind < len(word_mapping):
        agg_log_probs.append(cur_prob)
        cur_word_ind += 1

    zipped_surp = list(zip(words, agg_log_probs))
    return agg_log_probs, zipped_surp


def remove_redundant_left_context(
    tokenizer,
    left_context_text: str,
    max_ctx_in_tokens: int,
):
    """In surprise, we return surprisals only for the target_text. This function removes the redundant left context from the target_text

    Args:
        target_text (str): _description_
        left_context_text (str): _description_
    """
    # Split the left context into words
    words = left_context_text.split()

    # Remove words from the left context until the number of tokens is within the limit
    while len(tokenizer.encode(" ".join(words))) > max_ctx_in_tokens:
        words.pop(0)

    # Join the remaining words to form the trimmed text
    trimmed_left_context = " ".join(words)

    return trimmed_left_context


# if __name__ == "__main__":
# text = (
#     """
# 113, 115, 117, and 118 are ... The International Union of Pure and Applied Chemistry (IUPAC) is the global organization that controls
# chemical names. IUPAC confirmed the new elements on 30 December, 2015 after examining studies dating back to 2004.
# The scientists who found them must now think of formal names for the elements, which have the atomic numbers,
# 113, 115, 117, and 118. The atomic number is the number of protons in an element’s atomic nucleus.
# """.replace(
#         "\n", ""
#     )
#     .replace("\t", "")
#     .replace("    ", "")
# )
# model_names = ["gpt2", "gpt2-medium"]

# models_tokenizers = [init_tok_n_model(model_name) for model_name in model_names]
# tokenizers = [tokenizer for tokenizer, _ in models_tokenizers]
# models = [model for _, model in models_tokenizers]

# surp_res = get_metrics(
#     text=text,
#     models=models,
#     tokenizers=tokenizers,
#     model_names=model_names,
#     parsing_model=spacy.load("en_core_web_sm"),
#     add_parsing_features=False,
# )

# print(surp_res.head(10).to_markdown())
