"""Initialization of HuggingFace language models and tokenizers."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertTokenizerFast,
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
    LlamaForCausalLM,
    MambaForCausalLM,
)

# Supported model families and their HuggingFace identifiers:
#
# GPT-2:    gpt2, gpt2-medium, gpt2-large, gpt2-xl
# CzeGPT-2: MU-NLPC/CzeGPT-2 (Czech GPT-2)
# mGPT:     ai-forever/mGPT (multilingual GPT-2)
# Chinese GPT-2: uer/gpt2-xlarge-chinese-cluecorpussmall and siblings (uses BertTokenizerFast)
# GPT-Neo:  EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B,
#           EleutherAI/gpt-j-6B, EleutherAI/gpt-neox-20b
# OPT:      facebook/opt-125m, facebook/opt-350m, ..., facebook/opt-66b
# Pythia:   EleutherAI/pythia-70m, ..., EleutherAI/pythia-12b
# Llama:    meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf, meta-llama/Llama-2-70b-hf
# Gemma:    google/gemma-2b, google/gemma-7b, google/recurrentgemma-2b, google/recurrentgemma-9b
# Mamba:    state-spaces/mamba-130m-hf, ..., state-spaces/mamba-2.8b-hf
# Mistral:  mistralai/Mistral-7B-Instruct-v0.*


def load_tokenizer_and_model(
    model_name: str,
    device: str = "cpu",
    pythia_checkpoint: str | None = "step143000",
    hf_access_token: str | None = None,
):
    """Load a HuggingFace causal language model and its tokenizer.

    :param model_name: HuggingFace model identifier (e.g. 'gpt2', 'EleutherAI/pythia-70m').
    :param device: target device for the model. Defaults to 'cpu'.
    :param pythia_checkpoint: checkpoint revision for Pythia models. Defaults to 'step143000'.
    :param hf_access_token: HuggingFace access token, required for gated models (Llama, Gemma, Mistral).
    :raises ValueError: if the model family is not supported.
    :return: (tokenizer, model) tuple.
    """
    model_variant = model_name.split("/")[-1]

    if "chinese" in model_variant:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    elif any(
        variant in model_variant
        for variant in ["gpt-neo", "gpt", "CzeGPT", "mGPT", "opt", "mamba", "rwkv"]
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    elif "gpt-neox" in model_variant:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)

    elif "Eagle" in model_variant:  # RWKV V5
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    elif any(variant in model_variant for variant in ["Llama", "Mistral", "gemma"]):
        assert hf_access_token is not None, (
            f"Please provide the HuggingFace access token to load {model_name}"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, token=hf_access_token
        )

    elif "pythia" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=pythia_checkpoint, use_fast=True
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if any(
        variant in model_variant
        for variant in ["gpt-neo", "gpt", "CzeGPT", "mGPT", "opt", "rwkv"]
    ):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    elif "Eagle" in model_variant:  # RWKV
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        ).to(torch.float32)

    elif "pythia" in model_variant:
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=pythia_checkpoint, device_map="auto"
        )

    elif "mamba" in model_variant:
        model = MambaForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )

    elif "Llama" in model_variant:
        model = LlamaForCausalLM.from_pretrained(
            model_name, token=hf_access_token, device_map="auto"
        )

    elif any(variant in model_variant for variant in ["gemma-2"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_access_token,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    elif any(variant in model_variant for variant in ["Mistral", "gemma"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_access_token, device_map="auto"
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return tokenizer, model
