from enum import Enum


class SurprisalExtractorType(Enum):
    """Available strategies for computing word-level surprisal.

    The naming convention uses 'l' for left context and 't' for target text,
    with '*' denoting concatenation.

    SOFT_CAT_WHOLE_CTX_LEFT:
        l_rep = averaged_representations(l)
        full_context = l_rep * t
        Dimensions of the embedding level input: (1 + No. tokens in t, hidden_size)

    SOFT_CAT_SENTENCES:
        l_sentences = concat([averaged_representations(sentence) for sentence in l])
        full_context = l_sentences * t
        Dimensions of the embedding level input: (No. sentences in L + No. tokens in t, hidden_size)

    CAT_CTX_LEFT:
        full_context = l * t
        Standard text-level concatenation (buggy version per Pimentel & Meister, 2024).

    PIMENTEL_CTX_LEFT:
        Corrected surprisal computation per Pimentel & Meister (2024).

    INV_EFFECT_EXTRACTOR:
        Measures the inverse effect of context: surp(text|context) - surp(text|no_context).
    """

    SOFT_CAT_WHOLE_CTX_LEFT = "SoftCatWholeCtxExtractor"
    SOFT_CAT_SENTENCES = "SoftCatSentencesExtractor"
    CAT_CTX_LEFT = "ConcatenatedSurprisalExtractor"
    PIMENTEL_CTX_LEFT = "PimentelSurprisalExtractor"
    INV_EFFECT_EXTRACTOR = "InverseEffectExtractor"
