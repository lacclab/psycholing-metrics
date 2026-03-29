from enum import Enum


class SurpExtractorType(Enum):
    # left context: l, target context: t
    # '*' - concat

    """
    l_rep = averaged_representations(l)
    full_context = l_rep * t
    Dimensions of the embedding level input: (1 + No. tokens in t, hidden_size)
    """

    SOFT_CAT_WHOLE_CTX_LEFT = "SoftCatWholeCtxSurpExtractor"

    """
    l_sentences = concat([averaged_representations(sentence) for sentence in l])
    full_context = l_sentences * t 
    Dimensions of the embedding level input: (No. sentences in L + No. tokens in t, hidden_size)
    """

    SOFT_CAT_SENTENCES = "SoftCatSentencesSurpExtractor"

    """full_context = l * t"""

    CAT_CTX_LEFT = "CatCtxLeftSurpExtractor"

    PIMENTEL_CTX_LEFT = "PimentelSurpExtractor"

    INV_EFFECT_EXTRACTOR = "InvEffectExtractor"
