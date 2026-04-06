from enum import Enum


class SurprisalExtractorType(Enum):
    """Available strategies for computing word-level surprisal.

    The naming convention uses 'l' for left context and 't' for target text,
    with '*' denoting concatenation.

    SOFT_CAT_WHOLE_CTX_LEFT (column suffix: ``softwhole``):
        l_rep = averaged_representations(l)
        full_context = l_rep * t
        Dimensions of the embedding level input: (1 + No. tokens in t, hidden_size)

    SOFT_CAT_SENTENCES (column suffix: ``softsent``):
        l_sentences = concat([averaged_representations(sentence) for sentence in l])
        full_context = l_sentences * t
        Dimensions of the embedding level input: (No. sentences in L + No. tokens in t, hidden_size)

    CAT_CTX_LEFT (column suffix: ``cat``):
        full_context = l * t
        Standard text-level concatenation (buggy version per Pimentel & Meister, 2024).

    PIMENTEL_CTX_LEFT (column suffix: ``pimentel``):
        Corrected surprisal computation per Pimentel & Meister (2024).

    INV_EFFECT_EXTRACTOR (column suffix: ``inveffect``):
        Measures the inverse effect of context: surp(text|context) - surp(text|no_context).
    """

    SOFT_CAT_WHOLE_CTX_LEFT = "SoftCatWholeCtxExtractor"
    SOFT_CAT_SENTENCES = "SoftCatSentencesExtractor"
    CAT_CTX_LEFT = "ConcatenatedSurprisalExtractor"
    PIMENTEL_CTX_LEFT = "PimentelSurprisalExtractor"
    INV_EFFECT_EXTRACTOR = "InverseEffectExtractor"

    @property
    def column_suffix(self) -> str:
        """Short suffix appended to the surprisal column name.

        The resulting column name follows the pattern:
        ``{model_name}_{column_suffix}_Surprisal``
        (e.g., ``gpt2_cat_Surprisal``, ``gpt2_pimentel_Surprisal``).
        """
        return _COLUMN_SUFFIXES[self]


_COLUMN_SUFFIXES = {
    SurprisalExtractorType.SOFT_CAT_WHOLE_CTX_LEFT: "softwhole",
    SurprisalExtractorType.SOFT_CAT_SENTENCES: "softsent",
    SurprisalExtractorType.CAT_CTX_LEFT: "cat",
    SurprisalExtractorType.PIMENTEL_CTX_LEFT: "pimentel",
    SurprisalExtractorType.INV_EFFECT_EXTRACTOR: "inveffect",
}
