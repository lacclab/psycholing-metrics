"""text_metrics: Word-level linguistic metric extraction.

Extract surprisal, frequency, word length, and parsing features (POS, NER, morphology,
dependencies) for any text. Designed for psycholinguistics and eye-tracking research.

Quick start::

    from text_metrics import get_metrics, create_surprisal_extractor, SurprisalExtractorType

    extractor = create_surprisal_extractor(
        extractor_type=SurprisalExtractorType.CAT_CTX_LEFT,
        model_name="gpt2",
    )
    metrics = get_metrics(
        target_text="The cat sat on the mat.",
        surp_extractor=extractor,
        parsing_model=None,
        add_parsing_features=False,
    )
"""

from text_metrics.metrics import get_frequency, get_metrics, get_surprisal, get_word_length
from text_metrics.surprisal import (
    BaseSurprisalExtractor,
    ConcatenatedSurprisalExtractor,
    InverseEffectExtractor,
    PimentelSurprisalExtractor,
    SoftCatSentencesExtractor,
    SoftCatWholeCtxExtractor,
    SurprisalExtractorType,
    create_surprisal_extractor,
)

__all__ = [
    # Core metric functions
    "get_metrics",
    "get_surprisal",
    "get_frequency",
    "get_word_length",
    # Extractor factory
    "create_surprisal_extractor",
    # Extractor type enum
    "SurprisalExtractorType",
    # Extractor classes
    "BaseSurprisalExtractor",
    "ConcatenatedSurprisalExtractor",
    "PimentelSurprisalExtractor",
    "SoftCatWholeCtxExtractor",
    "SoftCatSentencesExtractor",
    "InverseEffectExtractor",
]
