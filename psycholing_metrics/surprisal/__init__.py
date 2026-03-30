"""Surprisal extraction from causal language models.

This package provides multiple strategies for computing word-level surprisal:

- ConcatenatedSurprisalExtractor: standard text-level context concatenation
- PimentelSurprisalExtractor: corrected surprisal per Pimentel & Meister (2024)
- SoftCatWholeCtxExtractor: embedding-level context aggregation (whole context)
- SoftCatSentencesExtractor: embedding-level context aggregation (per sentence)
- InverseEffectExtractor: measures the effect of context on surprisal
"""

from psycholing_metrics.surprisal.base import BaseSurprisalExtractor
from psycholing_metrics.surprisal.concatenated import ConcatenatedSurprisalExtractor
from psycholing_metrics.surprisal.factory import create_surprisal_extractor
from psycholing_metrics.surprisal.inverse_effect import InverseEffectExtractor
from psycholing_metrics.surprisal.pimentel import PimentelSurprisalExtractor
from psycholing_metrics.surprisal.soft_concatenated import (
    SoftCatSentencesExtractor,
    SoftCatWholeCtxExtractor,
)
from psycholing_metrics.surprisal.types import SurprisalExtractorType

__all__ = [
    "BaseSurprisalExtractor",
    "ConcatenatedSurprisalExtractor",
    "PimentelSurprisalExtractor",
    "SoftCatWholeCtxExtractor",
    "SoftCatSentencesExtractor",
    "InverseEffectExtractor",
    "SurprisalExtractorType",
    "create_surprisal_extractor",
]
