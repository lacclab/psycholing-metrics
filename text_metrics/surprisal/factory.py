"""Factory function for creating surprisal extractors."""

from text_metrics.surprisal.concatenated import ConcatenatedSurprisalExtractor
from text_metrics.surprisal.inverse_effect import InverseEffectExtractor
from text_metrics.surprisal.pimentel import PimentelSurprisalExtractor
from text_metrics.surprisal.soft_concatenated import (
    SoftCatSentencesExtractor,
    SoftCatWholeCtxExtractor,
)
from text_metrics.surprisal.types import SurprisalExtractorType


def create_surprisal_extractor(
    extractor_type: SurprisalExtractorType,
    model_name: str,
    model_target_device: str = "cpu",
    pythia_checkpoint: str | None = "step143000",
    hf_access_token: str | None = None,
):
    """Create a surprisal extractor of the specified type.

    :param extractor_type: which extraction strategy to use.
    :param model_name: HuggingFace model identifier.
    :param model_target_device: device for the model. Defaults to 'cpu'.
    :param pythia_checkpoint: checkpoint for Pythia models. Defaults to 'step143000'.
    :param hf_access_token: HuggingFace access token for gated models.
    :return: an initialized surprisal extractor instance.
    """
    kwargs = dict(
        model_name=model_name,
        extractor_type_name=extractor_type.value,
        model_target_device=model_target_device,
        pythia_checkpoint=pythia_checkpoint,
        hf_access_token=hf_access_token,
    )

    if extractor_type == SurprisalExtractorType.SOFT_CAT_WHOLE_CTX_LEFT:
        return SoftCatWholeCtxExtractor(**kwargs)
    elif extractor_type == SurprisalExtractorType.SOFT_CAT_SENTENCES:
        return SoftCatSentencesExtractor(**kwargs)
    elif extractor_type == SurprisalExtractorType.CAT_CTX_LEFT:
        return ConcatenatedSurprisalExtractor(**kwargs)
    elif extractor_type == SurprisalExtractorType.PIMENTEL_CTX_LEFT:
        return PimentelSurprisalExtractor(**kwargs)
    elif extractor_type == SurprisalExtractorType.INV_EFFECT_EXTRACTOR:
        return InverseEffectExtractor(
            **kwargs,
            target_extractor_type=SurprisalExtractorType.CAT_CTX_LEFT,
        )
    else:
        raise ValueError(f"Unrecognized extractor type: {extractor_type}")
