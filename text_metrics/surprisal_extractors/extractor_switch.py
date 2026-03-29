from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType
from text_metrics.surprisal_extractors.inv_effect_extractor import InvEffectExtractor
from text_metrics.surprisal_extractors.pimentel_extractor import PimentelSurpExtractor
from text_metrics.surprisal_extractors.soft_cat_extractors import (
    SoftCatSentencesSurpExtractor,
    SoftCatWholeCtxSurpExtractor,
)
from text_metrics.surprisal_extractors.text_cat_extractor import CatCtxLeftSurpExtractor


def get_surp_extractor(
    extractor_type: SurpExtractorType,
    model_name: str,
    model_target_device: str = "cpu",
    pythia_checkpoint: str | None = "step143000",
    hf_access_token: str | None = None,
):
    kwargs = dict(
        model_name=model_name,
        extractor_type_name=extractor_type.value,
        model_target_device=model_target_device,
        pythia_checkpoint=pythia_checkpoint,
        hf_access_token=hf_access_token,
    )

    if extractor_type == SurpExtractorType.SOFT_CAT_WHOLE_CTX_LEFT:
        return SoftCatWholeCtxSurpExtractor(**kwargs)
    elif extractor_type == SurpExtractorType.SOFT_CAT_SENTENCES:
        return SoftCatSentencesSurpExtractor(**kwargs)
    elif extractor_type == SurpExtractorType.CAT_CTX_LEFT:
        return CatCtxLeftSurpExtractor(**kwargs)
    elif extractor_type == SurpExtractorType.PIMENTEL_CTX_LEFT:
        return PimentelSurpExtractor(**kwargs)
    elif extractor_type == SurpExtractorType.INV_EFFECT_EXTRACTOR:
        return InvEffectExtractor(
            **kwargs,
            target_extractor_type=SurpExtractorType.CAT_CTX_LEFT,
        )
    else:
        raise ValueError(f"Unrecognized extractor type: {extractor_type}")
