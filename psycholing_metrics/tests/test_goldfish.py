"""Goldfish models: loading by name, BOS, and token-to-word alignment.

Goldfish (goldfish-models/{lang}_{script}_{size}) ship a SentencePiece
(metaspace) tokenizer instead of GPT-2's byte-level BPE. The alignment tests
use the real Greek tokenizer with a tiny randomly initialised GPT-2, so no
model weights are downloaded.
"""

import unittest
from unittest import mock

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from psycholing_metrics import model_loader
from psycholing_metrics.metrics import get_surprisal
from psycholing_metrics.surprisal.concatenated import ConcatenatedSurprisalExtractor
from psycholing_metrics.surprisal.types import SurprisalExtractorType

TOKENIZER = "goldfish-models/ell_grek_1000mb"


class TestGoldfishLoader(unittest.TestCase):
    def test_goldfish_names_load_with_auto_classes(self):
        with (
            mock.patch.object(model_loader, "AutoTokenizer") as auto_tokenizer,
            mock.patch.object(model_loader, "AutoModelForCausalLM") as auto_model,
        ):
            model_loader.load_tokenizer_and_model("goldfish-models/tur_latn_1000mb")
        auto_tokenizer.from_pretrained.assert_called_once_with(
            "goldfish-models/tur_latn_1000mb", use_fast=True
        )
        auto_model.from_pretrained.assert_called_once_with(
            "goldfish-models/tur_latn_1000mb", device_map="cpu"
        )

    def test_unknown_names_still_raise(self):
        with self.assertRaises(ValueError):
            model_loader.load_tokenizer_and_model("someone/tur_latn_1000mb")


class TestGoldfishAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        torch.manual_seed(0)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=512,  # Goldfish's trained length
                n_embd=8,
                n_layer=1,
                n_head=1,
            )
        ).eval()
        # Skip __init__, which downloads a real model.
        extractor = ConcatenatedSurprisalExtractor.__new__(
            ConcatenatedSurprisalExtractor
        )
        extractor.extractor_type_name = SurprisalExtractorType.CAT_CTX_LEFT.value
        extractor.extractor_type = SurprisalExtractorType.CAT_CTX_LEFT
        extractor.tokenizer = tokenizer
        extractor.model = model
        extractor.model_name = "tiny-gpt2"
        cls.extractor = extractor

    def test_bos_is_cls(self):
        # The model card prepends [CLS]; the tokenizer declares it as BOS.
        tokenizer = self.extractor.tokenizer
        self.assertEqual(tokenizer.convert_ids_to_tokens(50000), "[CLS]")
        self.assertEqual(self.extractor._get_bos_like_id(), 50000)

    def test_every_word_gets_its_tokens(self):
        # Capitals, guillemets, straight quotes, a Latin word, a character
        # the vocabulary lacks (U+2219 -> <unk>) and a lone parenthesis,
        # which tokenizes as a bare "▁" followed by the character.
        words = (
            'Ο «Κόσμος» είναι ΜΕΓΑΛΟΣ. "Aloha" αποικιοκρατία∙ ( ΠΗΠ ), τέλος.'
        ).split()
        text = " ".join(words)
        token_log_probs, _ = self.extractor.compute_surprisal(
            text, overlap_size=512, allow_overlap=False
        )
        df = get_surprisal(text, surp_extractor=self.extractor)
        self.assertEqual(df["Word"].tolist(), words)
        self.assertTrue((df["Surprisal"] > 0).all())
        # No token is dropped or counted twice.
        np.testing.assert_allclose(
            df["Surprisal"].sum(), token_log_probs.sum(), rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
