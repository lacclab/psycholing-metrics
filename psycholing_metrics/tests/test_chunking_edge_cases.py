"""Chunking and word alignment edge cases in surprisal extraction.

Uses the BERT WordPiece tokenizer of the Chinese GPT-2 models (one token per
Chinese character, private-use characters dropped) with a tiny randomly
initialised GPT-2, so a 16-position context makes chunking cheap to exercise.
"""

import contextlib
import io
import unittest

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from psycholing_metrics.metrics import get_surprisal
from psycholing_metrics.surprisal.concatenated import ConcatenatedSurprisalExtractor
from psycholing_metrics.surprisal.types import SurprisalExtractorType

TOKENIZER = "uer/gpt2-xlarge-chinese-cluecorpussmall"
N_POSITIONS = 16  # a chunk holds N_POSITIONS - 2 tokens
PUA = "\ue004"  # private-use character: BertTokenizerFast emits no token for it


def _words(n: int) -> list:
    """n distinct single-character Chinese words (one token each)."""
    return [chr(0x4E00 + 7 * i) for i in range(n)]


class TestChunkingEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        torch.manual_seed(0)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=tokenizer.vocab_size,
                n_positions=N_POSITIONS,
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

    def surprisal(self, words: list) -> list:
        df = get_surprisal(" ".join(words), surp_extractor=self.extractor)
        self.assertEqual(df["Word"].tolist(), words)
        return df["Surprisal"].tolist()

    def test_private_use_character_ending_a_text(self):
        # ChineseReading ends two texts with this interest area; it raised
        # IndexError because the text never registered as fully tokenized.
        words = ["计划", "的", "周旋", "。"]
        with_pua = self.surprisal(words + [PUA])
        self.assertEqual(with_pua[:-1], self.surprisal(words))
        self.assertEqual(with_pua[-1], 0.0)

    def test_private_use_character_inside_a_text(self):
        # The model never sees the character, so every other word keeps the
        # surprisal it has in the text without it.
        with_pua = self.surprisal(["计划", PUA, "的", "周旋"])
        self.assertEqual(
            [with_pua[0]] + with_pua[2:], self.surprisal(["计划", "的", "周旋"])
        )
        self.assertEqual(with_pua[1], 0.0)

    def test_long_text_chunks_with_a_small_context(self):
        # Default overlap_size (512) on a 16-position model: the overlap must
        # be capped below the chunk length, not just below max_ctx.
        words = _words(40)
        chunked = self.surprisal(words)
        # The first chunk sees exactly what a text of its own length sees.
        first_chunk = N_POSITIONS - 2
        self.assertEqual(chunked[:first_chunk], self.surprisal(words[:first_chunk]))

    def test_long_text_ending_in_a_private_use_character(self):
        words = _words(40)
        with_pua = self.surprisal(words + [PUA])
        self.assertEqual(with_pua[:-1], self.surprisal(words))
        self.assertEqual(with_pua[-1], 0.0)

    def test_text_of_exactly_one_chunk_and_one_token_more(self):
        for n in (N_POSITIONS - 2, N_POSITIONS - 1):
            with self.subTest(n_tokens=n):
                self.assertEqual(len(self.surprisal(_words(n))), n)

    def assert_covers_all_tokens(self, text: str) -> str:
        """Run compute_surprisal on raw text; return what it printed."""
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            log_probs, offset_mapping = self.extractor.compute_surprisal(
                target_text=text, overlap_size=8, allow_overlap=True
            )
        expected = self.extractor.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        self.assertEqual(len(log_probs), len(expected))
        self.assertEqual([tuple(o) for o in offset_mapping], expected)
        return printed.getvalue()

    def test_trailing_whitespace(self):
        # The tokenizer drops trailing whitespace too, so the last token ends
        # short of len(text). The "too long" notice is only for real overflow.
        for n, too_long in ((5, False), (40, True)):
            for suffix in ("", " ", "\n"):
                with self.subTest(n_tokens=n, suffix=repr(suffix)):
                    printed = self.assert_covers_all_tokens("".join(_words(n)) + suffix)
                    self.assertEqual("too long" in printed, too_long)


if __name__ == "__main__":
    unittest.main()
