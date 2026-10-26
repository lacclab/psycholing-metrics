"""Token-to-word aggregation of log probabilities.

Byte-level BPE (GPT-2) splits a multi-byte character into several tokens that
share one character span. Every one of them belongs to that character's word.
Token log probabilities are powers of two, so each word's sum identifies
exactly which tokens it received.
"""

import unittest

import numpy as np
from transformers import AutoTokenizer

from psycholing_metrics.text_processing import aggregate_token_log_probs, clean_text


class TestWordAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def aggregate(self, text: str):
        offsets = self.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        log_probs = 2.0 ** np.arange(len(offsets))
        word_log_probs, _ = aggregate_token_log_probs(text, log_probs, offsets)
        return word_log_probs, offsets, log_probs

    def assert_word_tokens(self, text: str, tokens_per_word: list):
        word_log_probs, offsets, log_probs = self.aggregate(text)
        self.assertEqual(len(offsets), sum(tokens_per_word))
        bounds = np.cumsum([0] + tokens_per_word)
        expected = [log_probs[a:b].sum() for a, b in zip(bounds[:-1], bounds[1:])]
        self.assertEqual(word_log_probs, expected)

    def test_ascii(self):
        # "hello" "," " how" " are" " you" "?"
        self.assert_word_tokens("hello, how are you?", [2, 1, 1, 2])

    def test_greek_capital_split_into_bytes(self):
        # "Ο" is two byte tokens with span (0, 1); the second used to be
        # summed into the next word.
        self.assert_word_tokens("Ο άνθρωπος", [2, 10])

    def test_cjk_word_ending_in_split_character(self):
        # 日 and 本 are two byte tokens each; 本's second used to leak to 語.
        self.assert_word_tokens("日本 語", [4, 3])

    def test_token_crossing_a_word_boundary_raises(self):
        with self.assertRaises(ValueError):
            aggregate_token_log_probs("ab cd", np.array([1.0]), [(0, 4)])

    def test_token_after_the_last_word_raises(self):
        with self.assertRaises(ValueError):
            aggregate_token_log_probs("ab", np.array([1.0, 1.0]), [(0, 2), (2, 3)])


class TestCleanText(unittest.TestCase):
    def test_keeps_accented_letters(self):
        text = "décadas café Noël naïve"
        self.assertEqual(clean_text(text), text)

    def test_maps_typographic_characters(self):
        self.assertEqual(
            clean_text("“don’t” ‘a’ 1–2 so… ﬁne"),
            "\"don't\" 'a' 1-2 so... fine",
        )


if __name__ == "__main__":
    unittest.main()
