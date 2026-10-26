"""Word frequencies, including languages wordfreq has no list for."""

import unittest

import numpy as np
from wordfreq import word_frequency

from psycholing_metrics.metrics import _load_frequency_list, get_frequency

FLOOR = -np.log2(1e-11)


class TestFrequency(unittest.TestCase):
    def test_wordfreq_languages_unchanged(self):
        text = "el perro, de 1999 años; ¡no!"
        expected = [
            -np.log2(word_frequency(w, lang="es", minimum=1e-11))
            for w in text.split()
        ]
        self.assertEqual(
            get_frequency(text, "es")["Wordfreq_Frequency"].tolist(), expected
        )

    def test_basque_uses_basque_list(self):
        words = ["eta", "egutegia", "saltzen"]
        freqs = get_frequency(" ".join(words), "eu")["Wordfreq_Frequency"]
        # wordfreq alone looks Basque up in Spanish: "eta" (and) gets 16.3
        # bits and the other two the floor.
        self.assertEqual(
            -np.log2(word_frequency("egutegia", lang="eu", minimum=1e-11)), FLOOR
        )
        self.assertTrue((freqs < 15).all(), freqs.tolist())
        self.assertLess(freqs[0], 5)

    def test_basque_lookup_matches_the_list(self):
        freqs = _load_frequency_list("eu")
        bits = get_frequency("etxea etxe-jabe xqzwvk", "eu")["Wordfreq_Frequency"]
        self.assertAlmostEqual(bits[0], -np.log2(freqs["etxea"]), places=2)
        # Two tokens: 1 / f = 1 / f1 + 1 / f2.
        combined = 1 / (1 / freqs["etxe"] + 1 / freqs["jabe"])
        self.assertAlmostEqual(bits[1], -np.log2(combined), places=2)
        self.assertEqual(bits[2], FLOOR)

    def test_basque_numbers(self):
        bits = get_frequency("2023an", "eu")["Wordfreq_Frequency"]
        self.assertLess(bits[0], FLOOR)


if __name__ == "__main__":
    unittest.main()
