import unittest

from psycholing_metrics.text_processing import clean_text


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
