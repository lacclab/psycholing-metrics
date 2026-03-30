import unittest

from psycholing_metrics.metrics import get_surprisal
from psycholing_metrics.surprisal.concatenated import ConcatenatedSurprisalExtractor
from psycholing_metrics.text_processing import clean_text


class TestSurprisalExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("psycholing_metrics/tests/below_2048.txt", "r") as file:
            cls.below_2048 = file.read()

        with open("psycholing_metrics/tests/over_2048.txt", "r") as file:
            cls.over_2048 = file.read()

        cls.model_names = [
            "facebook/opt-350m",
            "EleutherAI/pythia-70m",
            "gpt2",
            "EleutherAI/gpt-neo-125M",
        ]

    def test_surprisal_extraction(self):
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                surp_extractor = ConcatenatedSurprisalExtractor(
                    model_name=model_name,
                    extractor_type_name="ConcatenatedSurprisalExtractor",
                    model_target_device="cuda:0",
                )
                print(f"Model: {model_name}")
                try:
                    _ = get_surprisal(
                        clean_text(self.below_2048),
                        surp_extractor=surp_extractor,
                        overlap_size=500,
                    )
                    _ = get_surprisal(
                        clean_text(self.over_2048),
                        surp_extractor=surp_extractor,
                        overlap_size=500,
                    )
                    print("Didn't crash")
                except Exception as e:
                    self.fail(f"Model {model_name} raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
