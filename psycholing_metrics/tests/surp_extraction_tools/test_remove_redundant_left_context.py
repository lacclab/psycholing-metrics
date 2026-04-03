import unittest

from transformers import (
    AutoTokenizer,
    GPTNeoXTokenizerFast,
)

from psycholing_metrics.text_processing import trim_left_context


class TestTrimLeftContext(unittest.TestCase):
    def setUp(self):
        self.tokenizers = {
            "gpt2": AutoTokenizer.from_pretrained("gpt2"),
            "gpt_neox": GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b"),
        }

    def test_trim_left_context(self):
        left_context_text = "This is a long left context that needs to be truncated"
        max_tokens = 5

        for name, tokenizer in self.tokenizers.items():
            with self.subTest(tokenizer=name):
                result = trim_left_context(tokenizer, left_context_text, max_tokens)

                result_tokens = tokenizer.encode(result)
                self.assertLessEqual(len(result_tokens), max_tokens)

                print(f"Tokenizer: {name}, Result: '{result}'")


if __name__ == "__main__":
    unittest.main()
