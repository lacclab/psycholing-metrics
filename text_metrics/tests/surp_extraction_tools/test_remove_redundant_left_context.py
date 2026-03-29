import unittest

from transformers import (
    AutoTokenizer,
    GPTNeoXTokenizerFast,
)

from text_metrics.utils import remove_redundant_left_context


class TestRemoveRedundantLeftContext(unittest.TestCase):
    def setUp(self):
        # Initialize the different tokenizers
        self.tokenizers = {
            "gpt2": AutoTokenizer.from_pretrained("gpt2"),
            "gpt_neox": GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b"),
        }

    def test_remove_redundant_left_context(self):
        left_context_text = "This is a long left context that needs to be truncated"
        max_ctx_in_tokens = 5

        for name, tokenizer in self.tokenizers.items():
            with self.subTest(tokenizer=name):
                result = remove_redundant_left_context(
                    tokenizer, left_context_text, max_ctx_in_tokens
                )

                # Assert that the result has a maximum of max_ctx_in_tokens tokens
                result_tokens = tokenizer.encode(result)
                self.assertLessEqual(len(result_tokens), max_ctx_in_tokens)

                # Print the results for verification (optional)
                print(f"Tokenizer: {name}, Result: '{result}'")


if __name__ == "__main__":
    unittest.main()
