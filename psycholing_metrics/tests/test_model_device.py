"""The model goes on the device the caller asks for.

``device_map="auto"`` used to be hard-coded, so ``device`` was ignored: with
one GPU visible the model went to it even for ``device="cpu"``, and with
several it was split across all of them.
"""

import unittest
from unittest import mock

from psycholing_metrics import model_loader


class TestModelDevice(unittest.TestCase):
    def _device_map(self, model_name, **kwargs):
        with (
            mock.patch.object(model_loader, "AutoTokenizer"),
            mock.patch.object(model_loader, "AutoModelForCausalLM") as auto_model,
        ):
            model_loader.load_tokenizer_and_model(model_name, **kwargs)
        return auto_model.from_pretrained.call_args.kwargs["device_map"]

    def test_requested_device_is_the_device_map(self):
        for device in ["cpu", "cuda", "cuda:1"]:
            with self.subTest(device=device):
                self.assertEqual(self._device_map("gpt2", device=device), device)

    def test_default_is_cpu(self):
        self.assertEqual(self._device_map("gpt2"), "cpu")

    def test_auto_still_shards(self):
        self.assertEqual(self._device_map("gpt2", device="auto"), "auto")

    def test_meta_tensor_fallback_keeps_the_device(self):
        # Broken tied-weight safetensors retry with .bin on the same device.
        with (
            mock.patch.object(model_loader, "AutoTokenizer"),
            mock.patch.object(model_loader, "AutoModelForCausalLM") as auto_model,
        ):
            auto_model.from_pretrained.side_effect = [
                NotImplementedError("Cannot copy out of meta tensor"),
                mock.MagicMock(),
            ]
            model_loader.load_tokenizer_and_model("gpt2", device="cuda:1")
        retry = auto_model.from_pretrained.call_args_list[1]
        self.assertEqual(retry.kwargs["device_map"], "cuda:1")
        self.assertFalse(retry.kwargs["use_safetensors"])


if __name__ == "__main__":
    unittest.main()
