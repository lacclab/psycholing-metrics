"""Base class for all surprisal extractors."""

from typing import List, Tuple

import numpy as np
import torch

from psycholing_metrics.model_loader import load_tokenizer_and_model
from psycholing_metrics.surprisal.types import SurprisalExtractorType


class BaseSurprisalExtractor:
    """Abstract base for computing word-level surprisal from causal language models.

    Handles model/tokenizer initialization, input tokenization, log probability
    computation, and chunking for texts that exceed the model's context window.
    """

    def __init__(
        self,
        model_name: str,
        extractor_type_name: str,
        model_target_device: str = "cpu",
        pythia_checkpoint: str | None = "step143000",
        hf_access_token: str | None = None,
    ):
        self.extractor_type_name = extractor_type_name
        self.extractor_type = SurprisalExtractorType(extractor_type_name)
        self.tokenizer, self.model = load_tokenizer_and_model(
            model_name=model_name,
            device=model_target_device,
            hf_access_token=hf_access_token,
        )

        self.model_name = model_name

        if "pythia" in model_name:
            self.pythia_checkpoint = pythia_checkpoint
            assert self.pythia_checkpoint is not None, (
                "Pythia model requires a checkpoint name"
            )

    def compute_surprisal(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int = 512,
        allow_overlap: bool = False,
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        """Compute token-level surprisal values for the target text.

        :param target_text: the text to compute surprisal for.
        :param left_context_text: optional left context to condition on.
        :param overlap_size: token overlap between chunks for long texts.
        :param allow_overlap: whether to allow chunking for long texts.
        :return: (log_probs, offset_mapping) - arrays of per-token surprisal and character offsets.
        """
        raise NotImplementedError

    def _create_input_tokens(
        self,
        encodings: dict,
        start_ind: int,
        is_last_chunk: bool,
        device: str,
    ):
        try:
            bos_token_added = self.tokenizer.bos_token_id
        except AttributeError:
            bos_token_added = self.tokenizer.pad_token_id

        tokens_lst = encodings["input_ids"]
        if is_last_chunk:
            tokens_lst.append(self.tokenizer.eos_token_id)
        if start_ind == 0:
            tokens_lst = [bos_token_added] + tokens_lst
        tensor_input = torch.tensor(
            [tokens_lst],
            device=device,
        )
        return tensor_input

    def _tokens_to_log_probs(
        self,
        tensor_input: torch.Tensor,
        is_last_chunk: bool,
    ):
        output = self.model(tensor_input, labels=tensor_input)
        shift_logits = output["logits"][..., :-1, :].contiguous()
        shift_labels = tensor_input[..., 1:].contiguous()

        log_probs = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )

        shift_labels = shift_labels[0]

        if is_last_chunk:
            log_probs = log_probs[:-1]
            shift_labels = shift_labels[:-1]

        return log_probs, shift_labels

    def _compute_log_probs_with_chunking(
        self, full_context: str, overlap_size: int, allow_overlap: bool, max_ctx: int
    ) -> Tuple[torch.Tensor, List[Tuple[int]], List[torch.Tensor]]:
        """Compute log probabilities for text, splitting into chunks if it exceeds max context.

        :param full_context: the text to compute log probs for.
        :param overlap_size: number of tokens to overlap between chunks.
        :param allow_overlap: if True, split long text into overlapping chunks.
        :param max_ctx: the model's maximum context size.
        :raises ValueError: if text is too long and allow_overlap is False.
        :return: (all_log_probs, offset_mapping, accumulated_tokenized_text)
        """
        start_ind = 0
        accumulated_tokenized_text = []
        all_log_probs = torch.tensor([], device=self.model.device)
        offset_mapping = []
        while True:
            encodings = self.tokenizer(
                full_context[start_ind:],
                max_length=max_ctx - 2,
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            is_last_chunk = (encodings["offset_mapping"][-1][1] + start_ind) == len(
                full_context
            )

            tensor_input = self._create_input_tokens(
                encodings,
                start_ind,
                is_last_chunk,
                self.model.device,
            )

            log_probs, shift_labels = self._tokens_to_log_probs(
                tensor_input, is_last_chunk
            )

            offset = 0 if start_ind == 0 else overlap_size - 1
            all_log_probs = torch.cat([all_log_probs, log_probs[offset:]])
            accumulated_tokenized_text += shift_labels[offset:]

            left_index_add_offset_mapping = offset if start_ind == 0 else offset + 1
            offset_mapping_to_add = encodings["offset_mapping"][
                left_index_add_offset_mapping:
            ]

            offset_mapping.extend(
                [(i + start_ind, j + start_ind) for i, j in offset_mapping_to_add]
            )
            if is_last_chunk:
                break

            if start_ind == 0:
                context_length = len(self.tokenizer.encode(full_context))
                if allow_overlap:
                    print(
                        f"The context length is too long ({context_length}>{max_ctx}) for {self.model_name}. Splitting the full text into chunks with overlap {overlap_size}"
                    )
                else:
                    raise ValueError(
                        f"The context length is too long ({context_length}>{max_ctx}) for {self.model_name}. Try enabling allow_overlap and specify overlap size"
                    )

            start_ind += encodings["offset_mapping"][-overlap_size - 1][1]

        return all_log_probs, offset_mapping, accumulated_tokenized_text

    def compute_surprisal_no_context(
        self, target_text: str, allow_overlap: bool = False, overlap_size: int = 512
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        """Compute surprisal for target text without any left context.

        :param target_text: the text to compute surprisal for.
        :param allow_overlap: whether to allow chunking for long texts.
        :param overlap_size: token overlap between chunks.
        :return: (log_probs, offset_mapping)
        """
        if allow_overlap:
            assert overlap_size is not None, "overlap_size must be specified"
        full_context = target_text

        with torch.no_grad():
            try:
                max_ctx = self.model.config.max_position_embeddings
            except AttributeError:
                max_ctx = int(1e6)

            assert overlap_size < max_ctx, (
                f"Stride size {overlap_size} is larger than the maximum context size {max_ctx}"
            )

            all_log_probs, offset_mapping, accumulated_tokenized_text = (
                self._compute_log_probs_with_chunking(
                    full_context, overlap_size, allow_overlap, max_ctx
                )
            )

        assert (
            accumulated_tokenized_text
            == self.tokenizer(full_context, add_special_tokens=False)["input_ids"]
        )

        all_log_probs = np.asarray(all_log_probs.cpu())

        assert all_log_probs.shape[0] == len(offset_mapping)

        return all_log_probs, offset_mapping
