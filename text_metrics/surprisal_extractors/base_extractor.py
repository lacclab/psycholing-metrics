from typing import List, Tuple
import numpy as np
from text_metrics.utils import init_tok_n_model
import torch


class BaseSurprisalExtractor:
    def __init__(
        self,
        model_name: str,
        extractor_type_name: str,
        model_target_device: str = "cpu",
        pythia_checkpoint: str | None = "step143000",
        hf_access_token: str | None = None,
    ):
        self.extractor_type_name = extractor_type_name
        self.tokenizer, self.model = init_tok_n_model(
            model_name=model_name,
            device=model_target_device,
            hf_access_token=hf_access_token,
        )

        self.model_name = model_name

        # if the model is pythia model, save the checkpoint name
        if "pythia" in model_name:
            self.pythia_checkpoint = pythia_checkpoint
            assert (
                self.pythia_checkpoint is not None
            ), "Pythia model requires a checkpoint name"

    def surprise(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int = 512,
        allow_overlap: bool = False,
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
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
        shift_logits = output["logits"][
            ..., :-1, :
        ].contiguous()  # remove the last token from the logits

        #  This shift is necessary because the labels are shifted by one position to the right
        # (because the logits are w.r.t the next token)
        shift_labels = tensor_input[
            ..., 1:
        ].contiguous()  #! remove the first token from the labels,

        log_probs = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )


        shift_labels = shift_labels[0]

        if is_last_chunk:
            # remove the eos_token log_prob
            log_probs = log_probs[:-1]
            shift_labels = shift_labels[:-1]

        return log_probs, shift_labels

    def _full_context_to_probs_and_offsets(
        self, full_context: str, overlap_size: int, allow_overlap: bool, max_ctx: int
    ) -> Tuple[torch.Tensor, List[Tuple[int]], List[torch.Tensor]]:
        """This function calculates the surprisal of a full_context using a language model.

        Args:
            full_context (str): The text for which to calculate surprisal.
            overlap_size (int): A number of tokens to overlap between chunks in case the full_context is too long.
            allow_overlap (bool): If True, the full_context will be split into chunks with overlap_size tokens.
            max_ctx (int): The maximum context size of the model.

        Raises:
            ValueError: In case the full_context is too long and allow_overlap is False.

        Returns:
            all_log_probs (torch.Tensor): The log probabilities of the full_context.
            offset_mapping (List[Tuple[int]]): The offset mapping of the full_context.
            accumulated_tokenized_text (List[torch.Tensor]): The tokenized text of the full_context.
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

            # Handle the case where the target_text is too long for the model
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

    def surprise_target_only(
        self, target_text: str, allow_overlap: bool = False, overlap_size: int = 512
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        if allow_overlap:
            assert overlap_size is not None, "overlap_size must be specified"
        full_context = target_text  # even if there isn't left context, we add a space at the beginning to make sure this word is tokenized as the same token always

        with torch.no_grad():
            try:
                max_ctx = self.model.config.max_position_embeddings
            except AttributeError:
                max_ctx = int(1e6)

            assert (
                overlap_size < max_ctx
            ), f"Stride size {overlap_size} is larger than the maximum context size {max_ctx}"

            all_log_probs, offset_mapping, accumulated_tokenized_text = (
                self._full_context_to_probs_and_offsets(
                    full_context, overlap_size, allow_overlap, max_ctx
                )
            )

        # The accumulated_tokenized_text is the text we extract surprisal values for
        # It is after removing the BOS/EOS tokens
        # Make sure the accumulated_tokenized_text is equal to the original target_text
        assert (
            accumulated_tokenized_text
            == self.tokenizer(full_context, add_special_tokens=False)["input_ids"]
        )

        all_log_probs = np.asarray(all_log_probs.cpu())

        assert all_log_probs.shape[0] == len(offset_mapping)

        return all_log_probs, offset_mapping
