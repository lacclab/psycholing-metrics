# Define classes for each mode
from text_metrics.surprisal_extractors.base_extractor import BaseSurprisalExtractor
from typing import List, Tuple
import numpy as np
from text_metrics.utils import remove_redundant_left_context
import torch


class CatCtxLeftSurpExtractor(BaseSurprisalExtractor):
    def __init__(
        self,
        model_name: str,
        extractor_type_name: str,
        model_target_device: str = "cpu",
        pythia_checkpoint: str | None = "step143000",
        hf_access_token: str | None = None,
    ):
        super().__init__(
            model_name,
            extractor_type_name,
            model_target_device,
            pythia_checkpoint,
            hf_access_token,
        )

    def _omit_left_ctx_from_surp_res(
        self,
        target_text_char_onset: int,
        all_log_probs: torch.Tensor,
        offset_mapping: List[Tuple[int]],
    ):
        # cut all_log_probs and offset_mapping to the length of the target_text ONLY (without the left context)
        # notice that accumulated_tokenized_text contains both the left context and the target_text
        # so we need to remove the left context from it

        # in offset_mapping, find the index of the first token of the target_text (the first tuple with the first element equal to target_text_char_onset)
        offset_mapping_first_index = [
            i
            for i, (start, end) in enumerate(offset_mapping)
            if start
            == target_text_char_onset
            - 1  #! -1 because the first token includes the space
        ][0]

        target_text_log_probs = all_log_probs[
            offset_mapping_first_index:
        ]  # because we concatenate the left context to the target text

        target_text_offset_mapping = offset_mapping[offset_mapping_first_index:]
        target_text_offset_mapping = [
            (i - target_text_char_onset, j - target_text_char_onset)
            for i, j in target_text_offset_mapping
        ]

        assert target_text_log_probs.shape[0] == len(target_text_offset_mapping)

        return target_text_log_probs, target_text_offset_mapping

    def surprise(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = True,
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        """This function calculates the surprisal of a target_text using a language model.
         In case the target_text is too long, it is split into chunks while keeping previous context of STRIDE tokens.
         Args:
             target_text (str): The text to calculate surprisal for.
             left_context_text (str | None, optional): A left context to be concatenated with the target_text BEFORE PASSED THROUGH THE MODEL. Defaults to None.
             overlap_size (int | None, optional): In case the target_text + left_context is too long, it is split into chunks with overlap_size tokens. Defaults to None.
             allow_overlap (bool, optional): If True, the target_text will be split into chunks with overlap_size tokens. Defaults to False.

         Raises:
             ValueError: In case the target_text is too long and allow_overlap is False.

         Returns:
             Tuple[np.ndarray, List[Tuple[int]]]: The surprisal values of the target_text and the offset mapping of the target_text.

         Example:
         target_text = "Angela Erdmann never knew her grandfather. He died in 1946, six years before she was born. But, on Tuesday 8th April,
         2014, she described the extraordinary moment when she received a message in a bottle, 101 years after he had lobbed it into the Baltic Sea.
         Thought to be the world's oldest message in a bottle, it was presented to Erdmann by the museum that is now exhibiting it in Germany."

         left_context_text = 'How did Angela Erdmann find out about the bottle?'

         Returns:
         target_text_log_probs = array([4.97636890e+00, 5.20506799e-01, 6.58840360e-03, 5.63895130e+00,
            2.56178689e+00, 3.68980312e+00, 4.96654510e+00, 3.15608168e+00,
            3.41664624e+00, 3.26654100e+00, 9.50351954e-01, 5.09015131e+00,
            ...
            3.03512335e+00, 9.98807430e+00, 1.23654938e+00, 1.98460269e+00,
            4.46555328e+00, 3.09026152e-01], dtype=float32)

        target_text_offset_mapping = [(0, 7), (7, 11), (11, 15), (15, 21), (21, 26), (26, 30), (30, 42), (42, 43),
        (43, 46), (46, 51), (51, 54), (54, 59), (59, 60), (60, 64), (64, 70), (70, 77), (77, 81), (81, 85),
        ....]

        ! Notice that the offset mapping is non-inclusice on the right index (it says 0-7 but actially the first word is chars 0-6)
        """
        # assert that if allow_overlap is True, overlap_size is not None
        if allow_overlap:
            assert overlap_size is not None, "overlap_size must be specified"

        if left_context_text in [None, ""]:
            return self.surprise_target_only(target_text, allow_overlap, overlap_size)

        with torch.no_grad():
            try:
                max_ctx = self.model.config.max_position_embeddings
            except AttributeError:
                max_ctx = int(1e6)

            # only surprisal for the target text will be extracted, so if the left_context is longer
            # than the max context, we remove the redundant left context tokens that cannot be used in-context with the target text
            left_context_text = remove_redundant_left_context(
                self.tokenizer,
                left_context_text=left_context_text,
                max_ctx_in_tokens=max_ctx,
            )

            full_context = left_context_text + " " + target_text
            target_text_char_onset = len(full_context) - len(target_text)

            assert (
                overlap_size < max_ctx
            ), f"Stride size {overlap_size} is larger than the maximum context size {max_ctx}"

            (
                all_log_probs,
                offset_mapping,
                accumulated_tokenized_text,
            ) = self._full_context_to_probs_and_offsets(
                full_context, overlap_size, allow_overlap, max_ctx
            )

        # The accumulated_tokenized_text is the text we extract surprisal values for
        # It is after removing the BOS/EOS tokens
        # Make sure the accumulated_tokenized_text is equal to the original target_text
        assert (
            accumulated_tokenized_text
            == self.tokenizer(full_context, add_special_tokens=False)["input_ids"]
        )

        all_log_probs = np.asarray(all_log_probs.cpu())

        target_text_log_probs, target_text_offset_mapping = (
            self._omit_left_ctx_from_surp_res(
                target_text_char_onset, all_log_probs, offset_mapping
            )
        )

        return target_text_log_probs, target_text_offset_mapping
