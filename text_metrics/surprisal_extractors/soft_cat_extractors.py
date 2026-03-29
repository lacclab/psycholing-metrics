from typing import List, Tuple

import numpy as np
import spacy
import torch
from sentence_splitter import split_text_into_sentences

from text_metrics.surprisal_extractors.base_extractor import BaseSurprisalExtractor


class SoftCatCtxSurpExtractor(BaseSurprisalExtractor):
    def __init__(
        self,
        model_name: str,
        extractor_type_name: str,
        model_target_device: str = "cpu",
        pythia_checkpoint: str | None = "step143000",
        hf_access_token: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            extractor_type_name=extractor_type_name,
            model_target_device=model_target_device,
            pythia_checkpoint=pythia_checkpoint,
            hf_access_token=hf_access_token,
        )

        if "pythia" in self.model_name:
            self.model_wte = self.model.gpt_neox.embed_in
        elif "gpt2" in self.model_name:
            self.model_wte = self.model.transformer.wte
        else:
            raise NotImplementedError(
                f"{self.model_name} isn't supported for extracting embedding-level word-embeddings"
            )

    def _get_embedded_left_context(self, left_context_text: str, device: str):
        raise NotImplementedError

    def _encode_target_text(self, target_text: str):
        target_encodings = self.tokenizer(
            target_text,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        ).to(self.model.device)
        target_labels = target_encodings["input_ids"]
        target_offset_mappings = target_encodings["offset_mapping"]

        return target_encodings, target_labels, target_offset_mappings

    def _create_embedding_level_input(
        self,
        target_encodings: torch.Tensor,
        left_context_text: str,
    ) -> Tuple[torch.Tensor, int]:
        # Tokenize the target text
        target_word_embeddings = self.model_wte(target_encodings["input_ids"])

        left_context_embedding = (
            self._get_embedded_left_context(left_context_text, self.model.device)
            .unsqueeze(0)
            .to(self.model.device)
        )

        try:
            bos_token_added = self.tokenizer.bos_token_id
        except AttributeError:
            bos_token_added = self.tokenizer.pad_token_id

        eos_token_added = self.tokenizer.eos_token_id

        bos_embd = (
            self.model_wte(torch.tensor(bos_token_added).to(self.model.device))
            .unsqueeze(0)
            .unsqueeze(1)
        )
        eos_embd = (
            self.model_wte(torch.tensor(eos_token_added).to(self.model.device))
            .unsqueeze(0)
            .unsqueeze(1)
        )

        full_embeddings = torch.cat(
            [bos_embd, left_context_embedding, target_word_embeddings, eos_embd], dim=1
        )  # Concatenate at embedding level

        target_text_onset: int = left_context_embedding.shape[1]

        return full_embeddings, target_text_onset

    def _full_embds_to_log_probs(
        self, full_embeddings: torch.Tensor, target_labels, target_text_onset: int
    ):
        #  <bos> <left_ctx> w1, ..., wn-1, wn, <eos>
        output = self.model(inputs_embeds=full_embeddings)
        # logits of: <left_ctx> w1, ..., wn-1  (starts with the logits for CLS because it's the probs for w1, and ends with the logits of wn-1 that indicate the probs of wn)
        # 0, 1, 2, 3, 4
        # (let's say target text begins at 4. 0 is BOS and 1, 2, 3 are the left context. In this case, target_text_onset
        #  is 3 because it's onset with respect to the left context without the BOS token)
        # output["logits"][0] -> logits of the first token of the left context
        # output["logits"][1] -> logits of the second token of the left context
        # output["logits"][2] -> logits of the third token of the left context
        # output["logits"][3] -> logits of the first token of the target text
        shift_logits = output["logits"][
            ..., target_text_onset:-2, :
        ].contiguous()  # remove the last token from the logits

        #  Here the target lebels are pure target labels without the left context and BOS / EOS token
        shift_labels = target_labels.contiguous()

        log_probs = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )

        shift_labels = shift_labels[0]

        return log_probs, shift_labels

    def surprise(
        self,
        target_text: str,
        left_context_text: str | None = None,
        overlap_size: int | None = None,
        allow_overlap: bool = False,
    ) -> Tuple[np.ndarray, List[Tuple[int]]]:
        """Calculate the surprisal with the left context embedded and concatenated at the embedding level."""
        if allow_overlap:
            assert overlap_size is not None, "overlap_size must be specified"

        if left_context_text in [None, ""]:
            return self.surprise_target_only(target_text)

        with torch.no_grad():
            left_context_text = left_context_text.strip() + " "
            target_encodings, target_labels, target_offset_mappings = (
                self._encode_target_text(target_text)
            )

            full_embeddings, target_text_onset = self._create_embedding_level_input(
                target_encodings, left_context_text
            )

            log_probs, _ = self._full_embds_to_log_probs(
                full_embeddings, target_labels, target_text_onset
            )

        # Convert to numpy for return
        log_probs = log_probs.cpu().numpy()
        offset_mapping = target_offset_mappings.cpu().tolist()[0]
        offset_mapping = [tuple(mapping) for mapping in offset_mapping]

        return log_probs, offset_mapping


class SoftCatWholeCtxSurpExtractor(SoftCatCtxSurpExtractor):
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

    def _get_embedded_left_context(self, left_context_text: str, device: str):
        """Helper method to embed the left context using the model."""
        with torch.no_grad():
            # Tokenize the left context
            left_context_tokens = self.tokenizer(
                left_context_text, return_tensors="pt", truncation=True
            ).to(device)

            # Get the hidden state for the left context from the model
            left_context_output = self.model(
                **left_context_tokens, output_hidden_states=True
            )

            # Get the hidden states for the last layer
            hidden_states = left_context_output.hidden_states[
                -1
            ]  # Shape: (batch_size, seq_len, hidden_dim)

            # Aggregate hidden states by averaging over the sequence length
            left_context_embedding = torch.mean(
                hidden_states, dim=1
            )  # Shape: (batch_size, hidden_dim)

        return left_context_embedding


class SoftCatSentencesSurpExtractor(SoftCatCtxSurpExtractor):
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
        self.spacy_module = spacy.load("en_core_web_sm")

    def _get_embedded_left_context(self, left_context_text: str, device: str):
        """Helper method to embed the left context using the model."""
        with torch.no_grad():
            acc_sentence_embedding = []

            sentences = split_text_into_sentences(text=left_context_text, language="en")
            for sentence in sentences:
                # Tokenize the left context
                left_context_tokens = self.tokenizer(
                    sentence, return_tensors="pt", truncation=True
                ).to(device)

                # Get the hidden state for the left context from the model
                left_context_output = self.model(
                    **left_context_tokens, output_hidden_states=True
                )

                # Get the hidden states for the last layer
                hidden_states = left_context_output.hidden_states[
                    -1
                ]  # Shape: (batch_size, seq_len, hidden_dim)

                # Aggregate hidden states by averaging over the sequence length
                left_context_embedding = torch.mean(
                    hidden_states, dim=1
                )  # Shape: (batch_size, hidden_dim)

                acc_sentence_embedding.append(left_context_embedding.squeeze(0))

        left_context_embedding = torch.stack(acc_sentence_embedding, dim=0)

        return left_context_embedding
