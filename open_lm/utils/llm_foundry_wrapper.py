# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

from typing import Union, Optional, Any
from llmfoundry.eval.metrics.nlp import (
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningGenerationExactMatchAccuracy,
)
from composer.metrics.nlp import (
    LanguageCrossEntropy,
    LanguagePerplexity,
)
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch import dist

from composer.models.huggingface import HuggingFaceModel


__all__ = ["ComposerOpenLMCausalLM", "SimpleComposerOpenLMCausalLM"]

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

TRAIN_METRICS = [
    LanguageCrossEntropy(),
    LanguagePerplexity(),
]
EVAL_METRICS = [
    LanguageCrossEntropy(),
    LanguagePerplexity(),
    InContextLearningLMAccuracy(),
    InContextLearningMultipleChoiceAccuracy(),
    InContextLearningGenerationExactMatchAccuracy(),
    InContextLearningLMExpectedCalibrationError(),
    InContextLearningMCExpectedCalibrationError(),
]


class SimpleComposerOpenLMCausalLM(HuggingFaceModel):
    def __init__(self, model, tokenizer):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            metrics=TRAIN_METRICS,
            eval_metrics=EVAL_METRICS,
            shift_labels=True,
        )

    def generate(self, input_ids=None, inputs_embeds=None, **kwargs):
        return super().generate(input_ids=input_ids, **kwargs)
    
    def eval_forward(self, batch, outputs: Optional[Any] = None):

        # If input_ids are all 0 after a certain point, we can skip that
        num_zeros = 0
        if batch['input_ids'] is not None:
            # Find indices of the last non-zero element in each row
            last_nonzero_indices = torch.max(torch.nonzero(batch['input_ids'], as_tuple=True)[1])
            if last_nonzero_indices < batch['input_ids'].shape[1] - 1:
                num_zeros = batch['input_ids'].shape[1] - last_nonzero_indices - 1
                batch['input_ids'] = batch['input_ids'][:, : last_nonzero_indices + 1]
                if batch['attention_mask'] is not None:
                    batch['attention_mask'] = batch['attention_mask'][:, : last_nonzero_indices + 1]
                if batch['labels'] is not None:
                    batch['labels'] = batch['labels'][:, : last_nonzero_indices + 1]

        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward
        if batch.get('mode', None) == 'generate':
            if self.tokenizer is None:
                raise ValueError(
                    'Generation eval cannot be used without providing a tokenizer to the model constructor.',
                )

            self.labels = batch.pop('labels')
            generation = self.generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                synced_gpus=dist.get_world_size() > 1,
                **batch.get('generation_kwargs', {}),
            )

            # don't remove prefix space to sentencepiece models
            if len(
                self.tokenizer(' a', add_special_tokens=False)['input_ids'],  # pyright: ignore[reportGeneralTypeIssues]
            ) == 1:
                return self.tokenizer.batch_decode(
                    generation[:, batch['input_ids'].shape[1]:],
                    skip_special_tokens=True,
                )
            else:
                return [
                    ' ' + generation for generation in
                    self.tokenizer.batch_decode(generation[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
                ]

        if self.use_logits or batch.get('mode', None) == 'icl_task':
            # pop labels first to avoid computing loss
            self.labels = batch.pop('labels')

            # HF encoder decoder models like T5 expect either decoder_input_ids or labels,
            # so we add decoder_input_ids to the batch if it is missing
            if self.config.is_encoder_decoder and 'decoder_input_ids' not in batch:
                if hasattr(self.model, 'prepare_decoder_input_ids_from_labels'):
                    batch['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(labels=self.labels)
                else:
                    raise RuntimeError(
                        'Encoder decoder models require that either decoder_input_ids is present in the batch'
                        ' or that the model has a prepare_decoder_input_ids_from_labels method.',
                    )

            if self.shift_labels or batch.get('mode', None) == 'icl_task':
                assert self.labels is not None
                # HF CausalLM models internally shift labels before computing loss, so we do the same here
                self.labels[:, :-1] = self.labels[:, 1:].clone()
                self.labels[:, -1] = -100

            output = outputs if outputs else self.forward(batch)

            if self.config.use_return_dict:
                output = output['logits']
            else:
                # if loss was computed (cached outputs from forward), loss is at index 0 and logits are at index 1
                # if loss was not computed (no cached outputs during eval), loss is not present and logits are at index 0
                output = output[1] if len(output[0].shape) == 0 else output[0]

            # if we are in the single class case, then remove the classes dimension
            if output.ndim == 2 and output.shape[1] == 1:
                output = output.squeeze(dim=1)
        else:
            output = outputs if outputs else self.forward(batch)

        # Add back the 0 that we removed
        if num_zeros: 
            if hasattr(output, 'logits') or isinstance(output, dict):
                fake_logits = torch.zeros(output['logits'].shape[0], num_zeros, output['logits'].shape[2], device=output['logits'].device)
                fake_logits[:, :, 0] = 1
                output['logits'] = torch.cat([output['logits'], fake_logits], dim=1)
            else:
                fake_logits = torch.zeros(output.shape[0], num_zeros, output.shape[2], device=output.device)
                fake_logits[:, :, 0] = 1
                output = torch.cat([output, fake_logits], dim=1)

        return output
