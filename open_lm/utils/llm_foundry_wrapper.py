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

        output = super().eval_forward(batch, outputs)

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
