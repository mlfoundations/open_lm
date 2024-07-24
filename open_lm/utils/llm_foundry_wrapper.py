# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

from typing import Mapping, Union, List
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
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

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

class CustomStopTokensCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: List[str]) -> None:
        self.stop_tokens = stop_tokens

    def __call__(self, generated_tokens: torch.Tensor, *args, **kwargs) -> bool:
        return any(token in self.stop_tokens for token in generated_tokens.flatten())


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
        stop_token = self.tokenizer.eos_token_id
        stop_criteria = CustomStopTokensCriteria([stop_token])
        stop_criteria_list = StoppingCriteriaList([stop_criteria])
        if "stopping_criteria" in kwargs:
            stop_criteria_list += kwargs.pop("stopping_criteria")
        return super().generate(input_ids=input_ids, stopping_criteria=stop_criteria_list, **kwargs)
