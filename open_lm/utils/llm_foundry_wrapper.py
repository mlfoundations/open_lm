# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

from typing import Mapping, Union

from composer.metrics.nlp import (
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningQAAccuracy,
    InContextLearningCodeEvalAccuracy,
    LanguageCrossEntropy,
    LanguagePerplexity,
)
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

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
    InContextLearningQAAccuracy(),
    InContextLearningLMExpectedCalibrationError(),
    InContextLearningMCExpectedCalibrationError(),
    InContextLearningCodeEvalAccuracy(),
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
