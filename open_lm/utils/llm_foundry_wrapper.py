# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

from typing import Mapping, Union

from composer.metrics.nlp import (InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError,
                                  InContextLearningMultipleChoiceAccuracy,
                                  InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity)
from omegaconf import DictConfig
from open_lm.utils.transformers.config import OpenLMConfig
from open_lm.utils.transformers.model import OpenLMforCausalLM
from transformers import (PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithZLoss
from llmfoundry.models.utils import init_empty_weights

__all__ = ['ComposerOpenLMCausalLM', 'SimpleComposerOpenLMCausalLM']

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
    InContextLearningMCExpectedCalibrationError()
]


class SimpleComposerOpenLMCausalLM(HuggingFaceModelWithZLoss):
    def __init__(self, model, tokenizer):
        super().__init__(
            model=model, tokenizer=tokenizer, metrics=TRAIN_METRICS, eval_metrics=EVAL_METRICS, z_loss=0.0
        )
