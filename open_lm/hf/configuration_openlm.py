# Follows OLMo's HF template

"""
OpenLM configuration
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

from open_lm.model import Params

logger = logging.get_logger(__name__)


class OpenLMConfig(PretrainedConfig):
    model_type = "openlm"

    def __init__(self, **kwargs):
        kwargs["architectures"] = ["OpenLMForCausalLM"]
        super().__init__(**kwargs)


# Register the config class so that it is available for transformer pipelines, auto-loading etc.
AutoConfig.register("openlm", OpenLMConfig)
