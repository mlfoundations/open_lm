# Follows OLMo's HF template

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from open_lm.hf.configuration_openlm import OpenLMConfig


class OpenLMTokenizerFast(PreTrainedTokenizerFast):
    # Note: OpenLM's tokenizer is already a wrapper around huggingface. This is potentially unnecessary.
    pass

    # def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
    #     # This is required to make the implementation complete.
    #     pass


# Register the tokenizer class so that it is available for transformer pipelines, auto-loading etc.
AutoTokenizer.register(OpenLMConfig, fast_tokenizer_class=OpenLMTokenizerFast)
