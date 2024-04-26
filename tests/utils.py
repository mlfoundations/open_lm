import os
from tqdm import tqdm
import urllib.request
import hashlib
import warnings
import tarfile
import json

from pathlib import Path
from huggingface_hub import snapshot_download
import torch

from open_lm.utils import make_wds_manifest as mwm
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
import wikipedia
from composer.utils import dist, get_device
from open_lm.utils.llm_foundry_wrapper import SimpleComposerOpenLMCausalLM


def download_val_data(name: str, root: str = None):
    # modified from oai _download clip function

    if root is None:
        raise RuntimeError(f"{root} must not be None")

    cloud_checkpoints = {
        "shard_00000000.tar": {
            "url": "https://huggingface.co/datasets/mlfoundations/open_lm_example_data/resolve/main/example_train_data/shard_00000000.tar",
            "sha256": "f53d2cbaf5ffc0532aaefe95299e1ef5e1641f0a1cbf7ae12642f71eaa892d30",
        },
    }

    if name not in cloud_checkpoints:
        raise ValueError(
            f"unsupported cloud checkpoint: {name}. currently we only support: {list(cloud_checkpoints.keys())}"
        )

    os.makedirs(root, exist_ok=True)

    expected_sha256 = cloud_checkpoints[name]["sha256"]
    download_target = os.path.join(root, f"{name}")
    url = cloud_checkpoints[name]["url"]

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def download_dl_test_data(root: str = "./tests/assets"):
    """Downloads test files if the data doesn't exist in HF cache."""

    snapshot_args = dict(
        repo_id="mlfoundations/open_lm_test_data_v2",
        local_dir=root,
        repo_type="dataset",
    )

    snapshot_download(**snapshot_args)


def make_tar(tar_num, num_lines, source_num=0, dir_name=None):
    fname = lambda i: "%08d_chunk_%s.json" % (tar_num, i)

    if dir_name != None:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

    tarname = os.path.join(dir_name, "%08d.tar" % tar_num)
    if os.path.exists(tarname):
        return

    fnames = []
    with tarfile.open(tarname, "w") as tar:
        for line in range(num_lines):
            base_line = [666 for _ in range(2049)]
            base_line[0] = source_num
            base_line[1] = tar_num
            base_line[2] = line
            this_file = fname(line)
            with open(this_file, "w") as f:
                f.write(json.dumps(base_line))
            tar.add(this_file)
            fnames.append(this_file)

    for f in fnames:
        try:
            os.unlink(f)
        except:
            pass


def make_source(source_num, size_per_tar, total_size):
    num_tars = total_size // size_per_tar
    if total_size % size_per_tar != 0:
        num_tars += 1

    base_dir = "tests/assets"
    os.makedirs(base_dir, exist_ok=True)

    num_remaining = total_size
    for tar_num in range(num_tars):
        this_tar = min(num_remaining, size_per_tar)
        make_tar(tar_num, this_tar, source_num=source_num, dir_name="tests/assets/source_id_%02d" % source_num)
        num_remaining -= this_tar

    args = ["--data-dir", "tests/assets/source_id_%02d" % source_num]
    mwm.main(args)


def make_fake_tarfiles():
    """Makes sources for dataloader tests.
    Running main will...
    - generate 2 sources, titled 'source_id_00', 'source_id_01'
    - each source has 7 .tar files, each with 100 sequences (except the last which has 66)
    - each sequence has the first three tokens as (source_num, tar_num, line_num)

    This way we'll be able to identify where each sequence came from when we test...
    """
    for i in range(2):
        make_source(i, 100, 666)


def _get_tokens_inputs(tokenizer, args, wiki_page=None, start_index=None):
    if args.input_text == "random":
        wikipedia.set_lang("en")
        try:
            wiki_page = wikipedia.page(wiki_page)
            content = wiki_page.content
            content_tokenized = tokenizer(content)
            content_len = len(content_tokenized["input_ids"])
            if content_len <= args.context_len + start_index:
                print(f"Page too short, will load a different one than the one requested ({wiki_page}).")
                wiki_page = None  # If the page is too short, try again
        except:  # noqa
            wiki_page = None
        while wiki_page is None:
            rand_page_title = wikipedia.random(pages=1)
            try:
                wiki_page = wikipedia.page(rand_page_title)
            except:  # noqa
                continue
            content = wiki_page.content
            content_tokenized = tokenizer(content)
            content_len = len(content_tokenized["input_ids"])
            if content_len <= args.context_len:
                wiki_page = None  # If the page is too short, try again
        context_len = args.context_len
        if start_index is None:
            start_index = int((content_len - context_len) * torch.rand(1))
        content_tokenized["input_ids"] = content_tokenized["input_ids"][start_index : start_index + context_len]
        input = content_tokenized
    else:
        input = tokenizer(args.input_text)
    if torch.cuda.is_available():
        input = {k: torch.tensor(v).unsqueeze(0).cuda() for k, v in input.items()}
    else:
        input = {k: torch.tensor(v).unsqueeze(0) for k, v in input.items()}
    return input


@torch.inference_mode()
def run_model(open_lm, tokenizer, args, wiki_page=None, start_index=None):
    dist.initialize_dist(get_device(None), timeout=600)
    input = _get_tokens_inputs(tokenizer, args, wiki_page=wiki_page, start_index=start_index)
    composer_model = SimpleComposerOpenLMCausalLM(open_lm, tokenizer)
    if torch.cuda.is_available():
        composer_model = composer_model.cuda()

    generate_args = {
        "do_sample": args.temperature > 0,
        "pad_token_id": 50282,
        "max_new_tokens": args.max_gen_len,
        "use_cache": args.use_cache,
    }

    if args.num_beams > 1:
        generate_args["num_beams"] = args.num_beams
    # If these are set when temperature is 0, they will trigger a warning and be ignored
    if args.temperature > 0:
        generate_args["temperature"] = args.temperature
        generate_args["top_p"] = args.top_p

    output = composer_model.generate(
        input_ids=input["input_ids"],
        **generate_args,
    )
    output = tokenizer.decode(output[0].cpu().numpy())
    torch.distributed.destroy_process_group()
    return output


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], **kwargs):
        """Character tokenizer for Hugging Face transformers.
        Copied from https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py

        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
        """
        self.characters = characters
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
        }

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)
