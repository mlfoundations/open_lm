from utils.transformers.hf_model import OpenLMforCausalLM
from transformers import GPTNeoXTokenizerFast
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--prompt", type=str, default="I enjoy walking with my cute dog")
    args = parser.parse_args()
    model = OpenLMforCausalLM.from_pretrained(args.checkpoint)
    model = model.cuda()
    # hardcoded to neox tokenizer
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    greedy_output = model.generate(input_ids.to(0), max_length=500, do_sample=True, top_p=0.9)
    print("Output:\n" + 100 * "-")
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
