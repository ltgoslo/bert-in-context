import argparse
import random
import os
import torch
import copy
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

import nltk
nltk.download('punkt')  # Download the required model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ltg/deberta-xxlarge-fixed",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\\n ",
        help="Newline separator",
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=256,
        help="Prompt length",
    )
    args = parser.parse_args()

    return args


def format_prompt(args, inputs: dict):
    if len(inputs["lines_1"]) > 0:
        inputs["lines_1"] += args.separator
    if len(inputs["lines_2"]) > 0:
        inputs["lines_2"] = args.separator + inputs["lines_2"]
    prompt_template = """> Some special magic number is hidden within the following articles. Make sure to memorize it. I will quiz you about the magic number afterwards.

{lines_1}{needle}{lines_2}

> Question: What is the special magic number mentioned in the provided text?
> Answer: The special magic number mentioned in the provided text is"""

    prompt = prompt_template.format(**inputs)
    if "opt" in args.model_name_or_path:
        prompt = prompt.replace("\n", args.separator)

    return prompt


def load_model(model_path: str, args):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval().cuda()

    if args.prompt_length > 8192:
        model = model.bfloat16()

    return {
        "tokenizer": tokenizer,
        "model": model,
        "logits_processor": LogitsProcessorList([DigitsOnlyProcessor(tokenizer)])
    }


def load_data(args, tokenizer):
    segmenter = nltk.data.load('tokenizers/punkt/english.pickle')

    filename = "paul_graham.jsonl"
    documents = [json.loads(line) for line in open(filename, "r")]

    for document in documents:
        document["sentences"] = []
        for p in document["paragraphs"]:
            sentences = segmenter.tokenize(p)
            document["sentences"] += sentences

        if "opt" in args.model_name_or_path:
            line_lengths = [len(tokenizer(s)["input_ids"]) - 1 for s in document["sentences"]]
        else:
            line_lengths = [len(tokenizer(s)["input_ids"]) - 2 for s in document["sentences"]]

        document["line_lengths"] = line_lengths
    
    return documents


class DigitsOnlyProcessor:
    def __init__(self, tokenizer):
        self.non_digit_tokens = []
        for subword_id in range(tokenizer.vocab_size):
            subword = tokenizer.decode([subword_id])
            if not subword.isdigit():
                self.non_digit_tokens.append(subword_id)
        self.non_digit_tokens = torch.tensor(self.non_digit_tokens).cuda()

    def __call__(self, input_ids, scores):
        scores[..., self.non_digit_tokens] = -float('inf')
        return scores

@torch.no_grad()
def generate(text: str, model: dict, args):
    if "opt" in args.model_name_or_path:
        model["tokenizer"].truncation_side='left'
        input_ids = model["tokenizer"](text, return_tensors="pt", truncation=True, max_length=2048-6).input_ids.cuda()
    else:
        input_ids = model["tokenizer"](text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

    output = model["model"].generate(
        input_ids,
        max_new_tokens=6,
        logits_processor=model["logits_processor"],
        do_sample=False,
        num_beams=1,
        use_cache=None
    )[0, input_ids.size(1):]

    prediction = model["tokenizer"].decode(output)
    prediction = prediction.replace(args.separator, ' ')
    prediction = prediction.strip()

    return prediction


def main():
    args = parse_args()
    random.seed(42)

    model = load_model(args.model_name_or_path, args)
    documents = load_data(args, model["tokenizer"])

    log_file = open(
        f"result_haystack_{args.prompt_length}_{args.model_name_or_path.split('/')[-1]}",
        "w",
    )
    
    for i in tqdm(range(len(documents))):

        # random 6-digit number
        magical_number = random.randint(100000, 999999)
        needle = f"The magic number is {magical_number}."

        empty_prompt = format_prompt(args, {
            "lines_1": "",
            "needle": needle,
            "lines_2": "",
            "separator": args.separator
        })
        empty_prompt_len = len(model["tokenizer"](empty_prompt)["input_ids"])

        lines = []
        document_offset, line_offset = i, 0
        current_length = empty_prompt_len

        while current_length < args.prompt_length:
            
            if line_offset >= len(documents[document_offset]["sentences"]):
                document_offset = (document_offset + 1) % len(documents)
                line_offset = 0
            
            line = documents[document_offset]["sentences"][line_offset]
            line_length = documents[document_offset]["line_lengths"][line_offset]

            line_offset += 1
            lines.append(line)

            if "opt" in args.model_name_or_path:
                current_length += line_length + 2
            else:
                current_length += line_length + 2

        lines = lines[:-1]

        for placement in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                
            # random 6-digit number
            magical_number = random.randint(100000, 999999)
            needle = f"The magic number is {magical_number}."

            prompt = format_prompt(args, {
                "lines_1": args.separator.join(lines[:int(len(lines) * placement)]),
                "needle": needle,
                "lines_2": args.separator.join(lines[int(len(lines) * placement):]),
                "separator": args.separator
            })

            if i == 0:
                print(prompt, flush=True)

            prediction = generate(prompt, model, args)
            prediction = prediction.strip()

            predicted_number = ''
            for ch in prediction:
                if ch.isdigit():
                    predicted_number += ch
            predicted_number = predicted_number[:6]
            gold_number = str(magical_number)

            # compute exact match
            score = 1 if predicted_number == gold_number else 0

            print(f"{len(model['tokenizer'](prompt)['input_ids'])},{i},{placement},{prediction},{score}", flush=True)

            log_file.write(f"{args.prompt_length},{i},{placement},{score}\n")
            log_file.flush()

    log_file.close()


if __name__ == "__main__":
    main()
