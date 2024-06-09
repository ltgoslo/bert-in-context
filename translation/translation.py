import argparse
import random
from typing import List
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="ltg/deberta-xxlarge-fixed")  # Path to the pre-trained model
    parser.add_argument("--source_language", type=str, default="de")  # Source language code
    parser.add_argument("--target_language", type=str, default="en")  # Target language code
    parser.add_argument("--n_shots", type=int, default=1)  # Number of random examples to sample
    parser.add_argument("--n_repetitions", type=int, default=1)  # Number of repetitions
    parser.add_argument("--separator", type=str, default="\\n ")  # Separator between source and target texts

    args = parser.parse_args()

    # If n_shots is 0, we don't need to account for random sampling of examples
    if args.n_shots == 0:
        args.n_repetitions = 1

    return args


def format_prompt(
    source_texts: List[str],
    target_texts: List[List[str]],
    source_language: str,
    target_language: str,
    args
):
    # Format the prompt with source and target texts
    if target_language == "en":
        source_language = {"ro": "Romanian", "de": "German", "fr": "French"}[source_language]
        target_language = "English"
    elif target_language == "de":
        source_language = "Englisch"
        target_language = "Deutsch"
    elif target_language == "ro":
        source_language = "Engleză"
        target_language = "Română"
    elif target_language == "fr":
        source_language = "Anglais"
        target_language = "Français"

    # As in "The unreasonable effectiveness of few-shot learning for machine translation"
    # (https://arxiv.org/abs/2302.01398)
    prompt_template = (
        "{source_language}: {source_text}{separator}{target_language}:{target_text}"
    )

    examples = [
        prompt_template.format(
            source_language=source_language,
            target_language=target_language,
            source_text=source_text,
            target_text=" " + target_text[0] if i < len(source_texts) - 1 else "",  # Add space before target text (except for the last example, which should be empty)
            separator=args.separator
        )
        for i, (source_text, target_text) in enumerate(zip(source_texts, target_texts))
    ]
    prompt = (args.separator + args.separator).join(
        examples
    )  # Join the examples with two newlines (https://arxiv.org/abs/2302.01398)

    return prompt


def load_model(model_path: str):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval().cuda()

    return {
        "tokenizer": tokenizer,
        "model": model,
        "eos_token": "\\n"
    }


def load_data(source_language: str, target_language: str, reverse: bool = False):
    # Load the dataset for the given source and target languages

    if source_language == "de" or target_language == "de":
        dataset = load_dataset("wmt16", f"de-en", cache_dir='.')
    elif source_language == "ro" or target_language == "ro":
        dataset = load_dataset("wmt16", f"ro-en", cache_dir='.')
    elif source_language == "fr" or target_language == "fr":
        dataset = load_dataset("wmt/wmt14", f"fr-en", cache_dir='.')

    train_dataset = [
        {"source_text": sample["translation"][source_language], "target_texts": [sample["translation"][target_language]]}
        for sample in dataset["validation"]
    ]
    test_dataset = [
        {"source_text": sample["translation"][source_language], "target_texts": [sample["translation"][target_language]]}
        for sample in dataset["test"]
    ]

    return train_dataset, test_dataset


def sample_random_examples(dataset: List[dict], example_index: int, n_shots: int):
    sequence = list(range(len(dataset)))
    random_indices = random.sample(sequence, n_shots)
    return [dataset[j] for j in random_indices]


@torch.no_grad()
def generate(text: str, model: dict, args):
    input_ids = model["tokenizer"](text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

    prediction = model["model"].generate(
        input_ids,
        num_beams=4,
        do_sample=False,
        use_cache=None,
        max_new_tokens=128,
        min_new_tokens=8,
        temperature=1.0,
        eos_token_id=model["tokenizer"](".\\", add_special_tokens=False).input_ids[1:]
    )
    prediction = prediction[0, input_ids.size(1):]
    prediction = model["tokenizer"].decode(prediction)
    prediction = prediction.replace("\\n", "\n")
    prediction = prediction.strip().strip('\\')
    prediction = prediction.strip().split('\n')[0].strip().strip('\\').strip()

    return prediction


def main():
    args = parse_args()
    random.seed(42)

    model = load_model(args.model_name_or_path)
    train_dataset, dev_dataset = load_data(args.source_language, args.target_language)

    for repetition in range(args.n_repetitions):

        pred_file = open(
            f"pred_{args.source_language}_{args.target_language}_{args.model_name_or_path.split('/')[-1]}_{args.n_shots}-shots_{repetition}.txt",
            "w",
        )
        gold_file = open(
            f"gold_{args.source_language}_{args.target_language}_{args.model_name_or_path.split('/')[-1]}_{args.n_shots}-shots_{repetition}.txt",
            "w",
        )

        for i, example in enumerate(tqdm(dev_dataset)):
            shots = sample_random_examples(train_dataset, i, args.n_shots)
            source_texts = [example["source_text"] for example in shots] + [
                example["source_text"]
            ]
            target_texts = [example["target_texts"] for example in shots] + [
                example["target_texts"]
            ]

            prompt = format_prompt(
                source_texts, target_texts, args.source_language, args.target_language, args
            )
            prediction = generate(prompt, model, args)

            pred_file.write(f"{prediction}\n")
            gold_file.write(f"{example['target_texts'][0]}\n")
            pred_file.flush()
            gold_file.flush()

            if i < 20:
                print(f"Prompt:\n{prompt}\n")
                print(f"Target:\n{example['target_texts'][0]}\n")
                print(f"Prediction:\n{prediction}\n")
                print(len(model["tokenizer"](prompt).input_ids))
                print(flush=True)

    pred_file.close()
    gold_file.close()


if __name__ == "__main__":
    main()