import argparse
import random
import torch
import torch.nn.functional as F
from statistics import mean, stdev
from typing import List
from datasets import load_dataset
import copy
from sklearn import metrics
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ltg/deberta-xxlarge-fixed",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--n_shots", type=int, default=1, help="Number of examples to sample"
    )
    parser.add_argument(
        "--n_repetitions",
        type=int,
        default=5,
        help="Number of repetitions to average over",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\\n "
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for scoring"
    )
    args = parser.parse_args()
    # If n_shots is 0, we don't need to account for random sampling of examples
    if args.n_shots == 0:
        args.n_repetitions = 1

    return args


def format_prompt(args, inputs: List[dict]):

    prompt_template = "{answer}"

    examples = [
        prompt_template.format(**kwargs)
        for kwargs in inputs
    ]
    prompt = "<br>".join(examples)  # Join the examples with two newlines
    prompt = prompt.replace("\n", "<br>")  # Replace newline characters with <br> for formatting
    prompt = prompt.replace("<br>", args.separator)  # Replace <br> with the separator

    return prompt


def load_model(model_path: str):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).eval().cuda()

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def load_data(args):
    dataset = load_dataset("winograd_wsc", "wsc273")

    train_dataset = dataset["test"]
    val_dataset = dataset["test"]

    def to_dict(sample):
        d = {
            key: value.strip() if isinstance(value, str) else value
            for key, value in sample.items()
        }
        text = d["text"][:d["pronoun_loc"]] + "{}" + d["text"][d["pronoun_loc"] + len(d["pronoun"]):]
        options = [o[0].lower() + o[1:] if o.startswith("The") else o for o in d["options"]]
        d["options"] = [text.format(option.strip()) for option in options]
        d["label"] = int(d["label"])
        d["answer"] = d["options"][d["label"]]

        return d

    train_dataset = [
        to_dict(sample)
        for sample in train_dataset
    ]
    val_dataset = [
        to_dict(sample)
        for sample in val_dataset
    ]

    return train_dataset, val_dataset


def predict(args, model, samples, verbose=False):
    logps = []
    for option in samples[-1]["options"]:
        inputs = copy.deepcopy(samples)
        inputs[-1]["answer"] = option
        input_text = format_prompt(args, inputs)
        score_length = len(inputs[-1]["answer"].split())

        if verbose:
            print(input_text)

        logp = model["model"].score(
            input_text,
            score_length,
            model["tokenizer"],
            "cuda",
            8
        )
        logps.append(logp)

    prediction = logps.index(max(logps))
    return prediction


def sample_random_examples(dataset: List[dict], n_shots: int, example_index: int):
    # Sample n_shots different examples from the dataset (excluding the example at example_index)
    sequence = [i for i in range(len(dataset)) if i != example_index and i + 1 != example_index and i - 1 != example_index]
    random_indices = random.sample(sequence, n_shots)
    return [dataset[j] for j in random_indices]


def main():
    args = parse_args()
    random.seed(42)

    model = load_model(args.model_name_or_path)
    train_dataset, valid_dataset = load_data(args)

    log_file = open(
        f"result_winograd_{args.model_name_or_path.split('/')[-1]}_{args.n_shots}-shots.txt",
        "w",
    )

    accuracies, f1_scores = [], []
    for _ in range(args.n_repetitions):
        gold_labels, predictions = [], []
        for i, sample in enumerate(tqdm(valid_dataset)):
            shots = sample_random_examples(train_dataset, args.n_shots, i)
            examples = shots + [sample]

            predicted_answer = predict(args, model, examples, verbose=i < 10)
            predictions.append(predicted_answer)
            gold_labels.append(sample["label"])

            if (i + 1) % 100 == 0:
                accuracy = metrics.accuracy_score(gold_labels, predictions)
                print(f"Accuracy: {accuracy:.2%}")
                print(f"F1: {metrics.f1_score(gold_labels, predictions, average='macro'):.2%}")

        # Calculate metrics
        accuracy = metrics.accuracy_score(gold_labels, predictions)
        f1 = metrics.f1_score(gold_labels, predictions, average="macro")
        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")
        log_file.write(f"{accuracy}, {f1}\n")
        log_file.flush()

    log_file.write(f"\nMean accuracy: {mean(accuracies)} ± {stdev(accuracies) if len(accuracies) > 1 else 0}")
    log_file.write(f"\nMean F1: {mean(f1_scores)} ± {stdev(f1_scores) if len(f1_scores) > 1 else 0}")
    log_file.close()


if __name__ == "__main__":
    main()