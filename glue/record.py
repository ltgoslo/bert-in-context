import argparse
import random
import string
from collections import Counter
import torch
import torch.nn.functional as F
from typing import List
from datasets import load_dataset
import copy
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
import re
from statistics import mean, stdev


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ltg/deberta-xxlarge-fixed",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--n_shots", type=int, default=0, help="Number of examples to sample"
    )
    parser.add_argument(
        "--n_repetitions",
        type=int,
        default=1,
        help="Number of repetitions to average over",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="record",
        help="Name of the SuperGLUE task to evaluate on",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\\n "
    )
    args = parser.parse_args()

    # If n_shots is 0, we don't need to account for random sampling of examples
    if args.n_shots == 0:
        args.n_repetitions = 1

    assert args.task_name == "record"

    return args


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def format_prompt(args, inputs: List[dict]):
    for example in inputs:
        passage, *highlights = example["passage"].split("@highlight")
        example["passage"] = passage + '\n' + '@highlight'.join(highlights)
        example["passage"] = example["passage"].replace("@highlight\n", "- ")

    prompt_template = "{passage}<br>- {answer}"

    examples = [
        prompt_template.format(**kwargs)
        for kwargs in inputs
    ]
    prompt = "<br><br>".join(examples)  # Join the examples with two newlines
    prompt = prompt.replace("\n", "<br>")  # Replace newline characters with <br> for formatting
    prompt = prompt.replace("<br>", args.separator)  # Replace <br> with the separator

    return prompt


def load_model(model_path: str, args):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).eval().cuda()

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def load_data(args):
    dataset = load_dataset("super_glue", args.task_name)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    def to_dict(sample):
        d = {
            key: value.strip() if isinstance(value, str) else value
            for key, value in sample.items()
        }
        d["options"] = [
            d["query"].replace("@placeholder", entity)
            for entity in d["entities"]
        ]
        d["label"] = [
            d["entities"].index(answer)
            for answer in d["answers"]
        ]
        d["answer"] = d["options"][d["label"][0]]
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


def predict(args, model, samples):
    logps = []
    for option in samples[-1]["options"]:
        inputs = copy.deepcopy(samples)
        inputs[-1]["answer"] = option
        input_text = format_prompt(args, inputs)
        score_length = len(inputs[-1]["answer"].split())

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


def sample_random_examples(dataset: List[dict], n_shots: int, task_name: str):
    sequence = list(range(len(dataset)))
    random_indices = random.sample(sequence, n_shots)
    return [dataset[j] for j in random_indices]


def main():
    args = parse_args()
    random.seed(42)

    model = load_model(args.model_name_or_path, args)
    train_dataset, valid_dataset = load_data(args)

    log_file = open(
        f"record/result_causal_{args.model_name_or_path.split('/')[-1]}_{args.n_shots}-shots.txt",
        "w",
    )

    accuracies, f1_scores = [], []
    for _ in range(args.n_repetitions):
        gold_labels, predictions = [], []
        f1_sum, exact_match_sum, total = 0, 0, 0
        for i, sample in enumerate(tqdm(valid_dataset)):
            shots = sample_random_examples(train_dataset, args.n_shots, args.task_name)
            examples = shots + [sample]

            predicted_answer = predict(args, model, examples)
            predictions.append(predicted_answer)
            gold_labels.append(predicted_answer if predicted_answer in sample["label"] else sample["label"][0])

            predicted_answer_str = examples[-1]["entities"][predicted_answer]
            gold_answer_strs = [examples[-1]["entities"][label] for label in sample["label"]]

            sample_f1_score = metric_max_over_ground_truths(
                f1_score, predicted_answer_str, gold_answer_strs
            )
            sample_exact_match = metric_max_over_ground_truths(
                exact_match_score, predicted_answer_str, gold_answer_strs
            )

            f1_sum += sample_f1_score
            exact_match_sum += sample_exact_match
            total += 1

        # Calculate metrics
        accuracy = exact_match_sum / total
        f1 = f1_sum / total

        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")
        log_file.write(f"{exact_match_sum}, {f1_sum}, {total}, {accuracy}, {f1}\n")
        log_file.flush()

    log_file.write(f"\nMean accuracy: {mean(accuracies)} ± {stdev(accuracies) if len(accuracies) > 1 else 0}")
    log_file.write(f"\nMean F1: {mean(f1_scores)} ± {stdev(f1_scores) if len(f1_scores) > 1 else 0}")
    log_file.close()


if __name__ == "__main__":
    main()
