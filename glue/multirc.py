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
        "--task_name",
        type=str,
        default="multirc",
        help="Name of the SuperGLUE task to evaluate on",
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
    prefix = "READING COMPREHENSION ANSWER KEY"
    question_template = "{paragraph}<br><br>{question}"
    answer_template = "<br>- {answer}"

    prompt = prefix + "<br><br><br>"
    for i, sample in enumerate(inputs):
        prompt += question_template.format(**sample)
        for answer in sample["answers"]:
            prompt += answer_template.format(answer=answer)
        if i < len(inputs) - 1:
            prompt += "<br><br><br>"

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
    assert args.task_name == "multirc"
    dataset = load_dataset("super_glue", args.task_name)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    def to_dict(sample):
        d = {
            key: value.strip() if isinstance(value, str) else value
            for key, value in sample.items()
        }
        d["options"] = [f"[False] {d['answer']}", f"[True] {d['answer']}"]
        d["label"] = [d["label"]]
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

    def group_by_question(dataset):
        grouped = {}
        for sample in dataset:
            key = sample["idx"]["question"]
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(sample)

        groups = []
        for group in grouped.values():
            groups.append({
                "answers": [sample["answer"] for sample in group],
                "options": [sample["options"] for sample in group],
                "question": group[0]["question"],
                "paragraph": group[0]["paragraph"],
                "labels": [sample["label"] for sample in group]
            })

        return groups
    
    train_dataset = group_by_question(train_dataset)
    val_dataset = group_by_question(val_dataset)

    return train_dataset, val_dataset


def predict(args, model, samples, prev_answers):
    logps = []
    for option in samples[-1]["options"]:
        inputs = copy.deepcopy(samples)
        inputs[-1]["answers"] = prev_answers + [option]
        input_text = format_prompt(args, inputs)
        score_length = len(inputs[-1]["answers"][-1].split()) + len(inputs[-1]["question"].split()) + 1

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

    model = load_model(args.model_name_or_path)
    train_dataset, valid_dataset = load_data(args)

    log_file = open(
        f"result_{args.task_name}_{args.model_name_or_path.split('/')[-1]}_{args.n_shots}-shots.txt",
        "w",
    )

    accuracies = []
    f1_scores = []
    for _ in range(args.n_repetitions):
        gold_labels, predictions = [], []
        n_correct, n_total = 0, 0
        for i, sample in enumerate(tqdm(valid_dataset)):

            sample = copy.deepcopy(sample)
            options = sample["options"]
            answers = sample["answers"]
            all_answers_correct = True
            prev_answers = []

            for j in range(len(answers)):
                sample["answers"] = answers[j]
                sample["options"] = options[j]

                shots = sample_random_examples(train_dataset, args.n_shots, args.task_name)
                examples = shots + [sample]

                predicted_answer = predict(args, model, examples, prev_answers)
                predictions.append(predicted_answer)
                gold_labels.append(predicted_answer if predicted_answer in sample["labels"][j] else sample["labels"][j][0])

                if predicted_answer not in sample["labels"][j]:
                    all_answers_correct = False
            
            if all_answers_correct:
                n_correct += 1
            n_total += 1

            if (i + 1) % 10 == 0:
                f1_score = metrics.f1_score(gold_labels, predictions)
                print(f"EM Accuracy: {n_correct / n_total:.2%}")
                print(f"F1 Score: {f1_score:.2%}")

        # Calculate metrics
        accuracy = n_correct / n_total
        f1_score = metrics.f1_score(gold_labels, predictions)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1_score}")
        log_file.write(f"{accuracy},{f1_score}\n")
        log_file.flush()

    log_file.write(f"\nMean accuracy: {mean(accuracies)} ± {stdev(accuracies) if len(accuracies) > 1 else 0}")
    log_file.write(f"\nMean F1 score: {mean(f1_scores)} ± {stdev(f1_scores) if len(f1_scores) > 1 else 0}\n")
    log_file.close()


if __name__ == "__main__":
    main()
