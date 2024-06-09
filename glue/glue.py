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
        "--n_shots", type=int, default=32, help="Number of examples to sample"
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
        default="cb",
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

    assert args.task_name != "multirc", "multirc is not supported by this script. Please use glue/multirc.py instead."
    assert args.task_name != "record", "record is not supported by this script. Please use glue/record.py instead."

    return args


def format_prompt(args, inputs: List[dict]):

    def general_detokenize(string):
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')
        string = re.sub(r" (['.,])", r"\1", string)
        return string

    # preprocess the inputs
    prefix = None
    if args.task_name == "boolq":
        for example in inputs:
            if not example["question"].endswith("?"):
                example["question"] += "?"
    elif args.task_name == "cb":
        for example in inputs:
            example["hypothesis"] = example["hypothesis"].rstrip(".")
    elif args.task_name == "copa":
        for example in inputs:
            example["premise"] = example["premise"].rstrip(".")
            example["answer"] = example["answer"][0].lower() + example["answer"][1:]
            example["connector"] = "because" if example["question"] == "cause" else "therefore"
    elif args.task_name == "wsc" or args.task_name == "wsc.fixed":
        for example in inputs:
            words = example["text"].split()
            words[example["span2_index"]] = f"*{words[example['span2_index']]}*"
            example["text"] = " ".join(words)
            example["text"] = general_detokenize(example["text"])

    # Format the prompt
    if args.task_name == "boolq":
        prompt_template = "{passage}<br>question: {question}<br>answer: {answer}"
    elif args.task_name == "cb":
        prompt_template = "{premise}<br>question: {hypothesis}; true, false, or neither?<br>answer: {answer}"
    elif args.task_name == "copa":
        prompt_template = "{premise} {connector} {answer}"
    elif args.task_name == "rte":
        prompt_template = "{premise}<br>question: {hypothesis} True or False?<br>answer: {answer}"
    elif args.task_name == "wic":
        prompt_template = "{sentence1}<br>{sentence2}<br>question: Is the word '{word}' used in the same way in the two sentences above?<br>answer: {answer}"
    elif args.task_name == "wsc" or args.task_name == "wsc.fixed":
        prefix = "Final Exam with Answer Key<br>Instructions: Please carefully read the following passages. For each passage, you must identify which noun the pronoun marked in *bold* refers to.<br>====="
        prompt_template = "Passage: {text}<br>Question: In the passage above, what does the pronoun '*{span2_text}*' refer to?<br>answer: {answer}"

    examples = [
        prompt_template.format(**kwargs)
        for kwargs in inputs
    ]
    prompt = "<br><br>".join(examples)  # Join the examples with two newlines
    if prefix is not None:
        prompt = prefix + "<br><br>" + prompt
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
    dataset = load_dataset("super_glue", args.task_name)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    def to_dict(sample):
        d = {
            key: value.strip() if isinstance(value, str) else value
            for key, value in sample.items()
        }
        if args.task_name == "boolq":
            d["options"] = ["no", "yes"]
            d["label"] = [d["label"]]
            d["answer"] = d["options"][d["label"][0]]
        elif args.task_name == "cb":
            d["options"] = ["true", "false", "neither"]
            d["label"] = [d["label"]]
            d["answer"] = d["options"][d["label"][0]]
        elif args.task_name == "copa":
            d["options"] = [d["choice1"], d["choice2"]]
            d["label"] = [d["label"]]
            d["answer"] = d["options"][d["label"][0]]
        elif args.task_name == "rte":
            d["options"] = ["True", "False"]
            d["label"] = [d["label"]]
            d["answer"] = d["options"][d["label"][0]]
        elif args.task_name == "wic":
            d["options"] = ["no", "yes"]
            d["label"] = [d["label"]]
            d["answer"] = d["options"][d["label"][0]]
        elif args.task_name == "wsc" or args.task_name == "wsc.fixed":
            import spacy
            nlp = spacy.load("en_core_web_sm")

            doc = nlp(d["text"])
            noun_phrases = [np.text for np in doc.noun_chunks] + [d["span1_text"]]
            noun_phrases = [np for np in set(noun_phrases) if np == d["span1_text"] or (d["span1_text"] not in np and d["span2_text"] not in np)]
            d["options"] = noun_phrases
            if d["label"] == 0:
                d["label"] = [i for i, np in enumerate(noun_phrases) if np != d["span1_text"]]
                d["gold"] = False
            else:
                d["label"] = [i for i, np in enumerate(noun_phrases) if np == d["span1_text"]]
                d["gold"] = True
                d["answer"] = d["span1_text"]

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
        if args.task_name == "boolq":
            score_length = len(inputs[-1]["answer"].split()) + len(inputs[-1]["question"].split())
        elif args.task_name == "cb":
            score_length = len(inputs[-1]["answer"].split()) + len(inputs[-1]["premise"].split()) + len(inputs[-1]["hypothesis"].split()) + 4
        elif args.task_name == "copa":
            score_length = len(inputs[-1]["answer"].split()) + len(inputs[-1]["premise"].split()) + 1
        elif args.task_name == "rte":
            score_length = len(inputs[-1]["answer"].split()) + len(inputs[-1]["premise"].split()) + len(inputs[-1]["hypothesis"].split()) + 3
        elif args.task_name == "wic":
            score_length = len(inputs[-1]["answer"].split()) + len(inputs[-1]["sentence1"].split()) + len(inputs[-1]["sentence2"].split()) + len(inputs[-1]["word"].split()) + 12
        elif args.task_name == "wsc" or args.task_name == "wsc.fixed":
            score_length = len(inputs[-1]["answer"].split()) + 7
        else:
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


def sample_random_examples(dataset: List[dict], n_shots: int, task_name: str):
    # Sample n_shots different examples from the dataset (excluding the example at example_index)
    if task_name == "wsc" or task_name == "wsc.fixed":
        sequence = [i for i in range(len(dataset)) if dataset[i]["gold"]]
    else:
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

    accuracies, f1_scores = [], []
    for _ in range(args.n_repetitions):
        gold_labels, predictions = [], []
        for i, sample in enumerate(tqdm(valid_dataset)):
            shots = sample_random_examples(train_dataset, args.n_shots, args.task_name)
            examples = shots + [sample]

            predicted_answer = predict(args, model, examples, verbose=i < 20)
            predictions.append(predicted_answer)
            gold_labels.append(predicted_answer if predicted_answer in sample["label"] else sample["label"][0])

            if i % 10 == 0:
                accuracy = metrics.accuracy_score(gold_labels, predictions)
                print(f"Accuracy: {accuracy:.2%}")
                print(f"F1: {metrics.f1_score(gold_labels, predictions, average='macro'):.2%}", flush=True)

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