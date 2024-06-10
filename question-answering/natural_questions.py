import argparse
import random
from statistics import mean, stdev
from typing import List
import torch
import re
import string
import unicodedata
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="ltg/deberta-xxlarge-fixed"
    )  # Path to the pre-trained model
    parser.add_argument(
        "--n_shots", type=int, default=1
    )  # Number of random examples to sample
    parser.add_argument("--n_repetitions", type=int, default=5)  # Number of repetitions
    args = parser.parse_args()
    # If n_shots is 0, we don't need to account for random sampling of examples
    if args.n_shots == 0:
        args.n_repetitions = 1

    return args


def format_prompt(
    questions: List[str],
    answers: List[str]
):

    for i, q in enumerate(questions):
        q = q.strip()
        if q[-1] not in string.punctuation:
            q = q + "?"
        if q[0].islower():
            q = q[0].upper() + q[1:]
        questions[i] = q

    prompt_template = (
        "Q: {question}\\n A:{answer}"
    )

    examples = [
        prompt_template.format(
            question=q,
            answer=" " + a if i < len(questions) - 1 else "",  # Add space before target text (except for the last example,
            # which should be empty)
        )
        for i, (q, a) in enumerate(zip(questions, answers))
    ]
    prompt = "\\n \\n ".join(
        examples
    )

    return prompt


def load_model(model_path: str):
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval().cuda()

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def load_data():
    # Load the dataset for the given source and target languages

    dataset = load_dataset("nq_open", cache_dir='.')

    train_dataset = [
        {
            "question": sample["question"].strip(),
            "answers": [
                answer.strip() for answer in sample["answer"]
            ],
            "preferred_answer": sample["answer"][0].strip()
        }
        for sample in dataset["train"]
    ]

    validation_dataset = [
        {
            "question": sample["question"].strip(),
            "answers": [
                answer.strip() for answer in sample["answer"]
            ],
            "preferred_answer": sample["answer"][0].strip()
        }
        for sample in dataset["validation"]
    ]

    return train_dataset, validation_dataset


def sample_random_examples(dataset: List[dict], example_index: int, n_shots: int):
    # Sample n_shots different examples from the dataset (excluding the example at example_index)
    sequence = list(range(len(dataset)))
    random_indices = random.sample(sequence, n_shots)
    return [dataset[j] for j in random_indices]


def normalize_answer(s):
    """Normalize answer."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


@torch.no_grad()
def generate(text: str, model: dict):
    input_ids = model["tokenizer"](text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

    # Generate text using the pre-trained model
    prediction = model["model"].generate(
        input_ids,
        num_beams=4,
        do_sample=False,
        use_cache=None,
        max_new_tokens=16,
        min_new_tokens=2,
        temperature=1.0,
        eos_token_id=[model["tokenizer"](eos, add_special_tokens=False).input_ids[1] for eos in [".\\", "\\.", "\\,", "\\;"]]
    )
    prediction = prediction[0, input_ids.size(1):]
    prediction = model["tokenizer"].decode(prediction)
    prediction = prediction.replace('\\n', "\n")
    prediction = prediction.strip().strip('\\')
    prediction = prediction.strip().split('\n')[0].strip().strip('\\').strip()
    prediction = normalize_answer(prediction)

    return prediction


def main():
    args = parse_args()
    random.seed(42)

    model = load_model(args.model_name_or_path)
    train_dataset, dev_dataset = load_data()

    log_file = open(
        f"eval_nq_{args.model_name_or_path.split('/')[-1]}_{args.n_shots}-shots.txt",
        "w",
    )

    accuracies = []
    for _ in range(args.n_repetitions):
        n_correct, n_total = 0, 0

        for i, example in enumerate(tqdm(dev_dataset)):
            shots = sample_random_examples(train_dataset, i, args.n_shots)
            questions = [example["question"] for example in shots] + [
                example["question"]
            ]
            answers = [example["preferred_answer"] for example in shots] + [
                example["preferred_answer"]
            ]

            prompt = format_prompt(
                questions, answers
            )
            prediction = generate(prompt, model)

            gold_answers = [normalize_answer(answer) for answer in example["answers"]]
            if prediction in gold_answers:
                n_correct += 1
            n_total += 1

            if i < 10:
                print()
                print(len(model["tokenizer"](prompt).input_ids))
                print(f"\nQuestion: {example['question']}")
                print(f"Gold answer: {example['preferred_answer']}")
                print(f"Pred answer: {prediction}\n", flush=True)

            if (i + 1) % 100 == 0:
                accuracy = n_correct / n_total
                print(f"Accuracy after {i + 1} examples: {accuracy:.2%}")

        accuracy = n_correct / n_total
        accuracies.append(accuracy)

        print(f"\nAccuracy: {accuracy:.2%}")
        log_file.write(f"{accuracy}\n")
        log_file.flush()

    log_file.write(
        f"\nAccuracy: {mean(accuracies)} ± {stdev(accuracies) if len(accuracies) > 1 else 0}\n"
    )
    log_file.close()


if __name__ == "__main__":
    main()