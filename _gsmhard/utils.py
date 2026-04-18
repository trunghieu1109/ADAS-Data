import json
import random
import string

import numpy as np


def score_gsmhard(target: str, prediction: str) -> bool:
    try:
        target_value = float(target)
        prediction_value = float(prediction)
    except Exception:
        return False

    return abs(target_value - prediction_value) < 1e-1


def get_all_examples(filepath):
    with open(filepath, mode="r", encoding="utf-8") as file_obj:
        examples = [json.loads(line) for line in file_obj]

    for example in examples:
        example["inputs"] = "Solve this math problem:\n" + example["input"]
        example["targets"] = example["target"]

    return examples


def random_id(length=4):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    data = np.array(data)

    bootstrap_means = []
    for _ in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    bootstrap_means = np.array(bootstrap_means)

    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    median = np.median(bootstrap_means)

    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"