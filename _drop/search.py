import argparse
import copy
import json
import os
import random
import sys
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model_api import create_openai_client, get_default_model, get_json_completion, get_last_completion_raw
from run_logging import extract_reasoning, finish_query_logging, record_llm_invocation, start_query_logging, to_json_text, write_solution_run_outputs

from drop_prompt import get_init_archive, get_prompt, get_reflexion_prompt

client = create_openai_client()

from utils import random_id, bootstrap_confidence_interval, load_drop, drop_metric

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True
MAX_ERROR_HISTORY = 5
MAX_FAILURE_MEMORY = 3

POSTPROCESS_SOLUTION_PROMPT = """You are post-processing a candidate agent architecture so it can run inside a Python benchmark runtime.

Return a valid JSON object only.

Requirements:
1. Preserve the same overall algorithmic idea unless it is obviously malformed.
2. Ensure the JSON object contains at least the keys 'thought', 'name', and 'code'. Keep optional keys like 'debug_thought' or 'reflection' only if they are already present and still useful.
3. The 'code' field must be complete runnable Python code for a function named forward(self, taskInfo).
4. The code must define exactly one top-level callable named forward. Do not define any additional top-level functions, classes, lambdas, or helper aliases.
5. Put any imports, helper functions, constants, or utilities inside forward so exec(...) leaves only the symbol forward in the local namespace.
6. The code will be executed directly inside Python. Fix any issues that would make it fail at runtime.
7. Do not assume LLM outputs are always raw strings. If the code parses agent outputs, make it robust to already-parsed Python values such as dict, list, bool, or strings.
8. If the code uses json.loads, only do so after checking that the value is an instance of str, bytes, or bytearray. Otherwise use the value directly.
9. Do not add print statements, tests, markdown fences, or explanations outside the JSON object.

Your goal is not just to make the JSON valid. Your goal is to make the returned Python code runnable in this benchmark runtime."""


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.1
):
    json_dict = get_json_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=8192,
        stop=None,
    )
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
):
    json_dict = get_json_completion(
        client=client,
        model=model,
        messages=msg_list,
        temperature=temperature,
        max_tokens=8192,
        stop=None,
    )
    assert not json_dict is None
    return json_dict


class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model=get_default_model(), temperature=0.1) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Directly answer the question. Keep it very concise." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            # print(e)
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        record_llm_invocation(
            prompt=f"System:\n{system_prompt}\n\nUser:\n{prompt}",
            invoker_name=self.agent_name,
            output=get_last_completion_raw() or response_json,
            reasoning=extract_reasoning(response_json),
        )
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self) -> None:
        pass


def _normalize_error_text(error_text):
    normalized_text = " ".join(str(error_text).strip().split())
    return normalized_text[:300]


def _get_error_fingerprint(error):
    error_type = type(error).__name__
    error_text = _normalize_error_text(error)
    return f"{error_type}:{error_text.lower()}"


def _build_error_entry(error, attempt_idx, error_history):
    error_type = type(error).__name__
    error_text = _normalize_error_text(error)
    fingerprint = _get_error_fingerprint(error)
    repeated_count = sum(entry["fingerprint"] == fingerprint for entry in error_history) + 1
    return {
        "attempt": attempt_idx + 1,
        "error_type": error_type,
        "error_text": error_text,
        "fingerprint": fingerprint,
        "is_repeated": repeated_count > 1,
        "repeat_count": repeated_count,
    }


def _format_error_history(error_history):
    if not error_history:
        return "No previous debug failures."

    formatted_entries = []
    for entry in error_history[-MAX_ERROR_HISTORY:]:
        repeated_suffix = ""
        if entry["is_repeated"]:
            repeated_suffix = f" (repeated failure x{entry['repeat_count']})"
        formatted_entries.append(
            f"Attempt {entry['attempt']}: {entry['error_type']}: {entry['error_text']}{repeated_suffix}"
        )
    return "\n".join(formatted_entries)


def _format_failure_memory(recent_failure_summaries):
    if not recent_failure_summaries:
        return ""

    summary_lines = ["Recent failed generations to avoid repeating:"]
    for summary in recent_failure_summaries[-MAX_FAILURE_MEMORY:]:
        summary_lines.append(f"- {summary}")
    return "\n".join(summary_lines)


def _summarize_generation_failure(generation_idx, solution_name, error_history):
    if not error_history:
        return f"Generation {generation_idx} candidate {solution_name} failed without a captured evaluation error."

    latest_error = error_history[-1]
    repeated_failures = [entry for entry in error_history if entry["is_repeated"]]
    repeated_suffix = ""
    if repeated_failures:
        repeated_suffix = f" Repeated pattern: {repeated_failures[-1]['error_type']}: {repeated_failures[-1]['error_text']}."
    return (
        f"Generation {generation_idx} candidate {solution_name} failed after {len(error_history)} debug attempts. "
        f"Last error: {latest_error['error_type']}: {latest_error['error_text']}.{repeated_suffix}"
    )


def _postprocess_generated_solution(solution, model):
    repair_messages = [
        {"role": "system", "content": POSTPROCESS_SOLUTION_PROMPT},
        {"role": "user", "content": to_json_text(solution)},
    ]
    return _get_json_response_with_retries(repair_messages, model, temperature=0.2)


def _is_json_parse_error(error):
    if isinstance(error, (json.JSONDecodeError, SyntaxError)):
        return True

    error_text = str(error).lower()
    return any(token in error_text for token in [
        "model returned an empty response",
        "empty response",
        "unterminated string",
        "expecting value",
        "invalid control character",
        "failed to parse model response as json",
        "json",
    ])


def _get_json_response_with_retries(msg_list, model, temperature=0.1, max_attempts=5):
    current_messages = list(msg_list)
    last_error = None

    for attempt_idx in range(max_attempts):
        try:
            return get_json_response_from_gpt_reflect(current_messages, model, temperature=temperature)
        except Exception as error:
            last_error = error
            if not _is_json_parse_error(error) or attempt_idx == max_attempts - 1:
                raise

            current_messages = current_messages + [{
                "role": "user",
                "content": (
                    f"Your previous reply could not be parsed as valid JSON ({_normalize_error_text(error)}).\n"
                    "Return a single valid JSON object only.\n"
                    "Keep reasoning extremely brief and do not spend tokens on hidden or visible step-by-step analysis.\n"
                    "Do not include long explanations or extra prose.\n"
                    "Do not use markdown fences.\n"
                    "Do not leave any string unterminated.\n"
                    "Preserve the same intended fields and overall solution content."
                ),
            }]

    raise last_error


def search(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"], run_name=solution["name"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    recent_failure_summaries = []

    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        failure_memory_text = _format_failure_memory(recent_failure_summaries)
        if failure_memory_text:
            msg_list.append({"role": "user", "content": failure_memory_text})
        try:
            next_solution = _get_json_response_with_retries(msg_list, args.model)

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            # Reflexion 1
            msg_list.append({"role": "assistant", "content": to_json_text(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = _get_json_response_with_retries(msg_list, args.model)
            # Reflexion 2
            msg_list.append({"role": "assistant", "content": to_json_text(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = _get_json_response_with_retries(msg_list, args.model)
            next_solution = _postprocess_generated_solution(next_solution, args.model)
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            n -= 1
            continue

        acc_list = []
        error_history = []
        for debug_idx in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"], run_name=next_solution["name"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                error_entry = _build_error_entry(e, debug_idx, error_history)
                error_history.append(error_entry)
                error_history_text = _format_error_history(error_history)
                repeated_failure_text = ""
                if error_entry["is_repeated"]:
                    repeated_failure_text = (
                        f"\nRepeated failure detected for fingerprint {error_entry['fingerprint']}. "
                        "Your previous fix did not resolve the root cause."
                    )
                msg_list.append({"role": "assistant", "content": to_json_text(next_solution)})
                msg_list.append({"role": "user", "content": (
                    f"Error during evaluation:\n{error_entry['error_type']}: {error_entry['error_text']}\n\n"
                    f"Failure history for this generation:\n{error_history_text}{repeated_failure_text}\n\n"
                    "Carefully consider where you went wrong in your latest implementation. "
                    "Using insights from the failure history above, try to debug the current code to implement the same thought. "
                    "If a failure pattern is repeated, explicitly identify the root cause and change strategy instead of repeating the same fix. "
                    "The benchmark runtime expects exactly one top-level callable named forward; move all imports and helper logic inside forward and do not define any other top-level names. "
                    "Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'."
                )})
                try:
                    next_solution = _get_json_response_with_retries(msg_list, args.model)
                    next_solution = _postprocess_generated_solution(next_solution, args.model)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue
        if not acc_list:
            recent_failure_summaries.append(
                _summarize_generation_failure(n + 1, next_solution.get("name", "unknown"), error_history)
            )
            recent_failure_summaries = recent_failure_summaries[-MAX_FAILURE_MEMORY:]
            n -= 1
            continue

        if error_history:
            recent_failure_summaries = []

        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1

        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


def evaluate(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = archive[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        current_idx += 1
        try:
            acc_list = evaluate_forward_fn(args, sol["code"], run_name=sol["name"])
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)


def evaluate_forward_fn(args, forward_str, run_name="archive"):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    # set seed 0 for valid set
    examples = load_drop(args.data_filename)[1:-1]  # first one and the last one is for few-shot examples
    random.seed(args.shuffle_seed)
    random.shuffle(examples)

    if SEARCHING_MODE:
        base_examples = examples[:args.valid_size]
    else:
        base_examples = examples[args.valid_size:args.valid_size + args.test_size]

    examples = []
    attempt_indices = []
    for attempt_index in range(1, args.n_repreat + 1):
        examples.extend(base_examples)
        attempt_indices.extend([attempt_index] * len(base_examples))

    questions = [example['inputs'] for example in examples]
    answers = [example['targets'] for example in examples]

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()
    attempt_rows = {attempt_index: [] for attempt_index in range(1, args.n_repreat + 1)}
    attempt_scores = {attempt_index: [] for attempt_index in range(1, args.n_repreat + 1)}

    acc_list = []

    def call_forward(task_info):
        start_query_logging()
        try:
            result = agentSystem.forward(task_info)
            return result, finish_query_logging()
        except Exception:
            execution_logs = finish_query_logging()
            raise

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(call_forward, task_queue), total=len(task_queue)))

    for q_idx, (res, execution_logs) in enumerate(results):
        attempt_index = attempt_indices[q_idx]
        try:
            if isinstance(res, Info):
                extracted_answer = res.content
            else:
                extracted_answer = res
            correct_answers = answers[q_idx]
            em_score, f1_score = drop_metric(extracted_answer, correct_answers)
        except Exception as e:
            extracted_answer = res
            correct_answers = answers[q_idx]
            f1_score = 0

        acc_list.append(f1_score)
        attempt_scores[attempt_index].append(f1_score)
        attempt_rows[attempt_index].append({
            "task": task_queue[q_idx].content,
            "code": forward_str,
            "output": to_json_text(extracted_answer),
            "expected_output": to_json_text(correct_answers),
            "score": f1_score,
            "execution_logs": to_json_text(execution_logs),
        })
    write_solution_run_outputs(os.path.dirname(__file__), args.expr_name, run_name, SEARCHING_MODE, "f1", attempt_rows, attempt_scores)
    print(f"f1: {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="dataset/drop_v0_dev.jsonl.gz")
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', '--n_repeat', dest='n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default="drop_gpt3.5_results")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--skip_evaluate', action='store_true', default=False)
    parser.add_argument('--model',
                        type=str,
                        default=get_default_model())

    args = parser.parse_args()
    # search
    SEARCHING_MODE = True
    search(args)

    # evaluate
    if not args.skip_evaluate:
        SEARCHING_MODE = False
        evaluate(args)
