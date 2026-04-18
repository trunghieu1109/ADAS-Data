import os
from collections import OrderedDict, namedtuple
from typing import Union
import numpy as np
import json
import argparse
import copy
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import pandas
import random
import re

import openai
import backoff

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model_api import create_openai_client, get_default_model, get_json_completion, get_last_completion_raw
from run_logging import extract_reasoning, finish_query_logging, record_llm_invocation, start_query_logging, to_json_text, write_solution_run_outputs

client = create_openai_client()

from SVAMP_utils import get_all_examples, random_id, bootstrap_confidence_interval, score_fn
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

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
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY an integer. DO NOT return anything other than the integer answer." for key in self.output_fields}
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
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\n{content}\n\n'
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
            #try to fill in the missing field
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

def evaluate(args):
    eval_file_path = args.eval_file_path
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            test_entries = json.load(json_file)
    else:
        raise AssertionError(f"File {eval_file_path} does not exist.")

    for sol in test_entries:
        print(f"{sol['name']}")
        acc_list = evaluate_forward_fn(args, sol['code'], run_name=sol['name'])
        sol['test_fitness_Asdiv'] = bootstrap_confidence_interval(acc_list)

    # Step 5: Save the test entries
    with open(eval_file_path, 'w') as json_file:
        json.dump(test_entries, json_file, indent=4)


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
    examples = get_all_examples(args.data_filename)
    random.seed(args.shuffle_seed)
    random.shuffle(examples)

    if SEARCHING_MODE:
        base_examples = examples[:args.valid_size]
    else:
        base_examples = examples[args.valid_size:args.valid_size+args.test_size]

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
    run_group_name = os.path.splitext(os.path.basename(args.eval_file_path or args.data_filename))[0]
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
            correct_answer = answers[q_idx]
            correct = score_fn(correct_answer, extracted_answer)
        except Exception as e:
            extracted_answer = res
            correct = False

        score = 1 if correct else 0
        acc_list.append(score)
        attempt_scores[attempt_index].append(score)
        attempt_rows[attempt_index].append({
            "task": task_queue[q_idx].content,
            "code": forward_str,
            "output": to_json_text(extracted_answer),
            "expected_output": to_json_text(answers[q_idx]),
            "score": score,
            "execution_logs": to_json_text(execution_logs),
        })
    write_solution_run_outputs(os.path.dirname(__file__), run_group_name, run_name, SEARCHING_MODE, "score", attempt_rows, attempt_scores)
    print(f"acc: {bootstrap_confidence_interval(acc_list)}")
    return acc_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default='dataset/SVAMP.json')
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', '--n_repeat', dest='n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--eval_file_path', type=str, default='')
    parser.add_argument('--model',
                        type=str,
                        default=get_default_model())

    args = parser.parse_args()

    SEARCHING_MODE = False
    evaluate(args)
