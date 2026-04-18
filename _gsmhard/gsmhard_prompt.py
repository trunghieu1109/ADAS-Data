import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from _mgsm.mgsm_prompt import (  # type: ignore
    COT,
    COT_SC,
    EXAMPLE,
    LLM_debate,
    QD,
    Reflexion,
    Reflexion_prompt_1,
    Reflexion_prompt_2,
    Role_Assignment,
    Take_a_step_back,
    system_prompt,
)

base = """# Overview
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent performing well on the GSM-Hard benchmark, which evaluates robust mathematical problem-solving under distribution shift with larger and more adversarial numeric values than standard grade-school math.

## An example question from GSM-Hard:

**Question**: Solve this math problem:\nA robe takes 2287720 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?

**Answer (Not Given)**: 3431580

# The utility code:

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

client = openai.OpenAI()

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!\n"

ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.1):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=1024,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    return json_dict

class LLMAgentBase:
    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.1) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        output_fields_and_description = {key: f"Your {key}." if 'answer' not in key else f"Your {key}. Return ONLY an integer. DO NOT return anything other than the integer answer." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

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

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    def forward(self, taskInfo) -> Union[Info, str]:
        pass
```

# Discovered architecture archive
Here is the archive of the discovered architectures:

[ARCHIVE]

The fitness value is the median and 95% Bootstrap Confidence Interval of the correct rate on a validation question set. Your GOAL is to maximize the "fitness".

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture.
The third key ("code") corresponds to the exact Python code that implements the agent architecture. Do not output any additional keys.

Here is an example:
[EXAMPLE]
"""


def get_init_archive():
    return [COT, COT_SC, Reflexion, LLM_debate, Take_a_step_back, QD, Role_Assignment]


def get_prompt(current_archive, adaptive=False):
    archive_str = ",\n".join([json.dumps(solution) for solution in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))
    return system_prompt, prompt


def get_reflexion_prompt(prev_example):
    prev_example_str = "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    prompt_1 = Reflexion_prompt_1.replace("[EXAMPLE]", prev_example_str) if prev_example else Reflexion_prompt_1.replace("[EXAMPLE]", "")
    return prompt_1, Reflexion_prompt_2