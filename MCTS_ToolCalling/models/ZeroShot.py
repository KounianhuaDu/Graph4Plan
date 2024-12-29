import os
import sys

sys.path.append("..")

import time
from math import sqrt, log
import re
from collections import defaultdict, deque
import numpy as np
from transformers import LlamaTokenizer, GemmaTokenizer
import tiktoken

from transformers import AutoTokenizer, AutoModelForCausalLM

# from executors import AppsExecutor
from ChatModels import *
from models import *
from utils.instruction import *

from colorama import Fore, init

from utils.instruction import *
import json

class ZeroShotStep:
    def __init__(self, args):
        self.args = args
        self.sample_nums = 0
        self.gamma = 0.9

        ## Backbone LLM
        if 'gpt4omini' in args.arch:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.generator = GPTStepChat(args.arch, self.tokenizer, args)
        elif 'llama' in args.arch:
            self.tokenizer = AutoTokenizer.from_pretrained(args.modelpath, use_fast=False, padding_side='left')
            self.generator = LlamaChat(args.arch, self.tokenizer, args)
        elif 'gemma' in args.arch:
            self.tokenizer = GemmaTokenizer.from_pretrained(args.modelpath)
            self.generator = GemmaChat(args.arch, self.tokenizer, args)
        elif 'deepseek' in args.arch:
            # self.tokenizer = None
            self.generator = DeepSeekChat(args.arch, None, args)
            self.tokenizer = self.generator.tokenizer
        elif 'yi' in args.arch:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.generator = YiChat(args.arch, self.tokenizer, args)
        else:
            raise NotImplementedError
        self.root = None
        self.cached_reward = {}
        self.cached_value = {}
        self.cached_verbal_feedback = {}
        self.cur_prob_instance = None
        self.sample_times = []
        self.st = time.time()

    def generate(self, problem_instance, tool_string, demo_string, dependency_prior=None):
        # self.cur_prob_instance = problem_instance
        raw_prompt = problem_instance["user_request"]

        raw_prompt = build_question_instruct(raw_prompt, tool_string, demo_string, self.args.dataset)

        message, log_prob = self.generator.generate_response_api(raw_prompt)

        # print(Fore.RED +"message:"+str(message))
        message= self.convert_state_to_program(message)

        print(message)

        return message, []  # No trace needed since MCTS is skipped

    # def convert_state_to_program(self, s):
    #     # s = self.tokenizer.decode(s)

    #     pattern = r'\{"task_steps": \[.*?\], "task_nodes": \[.*?\], "task_links": \[.*?\]\}'
    #     # Extract the required part
    #     result_match = re.findall(pattern, s)
    #     # print(Fore.RED + "result_match:", result_match)
    #     fixed_json_output = result_match[0] if result_match else None

    #     if fixed_json_output:
    #         # content = fixed_json_output
    #         try:
    #             # Load the JSON object
    #             content = json.loads(fixed_json_output)
    #             # print('in convert_state_to_program()',content)
    #         except json.JSONDecodeError as e:
    #             print(f"JSON decoding error: {e}")
    #             content = {"task_steps": [], "task_nodes": [], "task_links": []}
    #     else:
    #         print("No valid result found.")
    #         content = {"task_steps": [], "task_nodes": [], "task_links": []}

    #     return content
    
    def convert_state_to_program(self, s):
        if isinstance(s, list):
            s = self.tokenizer.decode(s, skip_special_tokens=True)

        if '```json' in s:
            s = s.split('```json')[1].split('```')[0]
        s = s.replace("\n", "")
        s = s.replace("\_", "_")
        s = s.replace("\\", "")
        # print(Fore.BLUE + s)
        # pattern = r'\{"task_steps": \[.*?\], "task_nodes": \[.*?\], "task_links": \[.*?\]\}'
        # Extract the required part
        # result_match = re.findall(pattern, s)
        # print(Fore.RED + "result_match:", result_match)
        # fixed_json_output = result_match[0] if result_match else None

        if True:
            try:
                # Load the JSON object
                # print(s)
                content = json.loads(s)
                if isinstance(content["task_steps"][0], dict):
                    task_steps = []
                    for step in content["task_steps"]:
                        task_steps += step.values()
                    content["task_steps"] = task_steps
                # print(content)
            except json.JSONDecodeError as e:
                try:
                    s_debug = s.split("task_nodes")
                    steps = s_debug[0][1:]
                    if '}' in steps:
                        steps = steps.replace('{', "")
                        steps = steps.replace('}', "")
                        s0 = "{" + steps + "task_nodes" + s_debug[1]
                    content = json.loads(s0)
                except:
                    print(Fore.RED + s)
                    print(f"JSON decoding error: {e}")
                    content = {"task_steps": [], "task_nodes": [], "task_links": []}
        else:
            print("No valid result found.")
            content = {"task_steps": [], "task_nodes": [], "task_links": []}

        return content
