import os
import sys

sys.path.append("..")
from utils.instruction import *
import torch
from openai import OpenAI, InternalServerError
import time
from .cache import GPTTopKCache, GPTSeqCache
from utils.instruction import *
import json
from time import sleep
from colorama import Fore, init

init(autoreset=True)


class GPTStepChat:
    def __init__(self, model_name, tokenizer, args, save_mid_json=[]):
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode
        self.horizon = args.horizon
        self.client = OpenAI()
        self.client.base_url = "http://open.xiaoai.one/v1"
        self.terminal_token = self.tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        self.width = args.width
        self.top_k_cache_steps = args.top_k_cache_steps
        self.top_k_cache = GPTTopKCache(args.width, cache_steps=args.top_k_cache_steps, tokenizer=tokenizer,
                                        args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)
        self.save_mid_json = save_mid_json

    def generate_response_api(self, prompt, top_k=1, max_length=1024, system_message=None, temperature=0.0):
        sys_msg = "You are a great planner that generates plan to complete the given problem."
        if system_message:
            sys_msg = system_message

        # prompt = build_question_instruct(prompt, tool_string, demo_string)

        full_prompt = sys_msg + "\n" + prompt

        if self.args.arch == 'gpt4omini':
            try:
                response = self.client.chat.completions.create(
                    model='gpt-4o-mini-2024-07-18',
                    messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": full_prompt}],
                    max_tokens=max_length,  # 调整生成文本的长度
                    temperature=temperature,
                    # top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    logprobs=True,
                    top_logprobs=top_k,
                )
                message = response.choices[0].message.content
                log_prob = response.choices[
                    0].logprobs.content  # 是一个length等于top k的list，每个位置是一个list{token: .., logprob:.., bytes:..}
            except InternalServerError:
                message = json.dumps({"task_steps":[], "task_nodes":[], "task_links":[]})
                log_prob = []
        else:
            raise NotImplementedError

        input_token_num = len(self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'}))
        output_token_num = len(self.tokenizer.encode(message, allowed_special={'<|endoftext|>'}))
        self.args.total_input_token_num += input_token_num
        self.args.total_output_token_num += output_token_num

        return message, log_prob

    def get_top_k_rationale_predict(self, state, depth, with_verbal=False):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # 生成下面的line，以及line level的概率
            input_prompt = self.tokenizer.decode(input_ids[0].tolist())

            with_instru_input_prompt = input_prompt + build_intermediate_instruct(depth, self.args.width)

            print('\n-----------------Input (Generate sub-funtion)-----------------')
            print(Fore.GREEN + with_instru_input_prompt.split("-----Tool Calls-----")[1])

            response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=2048,
                                                                  temperature=0.0)
            print('\n-----------------Output (sub-function)-----------------')
            print(Fore.YELLOW + response_text)


            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
                # print(Fore.RED + "Extracted lines:")
                # print(response_text)
                # print(Fore.RED + "Extracted lines end.")
            try:
                if response_text.strip()[0] != '[':
                    response_text = '[' + response_text + ']'

                    # print(Fore.RED + "Extracted lines:")
                    # print(response_text)
                    # print(Fore.RED + "Extracted lines end.")
                response_text = json.loads(response_text)
                print(Fore.RED + "Extracted json lines:")
                print(response_text)
                print(Fore.RED + "Extracted json lines end.")
                top_scores = []
                top_lines = []
                top_lines_text = []
                for i, ele in enumerate(response_text):
                    top_scores.append(ele['Reasonableness'])
                    if ele[f'{i + 1} Tool Call of Step {depth + 1}'] == "Finish":
                        top_lines.append(self.terminal_token)
                    else:
                        ele[f'{i + 1} Tool Call of Step {depth + 1}'] = f'Tool Call of Step {depth + 1}:' + ele[
                        f'{i + 1} Tool Call of Step {depth + 1}']
                        top_lines.append(self.tokenizer.encode(ele[f'{i + 1} Tool Call of Step {depth + 1}'] + '\n')) #allow_special maybe
                        top_lines_text.append(ele[f'{i + 1} Tool Call of Step {depth + 1}'])
            except Exception as e:
                # self.args.failed_json_num += 1
                print(e)
                top_lines = [self.tokenizer.encode('\n') for i in range(self.width)]
                top_scores = [1.0 for i in range(self.width)]
            

            print(Fore.RED + "Extracted lines:")
            print(top_lines)
            print(Fore.RED + "Extracted lines end.")
            return top_lines, top_scores

    def get_rationale_predicted_sequence(self, state, problem, horizon=None, renewchild_count=0):
        with torch.no_grad():
            encoded_ids = state  # as a list
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # use_seq_cache:
            output_ids = self.seq_cache.get(encoded_ids)
            if output_ids is not None:
                return output_ids

            input_prompt = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            problem = input_prompt.split("Problem Description:")[1].split("# RESULT #:")
            problem = problem[0] + problem[1]
            # print(Fore.BLUE + input_prompt)
            previous_thoughts = input_prompt.split('-----Tool Calls-----')[-1]

            with_instru_input_prompt = get_reward_instruct(previous_thoughts, problem)

            print('\n-----------------Input with Tools (Generate Plan)-----------------')
            print(Fore.GREEN + with_instru_input_prompt.split("-----Tool Calls-----")[1])

            response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=2048)
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]

            print('\n-----------------Output (Plan)-----------------')
            print(Fore.YELLOW + response_text)

            sequences = self.tokenizer.encode(response_text)
            model_output = WithProbReturn(sequences=torch.tensor(sequences).unsqueeze(0).to(self.device),
                                          scores=log_probs,
                                          attentions=None,
                                          hidden_states=None,
                                          beam_indices=None)

            if self.top_k_cache_steps > 0:
                if hasattr(model_output, 'beam_indices'):
                    # beam search output
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores,
                                         beam_indices=model_output.beam_indices)
                else:
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores)

            output_ids_list = model_output.sequences.tolist()

            output_ids = output_ids_list[0]

            # use_seq_cache
            self.seq_cache.add(encoded_ids, output_ids)

            self.time_stamps.append(time.time())
            return output_ids

    def clean_cache(self):
        self.top_k_cache = GPTTopKCache(self.args.width, cache_steps=self.args.top_k_cache_steps,
                                        tokenizer=self.tokenizer, args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)
        self.time_stamps = []


class WithProbReturn:
    def __init__(self, sequences, scores, attentions, hidden_states, beam_indices=None, top_tokens=None):
        self.sequences = sequences
        self.scores = scores
        self.attentions = attentions
        self.hidden_states = hidden_states
        self.beam_indices = beam_indices
        self.top_tokens = top_tokens
