import sys

sys.path.append("..")
from utils.instruction import *
import torch
import time
from .cache import GPTTopKCache, GPTSeqCache
import json
from time import sleep
from vllm import LLM, SamplingParams
from colorama import Fore, init
import re
from transformers import AutoModelForCausalLM

init(autoreset=True)

class DeepSeekCoderChat:
    def __init__(self, model_name, tokenizer, args):
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode
        self.horizon = args.horizon
        self.llm = AutoModelForCausalLM.from_pretrained(args.modelpath, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        self.tokenizer = tokenizer
        self.terminal_token = self.tokenizer.eos_token_id
        print(self.terminal_token)
        self.width = args.width
        self.top_k_cache_steps = args.top_k_cache_steps
        self.top_k_cache = GPTTopKCache(args.width, cache_steps=args.top_k_cache_steps, tokenizer=self.tokenizer,
                                        args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)


        self.failed_json_num = 0

    def generate_response_api(self, prompt, top_k=1, max_length=1024, system_message=None, temperature=0.0):
        sys_msg = "You are a great planner that generates plan to complete the given problem."
        if system_message:
            sys_msg = system_message

        # prompt = build_question_instruct(prompt, tool_string, demo_string)
        messages = [
            {'role': "system", 'content': sys_msg},
            {'role': "user", 'content': prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        # print(Fore.BLUE + full_prompt)

        response = self.llm.generate(
            inputs,
            max_new_tokens=max_length,
            do_sample=False,
            top_k=top_k,
            temperature=temperature,
        )
        outputs = self.tokenizer.decode(response[0][len(inputs[0]):], skip_special_tokens=True)
        log_prob = [] # 是一个length等于top k的list，每个位置是一个list{token: .., logprob:.., bytes:..}

        # input_token_num = len(response[0].prompt_token_ids)
        # output_token_num = len(response[0].outputs[0].token_ids)
        # self.args.total_input_token_num += input_token_num
        # self.args.total_output_token_num += output_token_num

        return outputs, log_prob
    
    def get_top_k_rationale_predict(self, state, depth, graph=None, with_verbal=False):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # 生成下面的line，以及line level的概率
            input_prompt = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            next_possible_tools = None
            if graph and depth != 0:
                print(Fore.BLUE + "Should contain the dependency prior.")
                pattern = r"Tool Call of Step (\d+)\:Call (.*) tool"
                result_match = re.findall(pattern, input_prompt)
                if result_match and int(result_match[-1][0]) == depth and result_match[-1][1] in graph.keys(): 
                    next_possible_tools = graph[result_match[-1][1]]
            with_instru_input_prompt = input_prompt + build_intermediate_instruct(depth, self.args.width, next_possible_tools)

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
                        top_lines.append([self.terminal_token])
                    else:
                        ele[f'{i + 1} Tool Call of Step {depth + 1}'] = f'Tool Call of Step {depth + 1}:' + ele[
                        f'{i + 1} Tool Call of Step {depth + 1}']
                        top_lines.append(self.tokenizer.encode(ele[f'{i + 1} Tool Call of Step {depth + 1}'] + '\n')) #allow_special maybe
                        top_lines_text.append(ele[f'{i + 1} Tool Call of Step {depth + 1}'])
            except Exception as e:
                self.failed_json_num += 1
                print(self.failed_json_num)
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