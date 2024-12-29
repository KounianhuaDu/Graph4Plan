# -*- coding:utf-8 _*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import math
import re
import json
from .cache import GPTTopKCache, GPTSeqCache
from utils.instruction import *
from colorama import Fore, init
import os

# generate response api
# generate top k rationale
# get_rationale_predicted_sequence

from colorama import Fore, Back, Style, init


class LlamaChat:
    def __init__(self, model_name, tokenizer, args):
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = (0)  # unk. we want this to be different from the eos token
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode
        self.horizon = args.horizon
        # self.terminal_token = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(args.modelpath).to(self.device)
        self.terminal_token = self.tokenizer.encode('</s>')[
            0]  # '</s>' is often used as an end token in LLAMA  or self.tokenizer.encode('[/INST]')[0]
        self.width = args.width
        self.top_k_cache_steps = args.top_k_cache_steps
        self.top_k_cache = GPTTopKCache(args.width, cache_steps=args.top_k_cache_steps, tokenizer=tokenizer,
                                        args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)

    def generate_response_api(self, prompt, top_k=0.5, max_length = 1024,
                              system_message=None, temperature=0.5):


        sys_msg = "You are a great planner that generates plan to complete the given problem."
        if system_message:
            sys_msg = system_message
        # Prepare the prompt by combining system_message and user prompt
        full_prompt = sys_msg + "\n" + prompt

        # print(Fore.GREEN + full_prompt)

        # Tokenize the input prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True).to(self.device)

        # Generate the response
        output = self.model.generate(
            input_ids=inputs.input_ids,
            # attention_mask=inputs.attention_mask,
            max_new_tokens=int(max_length),
            temperature=temperature,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            # top_k=top_k,
        )

        # message = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # print(Fore.BLUE + "message0:", message)
        #
        # print("tpye:", type(inputs))

        log_probs_for_generated_tokens = None  # Initialize to handle cases where it's not needed

        response = output[0][inputs.input_ids.shape[1]:]
        message = self.tokenizer.decode(response)

        # If log probabilities are required, calculate them manually
        with torch.no_grad():
            logits = self.model(output).logits
            log_probs = torch.log_softmax(logits, dim=-1)
              # Ignore the input tokens

            if response.dim() == 1:
                response = response.unsqueeze(0)

            # Ensure dimensions match before gathering log_probs
            if log_probs.size(1) >= response.size(1):
                log_probs_for_generated_tokens = log_probs[:, -response.size(1):, :].gather(2, response.unsqueeze(-1)).squeeze(-1)
            else:
                raise ValueError("Logits dimension does not match the generated tokens dimension.")

        input_token_num = len(self.tokenizer.encode(prompt))
        output_token_num = len(self.tokenizer.encode(message))
        self.args.total_input_token_num += input_token_num
        self.args.total_output_token_num += output_token_num

        return message, log_probs_for_generated_tokens.tolist() if log_probs_for_generated_tokens is not None else None

    def get_top_k_rationale_predict(self, state, depth, with_verbal=False):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # Decode input tokens to get the input prompt
            input_prompt = self.tokenizer.decode(input_ids[0].tolist())

            with_instru_input_prompt = input_prompt + build_intermediate_instruct(depth, self.args.width)

            print('\n-----------------Input (Generate Thought)-----------------')
            # print(Fore.GREEN + with_instru_input_prompt)

            response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1)
            print('\n-----------------Output (Thought)-----------------')
            # print(Fore.YELLOW + response_text)

            # Remove code blocks if necessary
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
                # print(Fore.RED + "Extracted json lines:")
                # print(response_text)
                # print(Fore.RED + "Extracted json lines end.")
                top_scores = []
                top_lines = []
                top_lines_text = []

                keys = response_text[0].keys()
                if 'Reasonableness' in keys:
                    for i, ele in enumerate(response_text):
                        top_scores.append(ele['Reasonableness'])
                        ele[f'{i + 1} Clue of Step {depth + 1}'] = f'Clue of Step {depth + 1}:' + ele[
                            f'{i + 1} Clue of Step {depth + 1}']
                        top_lines.append(
                            self.tokenizer(ele[f'{i + 1} Clue of Step {depth + 1}'] + '\n', return_tensors="pt",
                                           padding=True).input_ids[0].tolist())
                        top_lines_text.append(ele[f'{i + 1} Clue of Step {depth + 1}'])
                else:
                    for i, ele in enumerate(response_text):
                        ele[f'{i + 1} Clue of Step {depth + 1}'] = f'Clue of Step {depth + 1}:' + ele[
                            f'{i + 1} Clue of Step {depth + 1}']
                        # self.tokenizer(ele[f'{i+1} Clue of Step {depth+1}'] + '\n', return_tensors="pt", padding=True)
                        top_lines.append(
                            self.tokenizer(ele[f'{i + 1} Clue of Step {depth + 1}'] + '\n', return_tensors="pt",
                                           padding=True).input_ids[0].tolist())
                        top_lines_text.append(ele[f'{i + 1} Clue of Step {depth + 1}'])
                        top_scores = [1.0 for _ in range(self.width)]

            except Exception as e:
                top_lines = [self.tokenizer.encode('\n') for _ in range(self.width)]
                top_scores = [1.0 for _ in range(self.width)]

            # print(Fore.RED + "Extracted lines:")
            # print(top_lines)
            # print(Fore.RED + "Extracted lines end.")
            return top_lines, top_scores

    def get_rationale_predicted_sequence(self, state, problem, horizon=None, renewchild_count=0):
        with torch.no_grad():
            input_ids = state  # as a list
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)

            # 使用序列缓存，如果缓存中已有结果则直接返回
            output_ids = self.seq_cache.get(input_ids)
            if output_ids is not None:
                return output_ids

            # 解码输入ID以获得提示
            input_prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            # input_prompt = self.tokenizer.decode(input_ids[0].tolist())
            previous_thoughts = input_prompt.split('-----Clues-----')[-1]

            with_instru_input_prompt = get_reward_instruct(previous_thoughts, problem)

            print('\n-----------------Input with Thought (Generate Code)-----------------')
            # print(Fore.GREEN + with_instru_input_prompt)

            response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1)

            print('\n-----------------Output (Code)-----------------')
            # print(Fore.YELLOW + response_text)

            # 将生成的响应文本编码为序列
            sequences = self.tokenizer(response_text, return_tensors="pt", padding=True).input_ids.to(self.device)
            # sequences = self.tokenizer.encode(response_text)
            model_output = WithProbReturn(
                sequences=sequences,
                scores=log_probs,
                attentions=None,
                hidden_states=None,
                beam_indices=None
            )

            # 更新缓存
            if self.top_k_cache_steps > 0:
                self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores)

            output_ids_list = model_output.sequences.tolist()
            output_ids = output_ids_list[0]

            # 将结果存入序列缓存
            self.seq_cache.add(input_ids, output_ids)

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
