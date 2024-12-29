import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# torch.multiprocessing.set_start_method('spawn')
from colorama import Fore, Back, Style, init
import re
import time
from data_handling.GRAPHHandler import *
from utils import *
import time
from models import *
# from data_handling.APPSHandler import *
import json
# import dgl
import matplotlib as mpl
import networkx as nx
from matplotlib import pyplot as plt
from collections import defaultdict
import argparse
from argument import parse_args

from colorama import Fore, init


def train_mcts(args):
    hander = GRAPHHander(args)

    inputs, tool_string, demo_string, dependency_prior = hander.get_data()

    # Hardcoded values for the options

    # model_path = args.modelpath  # "/ext0/hcchai/codemate/llama3/Meta-Llama-3-8B"
    # temperature = 0.5

    # print('model_path:', model_path)
    # multiworker = 1
    if args.model == 'ZeroShot':
        model = ZeroShotStep(args)
        print('===== ZeroShotStep =====')
    elif args.model == 'MCTSStep':
        model = MCTSStep(args)
        print('===== MCTSStep =====')

    # model = ZeroShotStep(args)
    if args.provide_graph:
        prediction_dir = f"data/predictions/{args.dataset}-graph/{args.arch}/{args.model}"
    else:
        prediction_dir = f"data/predictions/{args.dataset}/{args.arch}/{args.model}"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    wf_name = f"{prediction_dir}/has_inferenced.json"
    wf_name1 = f"{prediction_dir}/output_dict.json"
    wf_name2 = f"{prediction_dir}/trace_feats.json"
    #

    # num_train = len(inputs) args.problem_indices
    num_train = args.problem_indices if len(inputs) > args.problem_indices else len(inputs)
    print(Fore.RED + "num_train:", num_train)

    # print('The lenth of inputs:', len(inputs))

    for idx, input in enumerate(inputs[:num_train]):
        print(f"Solving problem #{idx}")
        st_time = time.time()
        # print('The lenth of input:',len(input))
        print(Fore.CYAN + "input:\n", input)
        output_dict, trace_feats = model.generate(input, tool_string, demo_string, dependency_prior)

        # print(Fore.GREEN + "output_dict:", str(output_dict))
        # print(Fore.BLUE + "trace_feats:", str(trace_feats))

        with open(wf_name, "a") as wf:
            res = {"id": input["id"], "user_request": input["user_request"]}
            # json.dump(res, wf)
            wf.write(json.dumps(res) + "\n")
            wf.flush()
            print('===== save wf =====')

        with open(wf_name1, "a") as wf1:
            output_dict["id"] = input["id"]
            output_dict["user_request"] = input["user_request"]
            wf1.write(json.dumps(output_dict) + "\n")
            wf1.flush()
            # json.dump(output_dict, wf1)
            print('===== save wf1 =====')

        with open(wf_name2, "a") as wf2:
            # json.dump(trace_feats, wf2)
            wf2.write(json.dumps(trace_feats) + "\n")
            wf2.flush()
            print('===== save wf2 =====')


if __name__ == "__main__":

    default_rollout = 3
    default_out_path = "./train_trace"

    args = parse_args(default_out_path=default_out_path, default_rollout=default_rollout)

    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
        # args.device = 'cuda'
    else:
        args.device = 'cpu'

    args.problem_indices = 100

    print(args)

    import openai

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # API_KEY = "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"
    # API_KEY = "sk-Qsp6KOh1uitRBpJEF4936f0dB4C74a1281FcFc132aAd3337" # huawei
    API_KEY = None
    openai.api_key = API_KEY

    os.environ["http_proxy"] = "http://127.0.0.1:8888"
    os.environ["https_proxy"] = "http://127.0.0.1:8888"
    os.environ["all_proxy"] = "socks5://127.0.0.1:8889"
    os.environ["OPENAI_API_KEY"] = API_KEY

    os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

    print("=====")
    print("args.arch:", args.arch)

    if args.arch == 'llama3-8b-it':
        args.modelpath = os.path.join(args.modelweight, 'llama3/Meta-Llama-3-8B-Instruct')
    elif args.arch == 'gemma-7b-it':
        args.modelpath = os.path.join(args.modelweight, 'gemma/gemma-7b-it')
    elif args.arch == 'gemma-2-27b-it':
        args.modelpath = os.path.join(args.modelweight, 'gemma-2-27b-it')
    elif args.arch == 'deepseek':
        args.modelpath == os.path.join(args.modelweight, 'DeepSeek-V2-Lite-Chat')

    # print('args.modelpath:', args.modelpath)

    train_mcts(args)
