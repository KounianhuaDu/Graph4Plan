# argument.py

import argparse

def parse_args(default_out_path="./train_trace", default_rollout=16, arch_choices=None):
    parser = argparse.ArgumentParser(
        description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ## dataset related
    parser.add_argument("--dataset", default="multimedia", help="Dataset to use, default: APPS") # "multimedia"  "huggingface" "dailylife"
    parser.add_argument("--data_path", default="./data", help="Path to save the data")
    parser.add_argument("--public_cases_type", type=str, default='half', help="Number of public test cases to use for evaluation.")
    parser.add_argument('--provide_graph', action='store_true', default=False, help="If True, dependency prior of tools is provided.")

    ## output & log
    parser.add_argument("--out_path", default=default_out_path, help="Path to save the output")
    parser.add_argument("--res_path", default="./evaluation_res", help="Path to save the evaluation")

    ## backbone LLM
    # parser.add_argument("--arch", default="gpt4omini", choices=['gpt4omini', 'llama', 'gemma'])
    # parser.add_argument("--arch", default="llama", choices=['gpt4omini', 'llama', 'gemma'])
    parser.add_argument("--arch", default="llama3-8b-it")  # ['gpt4omini', 'llama3-8b-it', 'gemma-2-27b-it','gemma-7b-it']

    # parser.add_argument('-m', '--model', type=str, default='mctsRationale', choices=['mctsag', 'mcts', 'bs', 'sample', 'ldb', 'mctsRationale'])
    parser.add_argument('-m', '--model', type=str, default='ZeroShot',  choices=['ZeroShot', 'MCTSStep'])
    parser.add_argument("--modelpath", default="/home/jxliu/models/DeepSeek-V2-Lite-Chat", help="Path of the model.")
    parser.add_argument("--modelweight", default="/ext0/hcchai/codemate", help="Path to save the models.")
    parser.add_argument('--total_input_token_num', type=int, default=0, help='The maximum number of tokens to input.')
    parser.add_argument('--total_output_token_num', type=int, default=0, help='The maximum number of tokens to output.')

    ## MCTS related
    parser.add_argument("--width", default=3, type=int, help="The maximum number of children for any node.")
    parser.add_argument("--horizon", default=3072, type=int, help="The maximum number of tokens to generate.")
    parser.add_argument("--num-beams", default=1, type=int, help="The number of beams for beam search or PG-TD.")
    parser.add_argument("--num-samples", default=1, type=int, help="The number of samples for Sampling + Filtering.")
    parser.add_argument("--max-sample-times", default=768, type=int, help="The maximum number of Transformer generation function calls." "Program stops when this number is reached (default to be 512 * 1.5 = 768).")
    parser.add_argument("--ucb-constant", default=4., type=float)
    parser.add_argument("--ucb-base", default=10., type=float)
    parser.add_argument("--uct-alg", default="var_p_uct", choices=["uct", "p_uct", "var_p_uct"],
                        help="The UCT algorithm to use."
                             "`uct` is the original UCT algorithm,"
                             "`p_uct` is the UCT algorithm with PUCT,"
                             "and `var_p_uct` is the UCT algorithm with variable PUCT.")
    parser.add_argument('--top-k-cache-steps', type=int, default=1024, help="Number of forward steps to cache top k caches, default 1024 means the whole horizon.")
    parser.add_argument("--public-cases-type", type=str, default='half', help="Number of public test cases to use for evaluation.")
    parser.add_argument("--ts-mode", default="best", choices=["best", "sample"], help="Tree search mode within the evaluation step. `best` uses beam search, `sample` uses sampling.")
    parser.add_argument('--mctsvalue', type=str, default='test', choices=['test', 'gpteval', 'gptevalTC', 'verbalMemory'], help='The value function to use for MCTS.')
    parser.add_argument("--rollout", default=default_rollout, type=int, help="The maximum number of rollouts for PG-TD.")

    ## Device
    parser.add_argument("--gpu", type=int, default="3", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")

    ## To avoid redundant generation
    parser.add_argument('--rerun', action='store_true', default=False, help="If True, rerun if the output file already exists.")
    parser.add_argument('--debug', action='store_true', default=False, help="If True, rerun if the output file already exists.")

    args = parser.parse_args()



    return args