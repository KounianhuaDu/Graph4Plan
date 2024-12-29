import numpy as np
from datasets import load_metric
import json
import prettytable as pt
import click
from utils import reformat_steps, reformat_task_links, reformat_task_nodes
from colorama import Fore, init
import argparse

from argument import parse_args


def f1_score(pred, gt):
    # print("pred, gt")
    #
    # print("pred:", pred)
    # print("gt:", gt)

    if len(pred) == 0 or len(gt) == 0:
        return 0

    intersect = set(pred) & set(gt)
    precision = len(intersect) / len(pred)
    recall = len(intersect) / len(gt)
    f = 2 * precision * recall / (precision + recall + 1e-9)
    # print(f)
    return f


def batch_f1_score(pred_list, gt_list):
    f1_score_list = [f1_score(pred, gt) for pred, gt in zip(pred_list, gt_list)]
    return round(np.mean(np.array(f1_score_list)), 4)


def node_hallucination_rate(solution, valid_tools, data_id=None):
    if len(solution) == 0:
        return [0.0, 0.0]

    hall_list = [1.0 if node not in valid_tools else 0.0 for node in solution]
    micro_hall = sum(hall_list) / len(solution)
    macro_hall = 1.0 if sum(hall_list) >= 1 else 0.0

    return [micro_hall, macro_hall]


def batch_node_hallucination(solutions, valid_tools, print_ids=None):
    if print_ids:
        hall_scores = [node_hallucination_rate(sol, valid_tools, id) for sol, id in zip(solutions, print_ids)]
    else:
        hall_scores = [node_hallucination_rate(sol, valid_tools) for sol in solutions]
    avg_score = np.round(np.mean(np.array(hall_scores), axis=0), 4)
    # avg_score[0] - micro_hallucination
    # avg_score[1] - macro_hallucination
    return avg_score


# def prediction_loader(filename, content_type):
#     return_data = {}
#
#     with open(filename, 'r') as readfile:
#         # print("readfile:", readfile)
#         for line in readfile:
#             # line = line.strip()  # 移除空格和换行符
#             # print(Fore.RED + "line:" + line)
#             if not line:
#                 continue  # 跳过空行
#
#             try:
#                 # 使用 json.loads 解析每一行数据
#                 data = json.loads(line)
#                 data_id = data.get("id")
#
#                 if content_type == 'id':
#                     # 返回数据的 ID
#                     retrieve_data = data_id
#                 elif content_type == "steps":
#                     # 处理步骤
#                     steps = data.get("task_steps", [])
#                     retrieve_data = ", ".join(steps)
#                 elif content_type == "graph":
#                     # 处理图的节点和链接
#                     nodes = data.get("task_nodes", [])
#                     links = data.get("task_links", [])
#                     retrieve_data = {"nodes": nodes, "links": links}
#                 elif content_type == "efficiency":
#                     # 处理任务的效率信息
#                     retrieve_data = {
#                         "cost_time": data.get("cost_time", 0),
#                         "llm_query_times": data.get("llm_query_times", 1)
#                     }
#
#                 # 将处理后的数据存入字典
#                 return_data[data_id] = retrieve_data
#
#             except json.JSONDecodeError as e:
#                 # print(f"Error parsing line: {line}\nError: {e}")
#                 continue  # 跳过解析错误的行，继续处理后续数据
#
#     return return_data


def prediction_loader(filename, content_type):
    readfile = open(filename, 'r')



    return_data = {}

    # count = 0

    for line in readfile:
        # count += 1
        data = json.loads(line)

        data_id = data["id"]
        if content_type == "reward":
            retrieve_data = round(data["test_reward"], 2)
        if "final_program" in data.keys():
            data = data["final_program"]
        if content_type == 'id':
            retrieve_data = data_id
        elif content_type == "steps":
            steps = reformat_steps(data)
            retrieve_data = ", ".join(steps)

        elif content_type == "graph":
            nodes, links = reformat_task_nodes(data), reformat_task_links(data)
            retrieve_data = {"nodes": nodes, "links": links}

        elif content_type == "efficiency":
            retrieve_data = {
                "cost_time": data.get("cost_time", 0),
                "llm_query_times": data.get("llm_query_times", 1)
            }
        

        return_data[data_id] = retrieve_data
    # print("lenth of count:", count)

    return return_data


def evaluate(args, dataset, llm_name, method, metrics=["graph"], modes=["chain"], compute_all=True,
             remove_non_pred=False):

    gt_dir = f"data/{dataset}"
    alignment = json.load(open(f"{gt_dir}/test_ids.json", 'r'))["test_ids"]

    prediction_dir = f"data/predictions/{args.dataset}/{args.arch}/{args.model}"
    if args.provide_graph:
        prediction_dir = f"data/predictions/{args.dataset}-graph/{args.arch}/{args.model}"
    gt_filename = f"{gt_dir}/data.json"
    pred_filename = f"{prediction_dir}/output_dict.json"

    table = pt.PrettyTable()
    if "step" in metrics:
        table.field_names = ['Dataset', 'LLM', 'Mode', 'Step-R1', 'Step-R2', 'NF', 'LF', 'NH-1', 'NH-2', 'LH-1', 'LH-2']
    else:
        table.field_names = ['Dataset', 'LLM', 'Mode', 'NF', 'LF', 'NH-1', 'NH-2', 'LH-1', 'LH-2']

    gt_tool_nodes = json.load(open(f"{gt_dir}/tool_desc.json", 'r'))["nodes"]
    gt_tool_links = json.load(open(f"{gt_dir}/graph_desc.json", 'r'))["links"]
    gt_tool_nodes = [tool["id"] for tool in gt_tool_nodes]
    gt_tool_links = [", ".join([link["source"], link["target"]]) for link in gt_tool_links]
    # print("This is gt_graph_dict:")
    gt_graph_dict = prediction_loader(gt_filename, content_type="graph")
    print("This is gt_graph_dict:",len(gt_graph_dict))
    pred_graph_dict = prediction_loader(pred_filename, content_type="graph")
    print("This is pred_graph_dict:",len(pred_graph_dict))
    # print(Fore.RED+"pred_graph_dict："+str(pred_graph_dict))
    pred_align = prediction_loader(pred_filename, "id")
    print("This is pred_align:",len(pred_align))
    # print(f"{pred_filename} # Valid Predictions {len(pred_align)}")

    # pred_rewards = prediction_loader(pred_filename, "reward")
    # print(pred_rewards)


    for mode in modes:
        alignment_ids = alignment[mode]
        if remove_non_pred:
            alignment_ids = [data_id for data_id in alignment[mode] if data_id in pred_align]

        if not len(alignment_ids):
            continue

        metrics_dict = {}

        if "step" in metrics:
            # step metrics: ['roug1', 'rouge2']
            gt_steps_dict = prediction_loader(gt_filename, content_type="steps")
            pred_steps_dict = prediction_loader(pred_filename, content_type="steps")

            pred_content = [pred_steps_dict.get(data_id, "...") for data_id in alignment_ids]
            gt_content = [gt_steps_dict[data_id] for data_id in alignment_ids]

            rouge = load_metric("rouge")
            rouge_scores = rouge.compute(predictions=pred_content, references=gt_content, use_aggregator=True)

            for key in ["rouge1", "rouge2"]:
                metrics_dict[f"step_{key}"] = round(rouge_scores[key].mid.fmeasure, 4)

        if "graph" in metrics:
            pred_graphs = [pred_graph_dict.get(data_id, {"nodes": [], "links": []}) for data_id in alignment_ids]
            gt_graphs = [gt_graph_dict[data_id] for data_id in alignment_ids]

            node_f1 = batch_f1_score([pred_g["nodes"] for pred_g in pred_graphs], [gt_g["nodes"] for gt_g in gt_graphs])
            link_f1 = batch_f1_score([pred_g["links"] for pred_g in pred_graphs],
                                     [gt_g["links"] for gt_g in gt_graphs]) if mode != "single" else 'N/A'

            node_hr = batch_node_hallucination([pred_g["nodes"] for pred_g in pred_graphs], gt_tool_nodes,
                                               alignment_ids)
            link_hr = batch_node_hallucination([pred_g["links"] for pred_g in pred_graphs], gt_tool_links)
            score_dict = dict()

            if 'step' not in metrics:
                table.add_row([dataset, llm_name, mode, node_f1, link_f1, node_hr[0], node_hr[1], link_hr[0], link_hr[1]])
            else:
                table.add_row([dataset, llm_name, mode, metrics_dict['step_rouge1'], metrics_dict['step_rouge2'], node_f1,
                     link_f1, node_hr[0], node_hr[1], link_hr[0], link_hr[1]])

    print(table)


def main():
    default_rollout = 3
    default_out_path = "./train_trace"
    problem_indices = range(10)
    # Set up the argument parser
    args = parse_args(default_out_path=default_out_path, default_rollout=default_rollout)

    # Call the evaluate function with the parsed arguments
    evaluate(args, args.dataset, args.arch, args.model)


if __name__ == "__main__":
    main()
