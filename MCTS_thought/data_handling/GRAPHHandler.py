import os
import json
from colorama import Fore, init
import ast

# 封装从文件加载测试/训练集的函数


class GRAPHHander:
    def __init__(self, args):
        self.llm = args.arch
        self.data_dir = args.dataset
        self.model = args.model
        if args.provide_graph:
            self.prediction_dir = f"data/predictions/{args.dataset}-graph/{args.arch}/{args.model}"
        else:
            self.prediction_dir = f"data/predictions/{args.dataset}/{args.arch}/{args.model}"
        self.rerun = args.rerun
        self.provide_graph = args.provide_graph

    def get_data(self):
        # llm = args.arch
        use_demos = 1  # range 0-3 the number of  example in prompt
        
        dependency_type = "temporal" if self.data_dir == 'dailylife' else "resource"

        # Load the model and tokenizer locally

        wf_name = f"{self.prediction_dir}/has_inferenced.json"
        wf_name1 = f"{self.prediction_dir}/output_dict.json"
        wf_name2 = f"{self.prediction_dir}/trace_feats.json"

        if not os.path.exists(self.prediction_dir):
            os.makedirs(self.prediction_dir, exist_ok=True)

        if self.rerun:
            if os.path.exists(wf_name):
                os.remove(wf_name)
            if os.path.exists(wf_name1):
                os.remove(wf_name1)
            if os.path.exists(wf_name2):
                os.remove(wf_name2)

        has_inferenced = []
        if os.path.exists(wf_name):
            with open(wf_name, "r") as rf:
                for line in rf:
                    data = json.loads(line)

                    has_inferenced.append(data['id'])
        print(Fore.RED + "len has_inferenced:", len(has_inferenced))

        with open(f"data/{self.data_dir}/test_data.json", "r") as rf_ur:
            # inputs = [json.loads(line) for line in rf_ur if json.loads(line)["id"] not in has_inferenced]
            inputs = []
            input_ids = []
            for line in rf_ur:
                input = json.loads(line)
                if input["id"] not in has_inferenced:
                    input_ids.append(input["id"])
                    input = {"id": input["id"], "user_request": input["user_request"]}
                    inputs.append(input)



        # with open(wf_name, "a") as wf, open(wf_name1, "a") as wf1, open(wf_name2, "a") as wf2:
        tool_list = json.load(open(f"data/{self.data_dir}/tool_desc.json", "r"))["nodes"]
        if "input-type" not in tool_list[0]:
            assert dependency_type == "temporal", "Tool type is not ignored, but the tool list does not contain input-type and output-type"
        if dependency_type == "temporal":
            for tool in tool_list:
                tool["parameters"] = [param["name"] for param in tool["parameters"]]
        # demos = []
        demo_string = ""
        if use_demos:
            demos_id_list = {
                "huggingface": ["10523150", "14611002", "22067492"],
                "multimedia": ["30934207", "20566230", "19003517"],
                "dailylife": ["27267145", "91005535", "38563456"],
                "tmdb": [1]
            }
            demos_id = demos_id_list[self.data_dir][:use_demos]
            # demos_rf = open(f"../data/{data_dir}/data.json", "r")
            demos = []
            with open(f"data/{self.data_dir}/data.json", "r") as demos_rf:
                for line in demos_rf:
                    data = json.loads(line)
                    if data["id"] in demos_id:
                        demo = {
                            "user_request": data["user_request"],
                            "result": {
                                "task_steps": data["task_steps"],
                                "task_nodes": data["task_nodes"],
                                "task_links": data["task_links"]
                            }
                        }
                        demos.append(demo)
                demos_rf.close()
                if len(demos) > 0:
                    demo_string += "\nHere are provided examples for your reference.\n"
                    for demo in demos:
                        demo_string += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: ```json\n{json.dumps(demo["result"])}\n```"""
                    # print(Fore.RED+"demo string"+demo_string)
        tool_string = "# TASK LIST #:\n"
        for tool in tool_list:
            tool_string += json.dumps(tool) + "\n"
        if len(inputs) == 0:
            print("All Completed!")
            return
        else:
            print(f"Detected {len(has_inferenced)} tasks have already been processed.")
            print(f"Starting to process {len(inputs)} tasks...")

        graph = None
        # if self.provide_graph:
        #     graph = dict()
        #     with open(f"data/{self.data_dir}/graph_desc.json", "r") as graph_rf:
        #         links = json.load(graph_rf)["links"]
        #         for link in links:
        #             source = link["source"]
        #             if source in graph.keys():
        #                 graph[source].append(link["target"])
        #             else:
        #                 graph[source] = [link["target"]]
        # Run inference synchronously
        return inputs, tool_string, demo_string, graph

