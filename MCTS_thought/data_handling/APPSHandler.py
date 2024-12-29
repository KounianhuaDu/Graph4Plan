from utils import *
import os
import json

def get_test_cases(prob_dir, public_cases_type):
    with open(os.path.join(prob_dir, "input_output.json")) as f:
        in_outs = json.load(f)
        in_out_len = len(in_outs['inputs'])

        train_in_outs, test_in_outs = {}, {}
        if public_cases_type == 'half':
            # split evenly by default
            public_test_cases = in_out_len // 2
        elif public_cases_type.isdigit():
            public_test_cases = int(public_cases_type)
        else:
            raise Exception(f"Can't understand public_test_cases {public_cases_type}")
        private_test_cases = in_out_len - public_test_cases

        if public_test_cases < 1 or private_test_cases < 1:
            print(f"Not enough test cases: {public_test_cases}, {private_test_cases}.")
            return None, None

        train_in_outs['inputs'] = in_outs['inputs'][:public_test_cases]
        train_in_outs['outputs'] = in_outs['outputs'][:public_test_cases]
        test_in_outs['inputs'] = in_outs['inputs'][public_test_cases:]
        test_in_outs['outputs'] = in_outs['outputs'][public_test_cases:]
        
    return train_in_outs, test_in_outs


class APPSHandler:
    def __init__(self, data_path, mode, problem_indices, difficulty='introductory', public_cases_type = 'half'):
        self.problems = []
        for idx in problem_indices:
            prob_dir = os.path.join(data_path, f"APPS/{mode}/{idx:04d}")
            pro_metadata_path = os.path.join(prob_dir, "metadata.json")
            test_case_path = os.path.join(prob_dir, "input_output.json")
            prompt_path = os.path.join(prob_dir, "question.txt")

            # difficulty filtering
            if difficulty is not None:
                with open(pro_metadata_path) as f:
                    pro_metadata = json.load(f)
                    if pro_metadata['difficulty'] != difficulty:
                        continue

            # problem instance formation
            problem_instance = {}
            problem_instance['index'] = idx
            input_prompt = "\nQUESTION:\n"
            with open(prompt_path, "r") as f:
                data = f.readlines()
                data = "".join(data)
            input_prompt += data

            with open(test_case_path, "r") as f:
                data = json.load(f)

            if not data.get("fn_name"):
                input_prompt += "\nUse Standard Input format"  # \n"
                problem_instance['code_type'] = "standard_input"
                problem_instance['method_name'] = None
            else:
                input_prompt += "\nUse Call-Based format"  # \n"
                problem_instance['code_type'] = "call_based"
                problem_instance['method_name'] = data.get("fn_name")
            
            input_prompt += "\nANSWER:\n"
            problem_instance["prompt"] = input_prompt

            # test cases for train and test
            train_in_outs, test_in_outs = get_test_cases(prob_dir, public_cases_type)
            if not train_in_outs:
                continue
            
            problem_instance["train_in_outs"] = train_in_outs
            problem_instance["test_in_outs"] = test_in_outs
            self.problems.append(problem_instance)
