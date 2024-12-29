import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from colorama import Fore, Back, Style, init
import re
import time




def main():
    gpu = 4
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(gpu)
    else:
        device = 'cpu'


    # Hardcoded values for the options
    data_dir = "multimedia"  #huggingface
    temperature = 0.2
    # top_p = 0.1
    # model_path = "/ext0/hcchai/codemate/llama2/Llama-2-7b-hf"
    model_path = "/ext0/hcchai/codemate/llama3/Meta-Llama-3-8B"

    print('model_path:',model_path)
    # multiworker = 1
    llm = "llama"
    use_demos = 3

    dependency_type = "resource"
    log_first_detail = False

    # Load the model and tokenizer locally
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Move the model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup directories and files
    prediction_dir = f"data/{data_dir}/predictions/"
    wf_name = f"{prediction_dir}/{llm}.json"

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    has_inferenced = []
    if os.path.exists(wf_name):
        with open(wf_name, "r") as rf:
            for line in rf:
                data = json.loads(line)
                has_inferenced.append(data["id"])

    with open(f"data/{data_dir}/user_requests.json", "r") as rf_ur:
        inputs = [json.loads(line) for line in rf_ur if json.loads(line)["id"] not in has_inferenced]

    with open(wf_name, "a") as wf:
        tool_list = json.load(open(f"data/{data_dir}/tool_desc.json", "r"))["nodes"]
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
            demos_id = demos_id_list[data_dir][:use_demos]

            # demos_rf = open(f"../data/{data_dir}/data.json", "r")
            demos = []


            with open(f"data/{data_dir}/data.json", "r") as demos_rf:
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
                        demo_string += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: {json.dumps(demo["result"])}"""
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

        # Run inference synchronously
        if log_first_detail:
            inference(inputs[0], model, tokenizer, temperature, tool_string, wf, demo_string, dependency_type, log_detail=True)
            inputs = inputs[1:]

        for input in inputs:
            print(Fore.CYAN+"input:\n",input)
            inference(input, model, tokenizer, temperature, tool_string, wf, demo_string, dependency_type)





def inference(input, model, tokenizer, temperature, tool_string, write_file, demo_string, dependency_type, log_detail=False):
    st_time = time.time()

    user_request = input["user_request"]
    prompt  = """Note: do not include anything related to the prompt in your output. Just refer to the demo and provide the final result. Especially avoid redundant and repetitive content in the output.\n"""

    if dependency_type == "resource":
        prompt += """\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TASK LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}], "task_links": [{"source": "task name i", "target": "task name j"}]} """
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. the dependencies among task steps should align with the argument dependencies of the task nodes; \n4. the tool arguments should be align with the input-type field of # TASK LIST #;\n5.do not include anything related to the prompt in your output. Just refer to the demo and provide the final result. Especially avoid redundant and repetitive content in the output."""
    else:
        prompt += """\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}"""
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;\n4. do not include anything related to the prompt in your output. Just refer to the demo and provide the final result. Especially avoid redundant and repetitive content in the output."""

    # print(Fore.YELLOW + "prompt:\n", prompt)
    # print(Fore.GREEN + "tool_string:\n", tool_string)

    prompt += demo_string

    prompt += """\n\n# USER REQUEST #: {{user_request}}\nnow please generate your result in a strict JSON format:\n# RESULT #:"""

    final_prompt = tool_string + prompt.replace("{{user_request}}", user_request)

    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    # Perform inference using the local model
    output = model.generate(**inputs, max_new_tokens=1024, do_sample=True,temperature=temperature,pad_token_id=model.config.eos_token_id)

    # Decode and process the result
    result = tokenizer.decode(output[0], skip_special_tokens=True)



    try:
        # Using regex to extract the first JSON string after '# RESULT #:'
        result_match = re.search(r'now please generate your result in a strict JSON format:\n# RESULT #: (\{.*?\}\s*\]\})', result)

        # input_string = '{"task_steps": ["Step 1: Classify the image to determine its category", "Step 2: Classify the table to determine its content"], "task_nodes": [{"task": "Tabular Classification", "arguments": ["example.jpg"]}'

        # fixed_json_output = auto_fix_json(result_match)
        # print(fixed_json_output)
        # print(Fore.BLUE+ "result_match:\n",result_match)
        # Extract the first result
        first_extracted_result = result_match.group(1) if result_match else None

        fixed_json_output = first_extracted_result #auto_fix_json(first_extracted_result)
        # print(fixed_json_output)
        # print(Fore.GREEN + "first_extracted_result:\n", fixed_json_output)

        if fixed_json_output:
            try:
                # Load the JSON object
                content = json.loads(fixed_json_output)
                print(content)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                content = {"task_steps": [], "task_nodes": [],"task_links":[]}
        else:
            print("No valid result found.")
            content = {"task_steps": [], "task_nodes": [],"task_links":[]}

        # content = json.loads(first_extracted_result)

        print("===== ===== success ===== ===== ")
    except json.JSONDecodeError as e:
        print(Fore.RED + '===== ===== =====' )
        print(Fore.BLUE + '===== ===== =====' )
        print(Fore.GREEN + '===== ===== =====' )
        print(f"Failed to decode JSON for input id:"+input["id"])
        print(Fore.GREEN + '===== ===== =====' )
        print(Fore.BLUE + '===== ===== =====' )
        print(Fore.RED + '===== ===== =====' )
        return

    res = {"id": input["id"], "user_request": input["user_request"]}

    if len(content.get("task_steps")):
        res["task_steps"] = content.get("task_steps")
    else:
        res["task_steps"] = ''

    if len(content.get("task_nodes")):
        res["task_nodes"] = content.get("task_nodes")
    else:
        res["task_nodes"] = ''

    if len(content.get("task_links")):
        res["task_links"] = content.get("task_links")
    else:
        res["task_links"] = ''


    res["cost_time"] = round(time.time() - st_time, 4)

    print(Fore.RED + 'res["task_steps"]:' + str(res["task_steps"] ))
    print(Fore.RED + 'res["task_nodes"]:' + str(res["task_nodes"] ))
    print(Fore.RED + 'res["task_links"]:' + str(res["task_links"]))
    print(Fore.RED + 'res["cost_time"]:' + str(res["cost_time"]))



    write_file.write(json.dumps(res) + "\n")
    write_file.flush()


if __name__ == "__main__":
    main()







