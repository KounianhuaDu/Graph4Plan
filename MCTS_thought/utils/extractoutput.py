import json
import re
import time
from colorama import Fore, Back, Style, init


def extraction_output_from_llm(res):
    try:
        # Using regex to extract the first JSON string after '# RESULT #:'
        pattern = r'\{"task_steps": \[.*?\], "task_nodes": \[.*?\], "task_links": \[.*?\]\}'
        # Extract the required part
        result_match = re.findall(pattern, res)
        # print(Fore.RED + "result_match:", result_match)
        fixed_json_output = result_match[0] if result_match else None
        # print(fixed_json_output)
        # print(Fore.GREEN + "first_extracted_result:\n", fixed_json_output)

        if fixed_json_output:
            try:
                # Load the JSON object
                content = json.loads(fixed_json_output)
                print(content)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                content = {"task_steps": [], "task_nodes": [], "task_links": []}
        else:
            print("No valid result found.")
            content = {"task_steps": [], "task_nodes": [], "task_links": []}

        # content = json.loads(first_extracted_result)

        print("===== ===== success ===== ===== ")
    except json.JSONDecodeError as e:
        print(Fore.RED + '===== ===== =====')
        print(Fore.BLUE + '===== ===== =====')
        print(Fore.GREEN + '===== ===== =====')
        print(f"Failed to decode JSON for input id:" + input["id"])
        print(Fore.GREEN + '===== ===== =====')
        print(Fore.BLUE + '===== ===== =====')
        print(Fore.RED + '===== ===== =====')
        content = {"task_steps": [], "task_nodes": [], "task_links": []}

    return content

