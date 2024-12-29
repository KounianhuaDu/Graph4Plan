def build_question_instruct(problem, tool_string, demo_string, dataset):
    if dataset == 'dailylife':
        rationale_instruct = """
        \n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. 
        The format must in a strict JSON format, like: {"task_steps": [one or more concrete steps, format as Step x: step description], "task_nodes": [{"task": "tool name must be from # TASK LIST #", "arguments": [{"name": name of the argument for the tool, "value": content of the argument}]}], "task_links": [{"source": "task name i", "target": "task name j"}]} 
        """
        rationale_instruct += """
        # REQUIREMENTS #: 
        1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #;\n
        2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes;\n
        3. the dependencies among task steps should align with the argument dependencies of the task nodes;\n
        4. the name of tool arguments should be align with the name field of # TASK LIST #;\n
        5. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;\n
        """
    else:
        rationale_instruct = """
        \n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. 
        The format must in a strict JSON format, like: {"task_steps": [one or more concrete steps, format as Step x: step description], "task_nodes": [{"task": "tool name must be from # TASK LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}], "task_links": [{"source": "task name i", "target": "task name j"}]} 
        """
        rationale_instruct += """
        # REQUIREMENTS #: 
        1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #;\n
        2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes;\n
        3. the dependencies among task steps should align with the argument dependencies of the task nodes;\n
        4. the tool arguments should be align with the input-type field of # TASK LIST #;\n
        5. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;\n
        """
    rationale_instruct += demo_string

    rationale_instruct += """
    \n\n
    # USER REQUEST #: {{user_request}}
    now please generate your result in a strict JSON format:
    # RESULT #:"""

    rationale_instruct = tool_string + rationale_instruct.replace("{{user_request}}", problem)

    return rationale_instruct


def build_rationale_instruct(problem):
    rationale_instruct = """
    You are tasked with breaking down a complex user request into solvable sub-tasks by creating a task plan.\n
    Problem Description:\n
    {}\n""".format(problem)

    rationale_instruct += """
    Below is an example of step-by-step tool calls for planning a task to decompose a complex request into sub-tasks:\n
    ```json
    [
    {"Tool Call of Step 1": "Call Text-to-Image tool with input: 'Describe the image content in text for the blog post, including the Eiffel Tower, sky, and clouds' and output: 'thumbnail image'"}
    {"Tool Call of Step 2": "Call Image Editing tool with input: 'thumbnail image' and output: 'edited thumbnail image'"}
    {"Tool Call of Step 3": "Finish"}
    ]
    ```
    Based on this example, generate step-by-step tool calls to solve the given problem by breaking it down into sub-tasks and forming a connected path.

    -----Tool Calls-----
    """
    # rationale_instruct += """
    # Below is an example of step-by-step tool calls for planning a task to decompose a complex request into sub-tasks:\n
    # ```json
    # [
    # {"Step 1": "Call xxx tool with input: 'xxx' and output: 'xxx'"}
    # {"Step 2": "Call xxx tool with input: 'xxx' and output: 'xxx'"}
    # {"Step 3": "Finish"}
    # ]
    # ```
    # Based on this example, generate step-by-step tool calls to solve the given problem by breaking it down into sub-tasks and forming a connected path.

    # -----Tool Calls-----
    # """
    return rationale_instruct


def build_intermediate_instruct(h, k, next_possible_tools=None):
    ins = "\n-----Instruction-----\n"
    if h == 0:
        ins += 'Now, please generate {} different tool calls for the first sub-task (Step 1).\n'.format(k)
    else:
        ins += 'Now that we have generated tool calls for the previous sub-tasks, follow the dependencies and generate {} different tool calls for the next sub-task (Step {}).\n'.format(
            k, h + 1)

    ins += """
    Please wrap your response into a JSON object that contains keys `i Tool Call of Step {}` with i as the number of your call, and key `Reasonableness` with the Reasonableness score of each tool call. The value of 
    key `i Tool Call of Step {}` is format as  "Call xxx tool with input: 'xxx' and output: 'xxx'". ONLY output the json content without any additional explanatory text.\n
    """.format(h + 1, h + 1)
    # Call xxx tool with input: 'name and content of input' and output: 'name and content of output'
    # print(ins)
    if next_possible_tools:
        ins += f"Your next tool should be selected from the following list: \"{', '.join(next_possible_tools)}\"\n"
    ins += """An example is given below. You should strictly follow the form of the example.\n
    ```json
    [
    {"1 Tool Call of Step x": "Call Text-to-Image tool with input: 'Describe the image content in text for the blog post, including the Eiffel Tower, sky, and clouds' and output: 'thumbnail image'", "Reasonableness": 0.9},
    {"2 Tool Call of Step x": "Call Image Editing tool with input: 'thumbnail image' and output: 'edited thumbnail image'", "Reasonableness": 0.85},
    {"3 Tool Call of Step x": "Finish", "Reasonableness": 0.7}
    ]
    ```\n
    If you think the previous tool calls are enough to solve the task, you can answer with "Finish".
    """
    # ins += """An example is given below. You should strictly follow the form of the example.\n
    # ```json
    # [
    # {"1 Tool Call of Step x": "Call xxx tool with input: 'xxx' and output: 'xxx'", "Reasonableness": 0.9},
    # {"2 Tool Call of Step x": "Call xxx tool with input: 'xxx' and output: 'xxx'", "Reasonableness": 0.85},
    # {"3 Tool Call of Step x": "Finish", "Reasonableness": 0.7}
    # ]
    # ```\n
    # If you think the previous tool calls are enough to solve the task, you can answer with "Finish".
    # """

    return ins


def get_reward_instruct(previous_tool_calls, problem):
    instruction = """
    You are tasked with breaking down a complex user request into sub-tasks and selecting the best sequence of steps to fulfill the original request.\n
    Problem Description:\n
    {}\n
    -----Tool Calls-----\n
    The following tool calls were generated as steps for solving the problem:\n
    {}\n
    -----Instruction-----\n
    Using the tool calls provided, plan a connected path through the sub-tasks. Each sub-task represents a node in a graph, and the edges represent the dependencies between them. Your objective is to select a valid, connected path that solves the problem efficiently. Please return the sub-task sequence in JSON format, ensuring that all dependencies are respected.
    """.format(problem, previous_tool_calls)
    return instruction