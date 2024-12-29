import os
import sys

sys.path.append("..")

import time
from math import sqrt, log
import re
from collections import defaultdict, deque
import numpy as np
from transformers import LlamaTokenizer, GemmaTokenizer, AutoTokenizer
import tiktoken

# from executors import AppsExecutor
from ChatModels import *
from models import *
from utils.instruction import *
from colorama import Fore, Back, Style, init
import json


def uct_tree_policy(mcts_agent, children):
    return max(children, key=mcts_agent.ucb)


def p_uct_tree_policy(mcts_agent, children):
    return max(children, key=mcts_agent.p_ucb)


def var_p_uct_tree_policy(mcts_agent, children):
    scores = [mcts_agent.var_p_ucb(child) for child in children]
    print(scores)
    return max(children, key=mcts_agent.var_p_ucb)


def f1_score(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return 0

    intersect = set(pred) & set(gt)
    precision = len(intersect) / len(pred)
    recall = len(intersect) / len(gt)
    f = 2 * precision * recall / (precision + recall + 1e-9)
    return f

def f1_score_with_order(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return 0

    # 确定较小的列表长度，避免超出索引范围
    min_len = min(len(pred), len(gt))

    # 计算有多少元素在相同的索引位置上是相等的
    matches = sum(1 for p, g in zip(pred.values(), gt.values()) if p == g and len(p) > 0 and len(g) > 0)

    # 计算精确率和召回率
    precision = matches / len(pred)
    recall = matches / len(gt)
    print(matches, len(pred), len(gt))
    # 计算F1分数
    f = 2 * precision * recall / (precision + recall + 1e-9)
    return f
    # pred_node = [node['task'] for node in pred['task_nodes']]
    # gt_node = [node['task'] for node in gt['task_nodes']]
    # node_f1 = f1_score(pred_node, gt_node)
    # pred_link = [", ".join([link['source'], link['target']]) for link in pred['task_links']]
    # gt_link = [", ".join([link['source'], link['target']]) for link in gt['task_links']]
    # link_f1 = f1_score(pred_link, gt_link)

    # return (node_f1 + link_f1) / 2


class MCTSStep:
    def __init__(self, args):
        self.args = args
        self.sample_nums = 0
        self.gamma = 0.9

        ## Backbone LLM
        if args.arch == 'gpt4omini':
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.generator = GPTStepChat(args.arch, self.tokenizer, args)
        elif 'llama' in args.arch:
            self.tokenizer = AutoTokenizer.from_pretrained(args.modelpath, use_fast=False, padding_side='left')
            self.generator = LlamaChat(args.arch, self.tokenizer, args)
        elif 'gemma' in args.arch: 
            self.tokenizer = GemmaTokenizer.from_pretrained(args.modelpath)
            self.generator = GemmaChat(args.arch, self.tokenizer, args)
        elif 'deepseek' in args.arch:
            # self.tokenizer = None
            self.generator = DeepSeekChat(args.arch, None, args)
            self.tokenizer = self.generator.tokenizer
        elif 'yi' in args.arch:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.generator = YiChat(args.arch, self.tokenizer, args)
        elif 'deepcoder' in args.arch:
            self.tokenizer = AutoTokenizer.from_pretrained(args.modelpath, trust_remote_code=True)
            self.generator = DeepSeekCoderChat(args.arch, self.tokenizer, args)
        else:
            raise NotImplementedError

        # ## Dataset
        # if args.dataset == 'APPS':
        #     self.executor = AppsExecutor(args)
        # else:
        #     raise NotImplementedError

        ## MCTS related
        self.term_cond = lambda: self.sample_nums > args.max_sample_times
        if args.uct_alg == 'uct':
            self.node_choose_policy = uct_tree_policy
        elif args.uct_alg == 'p_uct':
            self.node_choose_policy = p_uct_tree_policy
        elif args.uct_alg == 'var_p_uct':
            self.node_choose_policy = var_p_uct_tree_policy
            self.ucb_base = args.ucb_base

        self.root = None
        self.cached_reward = {}
        self.cached_value = {}
        self.cached_verbal_feedback = {}
        self.cur_prob_instance = None
        self.sample_times = []
        self.st = time.time()
        self.dataset = args.dataset
        self.dependency_prior = None

    def generate(self, problem_instance, tool_string, demo_string, dependency_prior=None):
        self.st = time.time()
        self.cur_prob_instance = problem_instance
        self.dependency_prior = dependency_prior
        # test ===== =====

        current_path = os.getcwd()

        # 打印当前工作目录
        print("Current working directory:", current_path)

        raw_prompt = problem_instance["user_request"]

        raw_prompt = build_question_instruct(raw_prompt, tool_string, demo_string, self.args.dataset)

        initial_state = build_rationale_instruct(raw_prompt)

        initial_state = self.tokenizer.encode(initial_state)

        print('The lenth of initial_state:', len(initial_state))

        if len(initial_state) >= self.args.horizon:
            return None
        done = False

        self.mcts_procedure(initial_state, problem_instance, done)

        node = self.root
        decision_to_chance = []
        chance_to_decision = []
        chance_id = -1
        chance_features = defaultdict(dict)
        decision_features = defaultdict(dict)
        queue = deque([node])
        while queue:
            node = queue.popleft()
            if type(node) == MineDecisionNode:
                decision_features[node.id]['state'] = self.tokenizer.decode(node.state)
                decision_features[node.id]['value'] = node.value

                '''for key in self.cached_value.keys():
                    print(self.tokenizer.decode(key))
                print(self.tokenizer.decode(tuple(node.state)))
                exit()'''

                # decision_features[node.id]['reward'] = self.cached_reward[tuple(node.state)]
                # decision_features[node.id]['verbal_feedback'] = self.cached_verbal_feedback[tuple(node.state)]

            for child in node.children:
                queue.append(child)
                if type(child) == MineChanceNode:
                    if child.action == self.generator.terminal_token:
                        continue
                    chance_id += 1
                    decision_to_chance.append((node.id, chance_id))
                    chance_features[chance_id]['ucb'] = self.ucb(child)
                    chance_features[chance_id]['p_ucb'] = self.p_ucb(child)
                    chance_features[chance_id]['var_p_ucb'] = self.var_p_ucb(child)
                    chance_features[chance_id]['action'] = self.tokenizer.decode(child.action)
                    chance_features[chance_id]['prob'] = child.prob
                    if len(child.children) > 0:
                        next_decision = child.children[0]
                        chance_to_decision.append((chance_id, next_decision.id))

        if len(self.cached_value) == 0:
            state = self.generator.get_rationale_predicted_sequence(initial_state, problem_instance['user_request'])
            complete_prog_score = self.get_reward(state)

        complete_programs_ids = list(map(lambda x: list(x), self.cached_value.keys()))
        if complete_programs_ids is None or len(complete_programs_ids) == 0:
            return None
        complete_programs = [self.convert_state_to_program(s) for s in complete_programs_ids]

        train_rewards = [self.cached_value[tuple(s)] for s in complete_programs_ids]
        test_rewards = [self.get_reward(s, mode='test') for s in complete_programs_ids]
        best_idx = np.argmax(train_rewards)

        output_dict = {}
        output_dict['final_program'] = complete_programs[best_idx]
        output_dict['train_reward'] = train_rewards[best_idx]
        output_dict['test_reward'] = test_rewards[best_idx]

        output_dict['all_programs'] = complete_programs
        output_dict['all_train_rewards'] = train_rewards
        output_dict['all_test_rewards'] = test_rewards
        output_dict['avg_sample_time'] = np.mean(np.array(self.sample_times))

        self.cached_reward = {}
        self.cached_value = {}
        self.cached_verbal_feedback = {}
        self.generator.clean_cache()
        self.sample_nums = 0

        trace_feats = (decision_to_chance, chance_to_decision, chance_features, decision_features)

        return output_dict, trace_feats

    def mcts_procedure(self, initial_state, problem_instance, done):
        """
        Compute the entire MCTS procedure wrt to the selected tree policy.
        Funciton tree_policy is a function taking an agent + a list of ChanceNodes as argument
        and returning the one chosen by the tree policy.
        """
        graph = None
        if self.args.provide_graph:
            graph = dict()
            with open(f"data/{self.dataset}/graph_desc.json", "r") as graph_rf:
                links = json.load(graph_rf)["links"]
                for link in links:
                    source = link["source"]
                    if source in graph.keys():
                        graph[source].append(link["target"])
                    else:
                        graph[source] = [link["target"]]
        # 开始时，root是None
        decision_node_num = 0
        self.root = MineDecisionNode(None, initial_state, done, generator=self.generator, id=decision_node_num)
        self.root.__expand__()
        decision_node_num += 1

        # 如果rollouts=1，产生的程序存在cached_rewards里面的只有一个完整程序，其实select那一步就已经选了走哪个完整程序了
        print("Performing rollouts.")
        for _ in range(self.args.rollout):  # 这个rollout控制的是选择次数，如果从根节点开始，第一次选第一层，第二次可能选的是第二层，第三次选第三层
            if self.term_cond():
                break
            rewards = []  # Rewards collected along the tree for the current rollout
            node = self.root  # Current node
            terminal = done

            # Selection
            select = True
            while select:
                if (type(node) == MineDecisionNode):  # DecisionNode
                    if node.is_terminal:
                        select = False  # Selected a terminal DecisionNode
                    else:
                        node = self.node_choose_policy(self,
                                                       node.children)  # 根据P-UCB从node的children中选择一个最大值的node， node is now a ChanceNode
                else:  # ChanceNode，（状态，动作）节点，相当于树中的一条边
                    state_p, reward, terminal = self.transition(node.parent.state, node.action)
                    rewards.append(reward)  # 做完动作没有terminal的情况下，reward为0，后面backpropagation主要靠estimation

                    new_state = True  # 如果树有很多层，这里的while循环会从根节点一层一层往下走，直到找到一个新的state_p
                    for i in range(len(node.children)):  # 其实chancenode只有一个child
                        if node.children[i].state == state_p:
                            # Shun: state_p already in the tree, point node to the corresponding Decision Node
                            node = node.children[i]
                            new_state = False
                            break
                    if new_state:  # 一开始如果是三个rollouts，就三个root的children都会经过这里
                        select = False  # Selected a ChanceNode

            # Expansion
            # If node is a decision node, then it must be a terminal node, do nothing here
            if type(node) == MineChanceNode:
                node.children.append(
                    MineDecisionNode(node, state_p, terminal, generator=self.generator, id=decision_node_num, decision_memory=node.chance_memory, 
                                     graph=graph))  # chance node 只有一个子节点，就是加上了那个动作的节点,但每一个decision node在创建的时候都会带有3个可能的动作
                if not self.args.mctsvalue in ['verbalMemory', 'verbalMemoHistory']:
                    node.children[-1].__expand__()
                decision_node_num += 1
                node = node.children[-1]  # 就是新增加的decision node

            # Evaluation
            # now `rewards` collected all rewards in the ChanceNodes above this node
            assert (type(node) == MineDecisionNode)
            state = node.state
            if not node.is_terminal:
                if self.args.mctsvalue == 'test':
                    """
                    pure test reward
                    """
                    program = self.generator.get_rationale_predicted_sequence(state, problem_instance['user_request'])
                    estimate = self.get_reward(program, depth=node.depth)  # 这里的state包含了输入的prompt。在get reward这步会将state cache起来
                    if tuple(program) not in self.cached_value.keys():
                        self.cached_value[tuple(program)] = estimate
                    node.value = estimate
                    self.sample_nums = self.sample_nums + 1
                    # save this information for demo
                    node.info['complete_program'] = program  # decision node的info里面存了这个节点的可能的complete_program
            else:
                # the rewards are defined on terminating actions, the terminal states have no rewards
                print(Fore.BLUE + "Already the terminal.")
                estimate = 0

            # Backpropagation
            node.visits += 1
            node = node.parent
            assert (type(node) == MineChanceNode)
            while node:
                if len(rewards) != 0:
                    estimate = rewards.pop() + self.gamma * estimate
                node.sampled_returns.append(estimate)
                node.parent.visits += 1
                node = node.parent.parent

            # should finish backpropagating all the rewards back
            assert len(rewards) == 0
            self.sample_times.append(time.time() - self.st)
        # root的children是chance node，每个对应于一个动作

    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
        s = s.replace("\n", "")
        s = s.replace("\_", "_")
        s = s.replace("\\", "")
        # print(Fore.BLUE + s)
        # pattern = r'\{"task_steps": \[.*?\], "task_nodes": \[.*?\], "task_links": \[.*?\]\}'
        # Extract the required part
        # result_match = re.findall(pattern, s)
        # print(Fore.RED + "result_match:", result_match)
        # fixed_json_output = result_match[0] if result_match else None

        if True:
            try:
                # Load the JSON object
                content = json.loads(s)
                # print(content)
            except:
                print(s)
                # print(f"JSON decoding error: {e}")
                content = {"task_steps": [], "task_nodes": [], "task_links": []}
        else:
            print("No valid result found.")
            content = {"task_steps": [], "task_nodes": [], "task_links": []}

        return content

    def get_reward(self, s, mode='train', with_verbal=False, depth=None):
        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            if with_verbal:
                return [self.cached_reward[tuple(s)], self.cached_verbal_feedback[tuple(s)]]
            else:
                return self.cached_reward[tuple(s)]

        # 转换成文本
        output_str = self.convert_state_to_program(s)
        # print(Fore.BLUE + output_str)

        with open(f"data/{self.dataset}/data.json", "r") as rf_ur:
            data_line = [json.loads(line) for line in rf_ur if json.loads(line)["id"]]

        target_id = self.cur_prob_instance['id']  # 设置你想要查找的ID
        target_data = None

        for entry in data_line:
            if entry['id'] == target_id:
                target_data = entry
                break
        print("task_steps:", target_data["task_steps"])
        print("task_nodes:", target_data["task_nodes"])
        print("task_links:", target_data["task_links"])
        print("output_str:",output_str)

        gt = {"task_steps": target_data["task_steps"],"task_nodes": target_data["task_nodes"],"task_links": target_data["task_links"]}


        # reward = f1_score(output_str, gt)
        reward = f1_score_with_order(output_str, gt)
        if depth and depth <= min(len(target_data["task_steps"]), len(output_str["task_steps"])):
            output_str = {"task_steps": output_str["task_steps"][:depth],"task_nodes": output_str["task_nodes"][:depth],
                          "task_links": output_str["task_links"][:depth-1]}
            gt = {"task_steps": target_data["task_steps"][:depth],"task_nodes": target_data["task_nodes"][:depth],
                  "task_links": target_data["task_links"][:depth-1]}
            reward += f1_score_with_order(output_str, gt)
            reward /= 2



        # 添加到cached reward
        if mode == 'train':
            self.cached_reward[tuple(s)] = reward
            if with_verbal:
                self.cached_verbal_feedback[tuple(s)] = 0  # verbal_feedbacks

        if with_verbal:
            return [reward, 0]
        else:
            return reward

    def transition(self, s, a):
        print(Fore.RED + "transition")
        print(a)
        
        if self.generator.terminal_token in a or len(s) >= self.args.horizon:
            # either the program finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            next_state = s
            program = self.generator.get_rationale_predicted_sequence(s, self.cur_prob_instance['user_request'])
            reward = self.get_reward(program)
            if tuple(program) not in self.cached_value.keys():
                self.cached_value[tuple(program)] = reward
        else:
            if isinstance(a, list):
                next_state = s + a
            else:
                next_state = s + [a]
            reward = 0  # no intermediate reward
        return next_state, reward, done

    def ucb(self, node):
        """
        Upper Confidence Bound of a chance node
        """
        return chance_node_value(node) + self.args.ucb_constant * sqrt(log(node.parent.visits)) / (
                1 + len(node.sampled_returns))

    def p_ucb(self, node):
        """
        Upper Confidence Bound of a chance node, weighted by prior probability
        """
        return chance_node_value(node) + self.args.ucb_constant * node.prob * sqrt(log(node.parent.visits)) / (
                1 + len(node.sampled_returns))

    def var_p_ucb(self, node):
        """
        Upper Confidence Bound of a chance node, the ucb exploration weight is a variable
        """
        ucb_parameter = log((node.parent.visits + self.ucb_base + 1) / self.ucb_base) + self.args.ucb_constant
        return chance_node_value(node) + ucb_parameter * node.prob * sqrt(log(node.parent.visits)) / (
                1 + len(node.sampled_returns))


class MineDecisionNode:
    """
    Decision node class, labelled by a state

    Args:
        dp: default policy, used to prioritize and filter possible actions
    """

    def __init__(self, parent, state, is_terminal=False, generator=None, id=None, decision_memory='', graph=None):
        self.id = id
        self.parent = parent
        self.state = state
        self.value = None
        self.is_terminal = is_terminal
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        self.generator = generator

        self.explored_children = 0
        # this decision node should be visited at least once, otherwise p-uct makes no sense for this node
        self.visits = 1
        # used to save any information of the state
        # we use this for saving complete programs generated from it
        self.info = {}
        self.decision_memory = decision_memory
        self.graph = graph

    def __expand__(self, verbal_feedback=''):
        if verbal_feedback == '':
            expand_prompt_id = self.state
            top_k_line_predict, top_k_scores = self.generator.get_top_k_rationale_predict(expand_prompt_id, self.depth, self.graph)
        else:
            expand_prompt_id = self.generator.tokenizer.encode(verbal_feedback)
            top_k_line_predict, top_k_scores = self.generator.get_top_k_rationale_predict(expand_prompt_id, self.depth, self.graph,
                                                                                          with_verbal=True)

        self.possible_actions = top_k_line_predict
        self.action_scores = top_k_scores

        # populate its children
        self.children = [MineChanceNode(self, (act, score), chance_memory=self.decision_memory) for act, score in
                         zip(self.possible_actions, self.action_scores)]

        # print(Fore.RED + '\n---------------2children tokens:')
        # children_tokens = []
        # for child_token in self.possible_actions:
        #     children_tokens.append(self.generator.tokenizer.decode(child_token))
        # print(f"{children_tokens}")

    def is_fully_expanded(self):
        return all([child.expanded() for child in self.children])


class MineChanceNode:
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """

    def __init__(self, parent, action_and_score, chance_memory=''):
        self.parent = parent
        self.action = action_and_score[0]
        self.depth = parent.depth
        self.children = []
        self.prob = action_and_score[1]  # the probability that this action should be token, provided by default policy
        self.sampled_returns = []
        self.chance_memory = chance_memory

    def expanded(self):
        return len(self.children) > 0


def chance_node_value(node, mode="best"):
    """
    Value of a chance node
    """
    if len(node.sampled_returns) == 0:
        return 0

    if mode == "best":
        # max return (reasonable because the model is deterministic?)
        return max(node.sampled_returns)
    elif mode == "sample":
        # Use average return
        return sum(node.sampled_returns) / len(node.sampled_returns)
    else:
        raise Exception(f"Unknown tree search mode {mode}")
