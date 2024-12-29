

### Shimo Doc of Dukou
Startup steps:[https://shimo.im/docs/pmkxdmBO70ie50kN](https://shimo.im/docs/pmkxdmBO70ie50kN)

1. To complete the APPS generation-testing process- direct prompt, MCTS with different rollouts on GPT-3.5/LLAMA 13B.
2. To run a small batch (e.g., 200 training samples) of MCTS on GPT-3.5, retain the trace, and construct preference pairs from the trace (based on reward/exploration counts at each level of the tree).
3. To use the constructed preference pairs to fine-tune LLAMA 13B and observe changes in performance before and after tuning.


### Instructions for tmux

Create a new session with a specific name

```bash
tmux new-session -s <session-name>
```

Attach to an existing session

```bash
tmux attach-session -t <session-name>
```

Replace `<session-name>` with the name of the session you want to create or attach to.

### Pip for trl
```bash
pip install trl
pip install prettytable

```

### Instructions for Installation

Follow the steps below to set up your environment and install the necessary dependencies for your project.

#### 1. Install PyTorch with CUDA 11.8 Support

To install the GPU-enabled version of PyTorch, along with `torchvision` and `torchaudio`, use the following command:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

#### 2. Install Additional Dependencies

Create a `requirements.txt` file with the following content to install the additional dependencies:

```plaintext
openai
accelerate
wandb
torchmetrics
torcheval
dgl==1.1.2 -f https://data.dgl.ai/wheels/cu118/repo.html
matplotlib
scikit-learn
transformers
tiktoken
pyext
astunparse
jsonlines
```

After saving the `requirements.txt` file, install the dependencies with:

```bash
pip install -r requirements.txt
```

#### 3. Install `httpx` with SOCKS Proxy Support

If your application requires SOCKS proxy support, you can install `httpx` with the necessary extensions by running:

```bash
pip install httpx[socks]
```

---

# Project of Multi Task

### Train Trace Collection
```bash
python multistep_train_trace.py
```
This will yield 'res.json' and 'trace.json' for each data sample. We will use 'trace.json' to create preference pairs as the training data.

### Evaluation
```bash
python evaluate.py
```

There are three data sets in this project.
1. dailylife
2. huggingface
3. multimedia

You can change the value of the argument in the Python file `argument.py`.
``` 
parser.add_argument("--dataset", default="multimedia", help="Dataset to use, default: APPS") # "multimedia"  "huggingface" "dailylife"
```

Similarly, we offer two models, `MCTS` and `Zeroshot`, for you to choose from Python file `argument.py`.
```
parser.add_argument('-m', '--model', type=str, default='ZeroShot',  choices=['ZeroShot', 'MCTSStep'])
```

Here's the description of the .get_data() function in the GRAPHHandler class, rewritten in clear English with Markdown formatting:
```
inputs, tool_string, demo_string = hander.get_data()
```
- **inputs**: This contains a collection of all problems that haven't been trained on yet. Each problem is formatted as follows: `{'id': '23427738', 'user_request': "I am looking for a list of relevant topics based on the input text 'How to improve communication skills in the workplace?'."}`.
- **tool_string**: This prepares a tool string that prompts the Large Language Model (LLM) on how to use specific tools.
- **demo_string**: Provides an example to help the LLM understand the type of answer that is expected.


Get output file (output_dict.json or trace_feats.json) from server (APEX)
```
scp -r username@172.16.2.100:path_on_server ./path_in_local
```

Upload local dir (Project) to server
```
scp -r ./path_in_local username@172.16.2.100:path_on_server 
```
Modify `path_in_local` and `path_on_server` to the desired paths.

The code in [1] deploys the LLM on a server and then makes calls to it. We have modified it to locally load the LLM parameters for direct inference.
```bash
python multistep_train_trace_backup.py
```


Reference web link:[1][https://github.com/WxxShirley/GNN4TaskPlan/](https://github.com/WxxShirley/GNN4TaskPlan/blob/main/data/split_data.py)  
Reference web link:[2] [https://github.com/TIGER-AI-Lab/MAmmoTH/tree/main](https://github.com/TIGER-AI-Lab/MAmmoTH/tree/main)