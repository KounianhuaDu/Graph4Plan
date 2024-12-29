import json
import random

def split(dataset_name):
    data_file = open(f"{dataset_name}/data.json", 'r')
    chains = []
    for line in data_file:
        content = json.loads(line)
        chains.append(content)
    random.shuffle(chains)
    test_data = chains[:100]
    train_data = chains[100:]
    with open(f"{dataset_name}/test_data.json", 'w') as file:
        for data in test_data:
            file.write(json.dumps(data) + '\n')
    with open(f"{dataset_name}/train_data.json", 'w') as file:
        for data in train_data:
            file.write(json.dumps(data) + '\n')
    


if __name__ == "__main__":
    random.seed(0)

    for dataset in ["huggingface", "multimedia", "dailylife"]:
        split(dataset)