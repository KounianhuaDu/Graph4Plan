import json

for dataset in ['huggingface', 'multimedia', 'dailylife']:
    task_ids = []
    with open(f"./{dataset}/test_data.json", 'r') as file:
        for line in file:
            task_id = json.loads(line)['id']
            task_ids.append(task_id)
    with open(f"./{dataset}/test_ids.json", 'w') as file:
        write_ids = {"test_ids":{
            "chain": task_ids
        }}
        json.dump(write_ids, file, indent=4)