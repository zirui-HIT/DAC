import json


if __name__ == '__main__':
    for part in ['dev', 'train']:
        with open(f'./dataset/KaggleDBQA/{part}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        for d in data:
            results.append({
                'db_id': d['db_id'],
                'question': d['question'],
                'query': d['query'],
                'alignment': {}
            })

        with open(f'./result/KaggleDBQA/{part}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)