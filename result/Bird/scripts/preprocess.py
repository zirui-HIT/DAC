import json


if __name__ == '__main__':
    for part in ['train', 'dev']:
        with open(f'./dataset/Bird/{part}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = [{
            "db_id": d['db_id'],
            "question": f"{d['question'].strip('.?!')}, where \" {d['evidence']} \"." if d['evidence'] else d['question'],
            "query": d['query'],
            "difficulty": d['difficulty'] if 'difficulty' in d else None,
            "alignment": {}
        } for d in data]
        with open(f'./result/Bird/{part}.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
