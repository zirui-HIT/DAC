import sys
import json

sys.path.append('.')


if __name__ == '__main__':
    from align import build_alignment

    with open('EtA_dir/Spider/dev.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    data_dev_aligned = [{
        "db_id": d['schema']['db_id'],
        "question": ' '.join(t['token'] for t in d['question']['tokens']),
        "query": ' '.join(str(t['value']) for t in d['sql']['tokens']),
        "alignment": {"gold": build_alignment(d)}
    } for d in data]
    with open('./result/dev.json', 'w', encoding='utf-8') as f:
        json.dump(data_dev_aligned, f, ensure_ascii=False, indent=4)
