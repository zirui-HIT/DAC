import os
import sys
import json
import argparse

from transformers import set_seed
from typing import List, Dict, Any
from tqdm.contrib.concurrent import process_map


sys.path.append('.')
os.environ["TOKENIZERS_PARALLELISM"] = "true"


PROMPT = """
Hallucinate a SQL to answer the question.
Quote your answer with: 
```sql
<answer sql>
```

---

For example:

{demonstration}

---

Based on the instruction and the examples, answer the following question:

{user}
""".strip()

EXAMPLE = """
Question: {question}
```sql
{sql}
```
""".strip()

EXAMPLE_USER = """
Question: {question}
""".strip()


def pack_question(data: Dict[str, Any], use_alignment: str = "") -> str:
    if use_alignment and data['alignment']:
        alignment: List[Dict[str, str]] = data['alignment'][use_alignment]
        return ' '.join([f"{a['token']} ( {a['schema']} )" if a['type'] and a['schema'] else a['token'] for a in alignment])
    return data['question']


def generate_prompt(data_demo_pair):
    data, demos, args, schemas_dev, schemas_train = data_demo_pair
    prompt_demos = [EXAMPLE.format(
        question=pack_question(demo),
        sql=demo['query']
    ) for demo in demos[:args.shot]]
    prompt_user = EXAMPLE_USER.format(
        question=pack_question(data)
    )
    return PROMPT.format(demonstration="\n\n---\n\n".join(prompt_demos), user=prompt_user)


if __name__ == '__main__':
    from utils.selector import select_multiple
    from utils.generator import generate_with_llm, consistency
    from utils.database import fix_sql, extract_skeleton

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name_or_path", type=str, help="llm path")
    parser.add_argument("--config_file", type=str, help="config path")
    parser.add_argument("--train_data_file", type=str, help="data path")
    parser.add_argument("--train_schema_file", type=str, help="schema file")
    parser.add_argument("--train_database_path",
                        type=str, help="database path")
    parser.add_argument("--dev_data_file", type=str, help="data path")
    parser.add_argument("--dev_schema_file", type=str, help="schema file")
    parser.add_argument("--dev_database_path", type=str, help="database path")
    parser.add_argument("--dump_file", type=str, help="dump path")
    parser.add_argument("--data_size", type=int, help="data size")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")
    parser.add_argument("--shot", type=int, default=3)
    args = parser.parse_args()
    set_seed(args.random_seed)

    with open(args.dev_data_file, 'r', encoding='utf-8') as f:
        data_dev = json.load(f)
    with open(args.train_data_file, 'r', encoding='utf-8') as f:
        data_train = json.load(f)
    with open(args.dev_schema_file, 'r', encoding='utf-8') as f:
        schemas_dev = {s['db_id']: s for s in json.load(f)}
    with open(args.train_schema_file, 'r', encoding='utf-8') as f:
        schemas_train = {s['db_id']: s for s in json.load(f)}
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if args.data_size:
        data_dev = data_dev[:args.data_size]

    demonstrations = select_multiple(
        data_dev, data_train, demonstration_number=args.shot)
    # Prepare the arguments for process_map
    data_demo_pairs = [(data, demos, args, schemas_dev, schemas_train)
                       for data, demos in zip(data_dev, demonstrations)]
    # Generate prompts using process_map for parallel processing
    prompts = process_map(generate_prompt, data_demo_pairs,
                          desc="Generating Prompts", chunksize=1)
    print(prompts[0])

    predictions = generate_with_llm(
        args.llm_name_or_path, prompts, config, 'chat')
    for d, p, x in zip(data_dev, predictions, prompts):
        results = []
        skeleton_prev = extract_skeleton(d['query_pred'])
        for pred in p:
            prediction = pred[0].strip()
            skeleton = extract_skeleton(fix_sql(prediction))
            if len(skeleton.split()) > len(skeleton_prev.split()):
                continue
            results.append((skeleton, prediction, pred[1]))

        if not results:
            d['hallucination'] = extract_skeleton(fix_sql(p[0][0].strip()))
            continue
        d['hallucination'] = consistency(results)[0]

    with open(args.dump_file, 'w', encoding='utf-8') as f:
        json.dump(data_dev, f, ensure_ascii=False, indent=4)
