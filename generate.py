import os
import sys
import json
import argparse

from nltk import word_tokenize
from typing import List, Dict, Any
from tqdm.contrib.concurrent import process_map
from transformers import set_seed, AutoTokenizer


sys.path.append('.')
os.environ["TOKENIZERS_PARALLELISM"] = "true"


PROMPT = """
Generate a SQL to answer the question with the given schema.
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
```sql
{schema}
```

Question: {question}
```sql
{sql}
```
""".strip()

EXAMPLE_USER = """
```sql
{schema}
```

Question: {question}
""".strip()


def pack_question(data: Dict[str, Any], use_alignment: str = "") -> str:
    if use_alignment and data['alignment']:
        alignment: List[Dict[str, str]] = data['alignment'][use_alignment]
        return ' '.join([f"{a['token']} ( {a['schema']} )" if a['type'] and a['schema'] else a['token'] for a in alignment])
    return data['question']


def generate_prompt(data_demo_pair):
    data, demos, args, schemas_dev, schemas_train, max_position_embeddings, tokenizer = data_demo_pair
    if args.aligned:
        schema_user = database_to_string(pack_db_path(
            args.dev_database_path, data['db_id']), "table", data['query_pred'], data['question'], schemas_dev[data['db_id']]).strip()
        if not schema_user:
            schema_user = database_to_string(
                pack_db_path(args.dev_database_path, data['db_id']), data['question'])
        prompt_demos = [EXAMPLE.format(
            schema=database_to_string(pack_db_path(
                args.train_database_path, demo['db_id']), "table", demo['query'], demo['question'], schemas_train[demo['db_id']]),
            question=pack_question(demo),
            sql=demo['query']
        ) for demo in demos[:args.shot]]
        prompt_user = EXAMPLE_USER.format(
            schema=schema_user,
            question=pack_question(data)
        )
    else:
        prompt_demos = [EXAMPLE.format(
            schema=database_to_string(pack_db_path(
                args.train_database_path, demo['db_id']), question=demo['question']),
            question=pack_question(demo),
            sql=demo['query']
        ) for demo in demos[:args.shot]]
        prompt_user = EXAMPLE_USER.format(
            schema=database_to_string(pack_db_path(
                args.dev_database_path, data['db_id']), question=data['question']),
            question=pack_question(data)
        )

    prompt = PROMPT.format(
        demonstration="\n\n---\n\n".join(prompt_demos), user=prompt_user)
    while len(tokenizer(prompt)['input_ids']) if tokenizer else len(word_tokenize(prompt)) > max_position_embeddings - 64 and prompt_demos:
        prompt_demos = prompt_demos[:-1]
        prompt = PROMPT.format(
            demonstration="\n\n---\n\n".join(prompt_demos), user=prompt_user)
    return prompt


if __name__ == '__main__':
    from utils.selector import select_multiple
    from utils.generator import generate_with_llm
    from utils.database import pack_db_path, database_to_string, fix_sql

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
    parser.add_argument("--aligned", action='store_true')
    parser.add_argument("--generate_mode", type=str,
                        choices=['chat', 'text'], default='chat')
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")
    parser.add_argument("--shot", type=int, default=5)
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

    model_config_file = os.path.join(args.llm_name_or_path, 'config.json')
    if os.path.exists(model_config_file):
        with open(model_config_file, 'r', encoding='utf-8') as f:
            max_position_embeddings = json.load(f)['max_position_embeddings']
        tokenzier = AutoTokenizer.from_pretrained(args.llm_name_or_path)
    else:
        max_position_embeddings = 8192
        tokenzier = None
    if args.data_size:
        data_dev = data_dev[:args.data_size]

    demonstrations = select_multiple(
        data_dev, data_train, demonstration_number=args.shot)
    # Prepare the arguments for process_map
    data_demo_pairs = [(data, demos, args, schemas_dev, schemas_train, max_position_embeddings, tokenzier)
                       for data, demos in zip(data_dev, demonstrations)]
    # Generate prompts using process_map for parallel processing
    prompts = process_map(generate_prompt, data_demo_pairs,
                          desc="Generating Prompts", chunksize=1)
    print(prompts[0])

    predictions = generate_with_llm(
        args.llm_name_or_path, prompts, config, args.generate_mode)
    for d, p, x in zip(data_dev, predictions, prompts):
        prediction = p[0][0].strip()
        d['prediction'] = {
            "text": prediction,
            "query": fix_sql(prediction, 1 if 'codellama' in args.llm_name_or_path.lower() else -1),
        }
    with open(args.dump_file, 'w', encoding='utf-8') as f:
        json.dump(data_dev, f, ensure_ascii=False, indent=4)
    with open(args.dump_file.replace('.json', '.sql'), 'w', encoding='utf-8') as f:
        f.write('\n'.join([d['prediction']['query'] + '\t' + d['db_id']
                if d['prediction'] else f"SELECT\t{d['db_id']}" for d in data_dev]))
