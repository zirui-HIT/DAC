import os
import sys
import json
import random
import sqlparse
import argparse

from transformers import set_seed
from func_timeout import func_timeout
from typing import List, Dict, Any, Tuple, Union, Callable
from tqdm.contrib.concurrent import process_map


set_seed(42)
sys.path.append('.')
os.environ["TOKENIZERS_PARALLELISM"] = "true"


SPIDER_EVALUATE_CMD = """
python3 ./evaluate/src/Spider/evaluation.py \
        --gold ./dataset/Spider/dev.sql \
        --pred {pred_sql_file} \
        --db ./dataset/Spider/database/ \
        --etype exec \
        --plug_value \
        --evaluate_results_file {eval_file}
""".strip()

BIRD_EVALUATE_CMD = """
python3 -u ./evaluate/src/Bird/evaluation.py \
        --db_root_path ./dataset/Bird/database/ \
        --predicted_sql_path {pred_sql_file} \
        --data_mode dev \
        --ground_truth_path ./dataset/Bird/dev.sql \
        --diff_json_path {pred_json_file} \
        --dump_file {eval_file}
"""

KAGGLE_EVALUATE_CMD = """
python3 ./evaluate/src/KaggleDBQA/evaluation.py \
        --gold ./dataset/KaggleDBQA/dev.sql \
        --pred {pred_sql_file} \
        --db ./dataset/KaggleDBQA/database/ \
        --etype exec \
        --plug_value \
        --evaluate_results_file {eval_file}
""".strip()


def prompt_execution_error(args_tuple: Tuple[Dict[str, Any], Dict[str, Any], str]) -> Tuple[str, bool]:
    PROMPT = """
```sql
{schema}
```

Fix the sql "{sql}" to answer the question "{question}" based on the above database and the error information.
The error information is "{error}".
Present your sql in the format:
```sql
<your sql>
```
    """.strip()

    d, schema, db_path = args_tuple
    sql_execution = execute_sql(
        d['prediction']['query'], pack_db_path(args.database_path, d['db_id']))
    if not sql_execution.startswith('Error'):
        return "", False

    prompt = PROMPT.format(
        schema=database_to_string(pack_db_path(
            db_path, d['db_id']), "table", d['prediction']['query'], d['question'], schema),
        sql=d['prediction']['query'],
        question=pack_question(d, "pred"),
        error=sql_execution
    )
    prompt = prompt.strip()

    return prompt, True


def prompt_entity_linking(args_tuple: Tuple[Dict[str, Any], Dict[str, Any], str]) -> Tuple[str, bool]:
    def extract_mismatch(sql: str, alignment: Union[List[Dict[str, str]], str], schema: Dict[str, Any]) -> Dict[str, List[str]]:
        def extract_sql_entities(sql: str, schema: Dict[str, Any]) -> Dict[str, List[str]]:
            def align_schema_entities(sql: str, entities: List[str], schema: Dict[str, Any]) -> Dict[str, List[str]]:
                from utils.database import extract_table_aliases

                result: Dict[str, List[str]] = {
                    "table": [],
                    "column": []
                }
                aliases = extract_table_aliases(sql)

                result['table'] = [x.lower() for x in aliases.values()]
                table_names = [name.lower()
                               for name in schema['table_names_original']]
                for e in entities:
                    if e.lower() in table_names:
                        result['table'].append(e)
                result['table'] = list(
                    set([t.lower() for t in result['table']]))

                columns: List[str] = []
                for e in entities:
                    used_flag = False
                    if e.lower() in result['table']:
                        continue
                    for t in aliases.keys():
                        if f" {t}.{e}".lower() in sql.lower():
                            result['column'].append(f"{aliases[t]}.{e}")
                            used_flag = True
                            break
                    if not used_flag:
                        for t in result['table']:
                            if f" {t}.{e}".lower() in sql.lower():
                                result['column'].append(f"{t}.{e}")
                                used_flag = True
                                break
                    if not used_flag:
                        columns.append(e)

                for e in columns:
                    for c in schema['column_names_original']:
                        if e.lower() == c[1].lower() and schema['table_names_original'][c[0]].lower() in result['table']:
                            result['column'].append(
                                f"{schema['table_names_original'][c[0]]}.{c[1]}")
                            break
                return result

            sql_removed_on_clause = remove_on_clause(
                list(sqlparse.parse(sql)[0]))[0]
            schema_entities = extract_schema(sql_removed_on_clause, schema)
            return align_schema_entities(sql_removed_on_clause, schema_entities, schema)

        def extract_alignment_entities(alignment: List[Dict[str, str]]) -> Dict[str, List[str]]:
            result = {
                "table": [],
                "column": []
            }
            for a in alignment:
                if not a['type'] or not a['schema'] or '*' in a['schema']:
                    continue
                if a['type'] in ['col', 'val']:
                    result['column'].append(a['schema'])
                if a['type'] == 'tbl':
                    result['table'].append(a['schema'])
            result['table'] = list(set(result['table']))
            result['column'] = list(set(result['column']))
            return result

        sql_entities_pred = extract_sql_entities(sql, schema)
        if isinstance(alignment, str):
            alignment_entities = extract_sql_entities(alignment, schema)
        else:
            alignment_entities = extract_alignment_entities(alignment)
        sql_entities_pred['table'] = list(
            set([x.lower() for x in sql_entities_pred['table']]))
        sql_entities_pred['column'] = list(
            set([x.lower() for x in sql_entities_pred['column']]))
        alignment_entities['table'] = list(
            set([x.lower() for x in alignment_entities['table']]))
        alignment_entities['column'] = list(
            set([x.lower() for x in alignment_entities['column']]))

        result = {
            "alignment_table": [],
            "sql_table": [],
            "alignment_column": [],
            "sql_column": []
        }
        if any(x not in sql_entities_pred['table'] for x in alignment_entities['table']):
            result['alignment_table'] = list(set(
                [x for x in alignment_entities['table'] if x not in sql_entities_pred['table']]))
        if any(x not in alignment_entities['table'] for x in sql_entities_pred['table']):
            result['sql_table'] = list(set(
                [x for x in sql_entities_pred['table'] if x not in alignment_entities['table']]))
        if any(x not in sql_entities_pred['column'] for x in alignment_entities['column']):
            result['alignment_column'] = list(set(
                [x for x in alignment_entities['column'] if x not in sql_entities_pred['column']]))
        if any(x not in alignment_entities['column'] for x in sql_entities_pred['column']):
            result['sql_column'] = list(set(
                [x for x in sql_entities_pred['column'] if x not in alignment_entities['column']]))
        return result

    PROMPT = """
```sql
{schema}
```

Fix the sql "{sql}" to answer the question "{question}" based on the above database and the alignment.
Present your sql in the format:
```sql
<your sql>
```
It should be noticed that {notification}. Your sql must contain the tables and columns mentioned by the question.
    """.strip()

    d, schema, db_path, use_oracle = args_tuple
    mismatch = extract_mismatch(
        d['prediction']['query'], d['query'] if use_oracle else d['alignment']['pred'], schema)
    notification = ""
    if mismatch['alignment_table']:
        notification += " , " + "tables " + \
            " , ".join([f'"{t}"' for t in mismatch
                        ['alignment_table']]) + " are mentioned by the question"
    if mismatch['alignment_column']:
        notification += " , " + "columns " + \
            " , ".join([f'"{t}"' for t in mismatch
                        ['alignment_column']]) + " are mentioned by the question"
    notification = notification.strip(" ,")
    if not notification:
        return "", False

    prompt = PROMPT.format(
        schema=database_to_string(pack_db_path(
            db_path, d['db_id']), "table", d['prediction']['query'], d['question'], schema, mismatch['alignment_table']),
        sql=d['prediction']['query'],
        question=pack_question(d, "pred"),
        notification=notification
    )
    prompt = prompt.strip()

    return prompt, True


def prompt_hallucination(args_tuple: Tuple[Dict[str, Any], Dict[str, Any], str]) -> Tuple[str, bool]:
    PROMPT = """
```sql
{schema}
```

Fix the sql "{sql}" to answer the question "{question}" with the above schema.
Present your sql in the format:
```sql
<your sql>
```
It should be noticed that the SQL skeleton could be like "{skeleton}", where each '_' can only be replaced with one single table, column or value.
    """.strip()

    d, schema, db_path, use_oracle = args_tuple

    d_sql_skeleton = extract_skeleton(d['prediction']['query'])
    if use_oracle:
        d_question_skeleton = extract_skeleton(d['query'])
    else:
        d_question_skeleton = d['hallucination']

    if d_sql_skeleton == d_question_skeleton:
        return "", False
    if len(d_sql_skeleton.split()) < len(d_question_skeleton.split()):
        return "", False

    prompt = PROMPT.format(
        schema=database_to_string(pack_db_path(
            db_path, d['db_id']), "table", d['prediction']['query'], d['question'], schema),
        sql=d['prediction']['query'],
        question=pack_question(d, "pred"),
        skeleton=d_question_skeleton
    )
    prompt = prompt.strip()

    return prompt, True


def pack_single_prompt(args_tuple: Tuple[Dict[str, Any], Dict[str, Any], str, Callable]) -> Tuple[str, bool]:
    try:
        args_tuple, pack_function = args_tuple[:-1], args_tuple[-1]
        return func_timeout(120, pack_function, (args_tuple, ))
    except:
        return "", False


def pack_prompt(data: List[Dict[str, Any]], prompt_function: Callable, use_oracle: bool = False) -> Tuple[List[str], List[bool]]:
    tasks = [(d, schemas[d['db_id']], args.database_path, use_oracle, prompt_function)
             for d in data]

    results = process_map(pack_single_prompt, tasks,
                          desc='Generating Prompt', chunksize=1)
    # results = [pack_single_prompt(t) for t in tqdm(tasks)]
    if not results:
        return [], []
    prompts, fix_flag = zip(*results)
    prompts = [p for p in prompts if p]
    fix_flag = list(fix_flag)

    # print(random.choice(prompts) if prompts else "No prompts generated")
    return prompts, fix_flag


def generate(data: List[Dict[str, Any]], prompt_function: Callable, temperature: float, use_oracle: bool = False) -> List[Dict[str, Any]]:
    retry_times = 0
    prompts_length_previous = -1
    fix_indices = list(range(len(data)))  # Initialize with all indices

    while True:
        config['temperature'] = temperature if retry_times > 0 else 0
        current_data = [data[i] for i in fix_indices]  # Filter data
        prompts, fix_flag = pack_prompt(
            current_data, prompt_function, use_oracle)
        if not prompts or len(prompts) == prompts_length_previous:
            break

        print("Retry times:", retry_times, "Temperature:",
              config['temperature'], "Data Number:", len(prompts))
        retry_times += 1
        prompts_length_previous = len(prompts)

        # Postprocess prediction
        fix_idx = 0
        predictions = generator.generate(prompts, config)
        new_fix_indices = []
        for i, f in zip(fix_indices, fix_flag):
            if not f:
                continue
            d = data[i]
            p = predictions[fix_idx]
            fix_idx += 1
            prediction = p[0][0].strip()
            query_pred = fix_sql(prediction)

            skeleton_prev = extract_skeleton(d['query_pred'])
            skeleton_post = extract_skeleton(query_pred)

            if execute_sql(query_pred, pack_db_path(args.database_path, d['db_id'])).startswith('Error'):
                query_pred = d['query_pred']
            elif query_pred == data[i]['prediction']['query']:
                continue
            elif len(skeleton_post.split()) > len(skeleton_prev.split()):
                continue
            data[i]['prediction'] = {
                "text": prediction,
                "query": query_pred
            }
            new_fix_indices.append(i)
        fix_indices = new_fix_indices

        if args.debug:
            # Evaluate result for debug
            print("Evaluating...")
            save_data(data, args.dump_file)
            if 'spider' in args.dump_file.lower():
                eval_cmd = SPIDER_EVALUATE_CMD.format(
                    pred_sql_file=args.dump_file.replace('.json', '.sql'),
                    eval_file=args.dump_file.replace('.json', '.eval.json')
                )
            elif 'bird' in args.dump_file.lower():
                eval_cmd = BIRD_EVALUATE_CMD.format(
                    pred_sql_file=args.dump_file.replace('.json', '.sql'),
                    pred_json_file=args.dump_file,
                    eval_file=args.dump_file.replace('.json', '.eval.json')
                )
            elif 'kaggle' in args.dump_file.lower():
                eval_cmd = KAGGLE_EVALUATE_CMD.format(
                    pred_sql_file=args.dump_file.replace('.json', '.sql'),
                    eval_file=args.dump_file.replace('.json', '.eval.json')
                )
            execute_command(eval_cmd)

    return data


def save_data(data: List[Dict[str, str]], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    with open(path.replace('.json', '.sql'), 'w', encoding='utf-8') as f:
        f.write('\n'.join([d['prediction']['query'] + '\t' + d['db_id']
                if d['prediction'] else f"SELECT\t{d['db_id']}" for d in data]))


if __name__ == '__main__':
    from generate import pack_question
    from utils.program import execute_command
    from utils.generator import detect_generator
    from utils.database import pack_db_path, database_to_string, fix_sql, execute_sql, extract_skeleton, extract_schema, remove_on_clause

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name_or_path", type=str, help="llm path")
    parser.add_argument("--config_file", type=str, help="config path")
    parser.add_argument("--data_file", type=str, help="data path")
    parser.add_argument("--schema_file", type=str, help="schema file")
    parser.add_argument("--database_path", type=str, help="database path")
    parser.add_argument("--dump_file", type=str, help="dump path")
    parser.add_argument("--data_size", type=int, help="data size")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--temperature", type=float,
                        default=0.3, help="temperature of retrying")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")
    parser.add_argument('--ablation', type=str,
                        choices=['none', 'entity', 'skeleton', 'all'], default='none')
    parser.add_argument(
        '--oracle', type=str, choices=['none', 'entity', 'skeleton', 'all'], default='none')
    args = parser.parse_args()
    set_seed(args.random_seed)

    if args.ablation == 'all':
        args.ablation = ['entity', 'skeleton']
    elif args.ablation == 'none':
        args.ablation = []
    else:
        args.ablation = [args.ablation]
    if args.oracle == 'all':
        args.oracle = ['entity', 'skeleton']
    elif args.oracle == 'none':
        args.oracle = []
    else:
        args.oracle = [args.oracle]

    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(args.schema_file, 'r', encoding='utf-8') as f:
        schemas = {table['db_id']: table for table in json.load(f)}
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    generator = detect_generator(args.llm_name_or_path, 'chat')
    if args.data_size:
        data = random.sample(data, args.data_size)

    if 'entity' not in args.ablation:
        data = generate(data, prompt_entity_linking,
                        args.temperature, 'entity' in args.oracle)
    if 'skeleton' not in args.ablation:
        data = generate(data, prompt_hallucination,
                        args.temperature, 'skeleton' in args.oracle)
    data = generate(data, prompt_execution_error, args.temperature)
    for d in data:
        if 'query_pred' in d:
            d.pop('query_pred')
    save_data(data, args.dump_file)
