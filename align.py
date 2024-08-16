import os
import sys
import json
import argparse
import sqlparse

from transformers import set_seed, AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm.contrib.concurrent import process_map
from typing import List, Tuple, Dict, Any, Union


sys.path.append('.')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
LEMMATIZER = WordNetLemmatizer()


PROMPT = """
Align the tokens in the given question to the table entities or the column entities of the schema above, considering the given SQL.
Present the aligned tokens in the python format List[Dict[str, str]], where each Dict[str, str] denoting each token in the question containing the following keys:
{{
    "token": the token in the question
    "schema": the schema entity aligned to the token
    "type": the type of the entity aligned to the token
}}
The "type" can be one of the following:
* "tbl": the table name
* "col": the column name
* "val": the value
"schema" and "type" are either both null or not null at the same time.

Here are some examples.

---

{demonstration}

---

Based on the instruction and the examples above, solve the following question:

{user}
""".strip()

EXAMPLE = """
{schema}

SQL: {sql}
Question: {question}
Alignments: {alignment}
""".strip()


def build_alignment(data: Dict[str, Any]) -> List[Dict[str, Union[str, None]]]:
    result: List[Tuple[str, str]] = []
    for t, c in zip(data['question']['tokens'], data['align_labels']):
        entity_aligned = ""
        if not c:
            entity_aligned = None
        elif c['type'] == 'tbl':
            entity_aligned = data['schema']['table_names_original'][int(
                c['id'])]
        elif c['type'] == 'col' or c['type'] == 'val':
            table_idx = data['schema']['column_to_table'][str(c['id'])]
            table_name = data['schema']['table_names_original'][int(table_idx)]
            column_name = data['schema']['column_names_original'][int(c['id'])]
            entity_aligned = f"{table_name}.{column_name}"
        else:
            entity_aligned = None
        result.append({
            "token": t['token'],
            "schema": entity_aligned,
            "type": c['type'] if c else None
        })
    return result


def pack_alignment(alignments: List[Dict[str, Union[str, None]]]) -> str:
    return '\n```\n' + json.dumps(alignments, ensure_ascii=False, indent=4) + '\n```'


def unpack_alignment(alignment: str) -> List[Dict[str, Union[str, None]]]:
    try:
        alignment = alignment.split('[')[1].split(']')[0].strip()
        alignment = '[' + alignment + ']'
        alignment_sentences = alignment.split('\n')
        for i, s in enumerate(alignment_sentences):
            s = s.strip()
            s = s.replace("，", ",")
            if '# ' in s:
                s = s.split('# ')[0].strip()
            if '", ' in s:
                s = s.replace('", ', '": ')
            if i > 0 and alignment_sentences[i-1].startswith('},') and not s.startswith('{'):
                alignment_sentences[i-1] += "\n{"
            alignment_sentences[i] = s
        alignment = '\n'.join(alignment_sentences)
        results = json.loads(alignment)
        return results
    except Exception as e:
        print(str(e) + ':')
        print(alignment)
        return []


def alignment_consistency(alignments: List[Tuple[List[Dict[str, Union[str, None]]], float]]) -> List[Dict[str, Union[str, None]]]:
    length_count: Dict[int, float] = {}
    for a in alignments:
        length = len(a[0])
        length_count[length] = length_count.get(length, 0) + a[1]
    length_best = max([(x[0], x[1])
                      for x in length_count.items()], key=(lambda x: x[1]))[0]
    alignments = [a for a in alignments if len(a[0]) == length_best]

    results: List[Dict[str, Union[str, None]]] = []
    for i in range(length_best):
        record: Dict[str, Tuple[Dict[str, Union[str, None]], float]] = {}
        for a in alignments:
            if not a or not isinstance(a[0], list):
                continue
            tag = str(a[0][i])
            if 'type' not in a[0][i] or 'schema' not in a[0][i]:
                continue
            if (a[0][i]['type'] and not a[0][i]['schema']) or (a[0][i]['schema'] and not a[0][i]['type']):
                continue
            if tag in record:
                record[tag] = (a[0][i], record[tag][1] + a[1])
            else:
                record[tag] = (a[0][i], a[1])
        if len(record) == 0:
            record_best = alignments[0][0][i]
        else:
            record_best = max([x for x in record.values()],
                              key=(lambda x: x[1]))[0]
        results.append(record_best)

    return results


def span_alignment_fix(question: str, alignment: List[Dict[str, str]], schema: Dict[str, List[str]]) -> List[Dict[str, str]]:
    def lemmatize_word(word: str) -> Tuple[str, str]:
        noun = LEMMATIZER.lemmatize(word, pos='n')
        if noun.lower() != word.lower():
            return noun
        verb = LEMMATIZER.lemmatize(word, pos='v')
        return verb

    def fix_question(question: str, alignment: List[Dict[str, str]]) -> List[Dict[str, str]]:
        def longest_common_subsequence(list1: list[str], list2: list[str]) -> tuple[list[int], list[int]]:
            m = len(list1)
            n = len(list2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if list1[i - 1] == list2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            indices_list1 = []
            indices_list2 = []
            i, j = m, n
            while i > 0 and j > 0:
                if list1[i - 1] == list2[j - 1]:
                    indices_list1.append(i - 1)
                    indices_list2.append(j - 1)
                    i -= 1
                    j -= 1
                elif dp[i - 1][j] > dp[i][j - 1]:
                    i -= 1
                else:
                    j -= 1
            indices_list1.reverse()
            indices_list2.reverse()
            return (indices_list1, indices_list2)

        question_tokens = word_tokenize(question.lower())
        alignment_tokens = [a['token'].lower()
                            for a in alignment if 'token' in a and a['token']]
        question_idx, alignment_idx = longest_common_subsequence(
            question_tokens, alignment_tokens)

        result: List[Dict[str, str]] = []
        for i, q in enumerate(question_tokens):
            if i not in question_idx:
                result.append({
                    "token": q,
                    "schema": None,
                    "type": None
                })
                continue
            idx = question_idx.index(i)
            if 'schema' not in alignment[alignment_idx[idx]] or 'type' not in alignment[alignment_idx[idx]]:
                result.append({
                    "token": q,
                    "schema": None,
                    "type": None
                })
                continue
            if not alignment[alignment_idx[idx]]['schema'] or not alignment[alignment_idx[idx]]['type']:
                result.append({
                    "token": q,
                    "schema": None,
                    "type": None
                })
                continue
            result.append({
                "token": q,
                "schema": str(alignment[alignment_idx[idx]]['schema']),
                "type": str(alignment[alignment_idx[idx]]['type'])
            })

        return result

    def get_span_length(span: str, idx: int, alignment: List[Dict[str, str]]) -> int:
        span_length = len(word_tokenize(span))
        if idx + span_length > len(alignment):
            return None
        if not all(x['schema'] is None for x in alignment[idx:idx + span_length]):
            return None
        return span_length

    if not alignment:
        alignment = [{
            "token": token,
            "schema": None,
            "type": None
        } for token in word_tokenize(question)]
    alignment = fix_question(question, alignment)

    used_tables = [x['schema'].split('.')[0].lower()
                   for x in alignment if x['schema'] and '.' in x['schema']] + [x['schema'].lower() for x in alignment if x['type'] == 'tbl']
    for i, p in enumerate(alignment):
        # Find Column with Used Table
        for c in schema['column_names_original']:
            table = schema['table_names_original'][c[0]].lower()
            if table not in used_tables:
                continue
            span_length = get_span_length(c[1], i, alignment)
            if not span_length:
                continue
            span = ''.join([lemmatize_word(x['token'])
                            for x in alignment[i:i + span_length]]).lower()
            column_span = ''.join([lemmatize_word(x)
                                   for x in c[1].split('_')]).lower()
            column_span = column_span.replace(' ', '')
            if span == column_span:
                for j in range(span_length):
                    alignment[i +
                              j]['schema'] = f"{schema['table_names_original'][c[0]]}.{c[1]}"
                    alignment[i + j]['type'] = "col"
                break
        if p['schema']:
            continue
        # Find Column without Used Table
        for c in schema['column_names_original']:
            table = schema['table_names_original'][c[0]].lower()
            span_length = get_span_length(c[1], i, alignment)
            if not span_length:
                continue
            span = ''.join([lemmatize_word(x['token'])
                            for x in alignment[i:i + span_length]]).lower()
            column_span = ''.join([lemmatize_word(x)
                                   for x in c[1].split('_')]).lower()
            column_span = column_span.replace(' ', '')
            if span == column_span:
                for j in range(span_length):
                    alignment[i +
                              j]['schema'] = f"{schema['table_names_original'][c[0]]}.{c[1]}"
                    alignment[i + j]['type'] = "col"
                break
        if p['schema']:
            continue
        # Find Table
        for t in schema['table_names_original']:
            span_length = get_span_length(t, i, alignment)
            if not span_length:
                continue
            span = ''.join([lemmatize_word(x['token'])
                            for x in alignment[i:i + span_length]]).lower()
            table_span = ''.join([lemmatize_word(x)
                                  for x in t.split('_')])
            table_span = table_span.replace(' ', '')
            if span.lower() == table_span.lower():
                for j in range(span_length):
                    alignment[i + j]['schema'] = t
                    alignment[i + j]['type'] = "tbl"
                break

    return alignment


def evaluate(data: List[Dict[str, str]]) -> Dict[str, List[float]]:
    def evaluate_performance(gold: List[Dict[str, Union[str, None]]], pred: List[Dict[str, Union[str, None]]]) -> Tuple[float, float, float]:
        # print(json.dumps(gold, ensure_ascii=False, indent=4))
        # print(json.dumps(pred, ensure_ascii=False, indent=4))
        gold_set = set([' | '.join([t for t in x.values()]).lower()
                        for x in gold if x["schema"] and x["type"]])
        pred_set = set([' | '.join([t for t in x.values()]).lower()
                        for x in pred if x["schema"] and x["type"]])
        if not gold_set:
            return (1.0, 1.0, 1.0)

        true_positives = len(gold_set & pred_set)

        if len(gold_set) == 0:
            precision = 0
        else:
            precision = true_positives / len(gold_set)
        if len(pred_set) == 0:
            recall = 0
        else:
            recall = true_positives / len(pred_set)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    evaluation: Dict[str, List[float]] = {
        "tbl": [0, 0, 0],
        "col": [0, 0, 0],
        "val": [0, 0, 0]
    }
    for d in data:
        d['alignment']['eval'] = {}
        for schema_type in ['tbl', 'col', 'val']:
            try:
                entities_gold = [x for x in d['alignment']['gold']
                                 if x['type'] and x['type'] == schema_type]
                entities_pred = [x for x in d['alignment']['pred']
                                 if x['type'] and x['type'] == schema_type]
                d['alignment']['eval'][schema_type] = evaluate_performance(
                    entities_gold, entities_pred)
            except Exception as e:
                print(f"Error \"{str(e)}\" occurred: {d['question']}")
                d['alignment']['eval'][schema_type] = [0, 0, 0]
            evaluation[schema_type] = [
                e + a for e, a in zip(evaluation[schema_type], d['alignment']['eval'][schema_type])]
    for schema_type in ['tbl', 'col', 'val']:
        evaluation[schema_type] = [e / len(data)
                                   for e in evaluation[schema_type]]
    return evaluation


def extract_mismatch(sql: str, alignment: List[Dict[str, str]], schema: Dict[str, Any]) -> Dict[str, List[str]]:
    def extract_sql_entities(sql: str, schema: Dict[str, Any]) -> Dict[str, List[str]]:
        def align_schema_entities(entities: List[str], schema: Dict[str, Any]) -> Dict[str, List[str]]:
            result = {
                "table": [],
                "column": []
            }
            for e in entities:
                if e in schema['table_names_original']:
                    result['table'].append(e)
            for e in entities:
                for c in schema['column_names_original']:
                    if e.lower() == c[1].lower() and schema['table_names_original'][c[0]] in result['table']:
                        result['column'].append(
                            f"{schema['table_names_original'][c[0]]}.{c[1]}")
                        break
            return result

        sql_removed_on_clause = remove_on_clause(
            list(sqlparse.parse(sql)[0]))[0]
        schema_entities = extract_schema(sql_removed_on_clause, schema)
        return align_schema_entities(schema_entities, schema)

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
    alignment_entities = extract_alignment_entities(alignment)
    result = {}
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


def generate_prompt(data_demo_pair):
    data, demos, args, schemas_train, schemas_dev, EXAMPLE, PROMPT, pack_db_path, database_to_string, pack_alignment, max_position_embeddings, tokenizer = data_demo_pair
    granularity = 'table'

    demos_used = demos[:args.shot]
    schema_type_used = set(
        [x['type'] for demo in demos_used for x in demo['alignment']['gold'] if x['type']])
    for demo in demos[args.shot:]:
        flag = False
        for x in demo['alignment']['gold']:
            if x['type'] and x['type'] not in schema_type_used:
                demos_used[-1] = demo
                flag = True
                break
        if flag:
            break

    prompt_demos = [EXAMPLE.format(
        schema=database_to_string(pack_db_path(
            args.train_database_path, demo['db_id']), granularity, demo['query'], demo['question'], schemas_train[demo['db_id']]),
        sql=demo['query'],
        question=demo['question'],
        alignment=pack_alignment(demo['alignment']['gold'])
    ) for demo in demos_used]

    schema_user = database_to_string(pack_db_path(
        args.dev_database_path, data['db_id']), granularity, data['query_pred'], data['question'], schemas_dev[data['db_id']]).strip()
    if not schema_user:
        schema_user = database_to_string(
            pack_db_path(args.dev_database_path, data['db_id']), question=data['question'])
    prompt_user = EXAMPLE.format(
        schema=schema_user,
        sql=data['query_pred'],
        question=data['question'],
        alignment=""
    )

    prompt = PROMPT.format(
        demonstration="\n\n---\n\n".join(prompt_demos), user=prompt_user)
    while len(tokenizer(prompt)['input_ids']) if tokenizer else len(word_tokenize(prompt)) > max_position_embeddings - 64 and prompt_demos:
        prompt_demos = prompt_demos[:-1]
        prompt = PROMPT.format(
            demonstration="\n\n---\n\n".join(prompt_demos), user=prompt_user)
    return prompt


def unpack_single_generation(args_item):
    d, p, schema = args_item
    alignments = [(unpack_alignment(pred[0].strip()), pred[1]) for pred in p]
    d['alignment']['pred'] = alignment_consistency(alignments)
    d['alignment']['pred'] = span_alignment_fix(
        d['question'], d['alignment']['pred'], schema)
    d['prediction'] = p[0][0].strip()
    return d


if __name__ == '__main__':
    from utils.selector import select_multiple
    from utils.generator import generate_with_llm
    from utils.database import pack_db_path, database_to_string, extract_schema, remove_on_clause

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name_or_path", type=str, help="llm path")
    parser.add_argument("--config_file", type=str, help="config path")
    parser.add_argument("--train_data_file", type=str, help="data path")
    parser.add_argument("--train_database_path",
                        type=str, help="database path")
    parser.add_argument("--train_schema_file", type=str, help="schema file")
    parser.add_argument("--dev_data_file", type=str, help="data path")
    parser.add_argument("--dev_database_path", type=str, help="database path")
    parser.add_argument("--dev_schema_file", type=str, help="schema file")
    parser.add_argument("--dump_file", type=str, help="dump path")
    parser.add_argument("--data_size", type=int, help="data size")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")
    parser.add_argument("--shot", type=int, default=3)
    args = parser.parse_args()
    set_seed(args.random_seed)

    with open(args.train_data_file, 'r', encoding='utf-8') as f:
        data_train = json.load(f)
    with open(args.dev_data_file, 'r', encoding='utf-8') as f:
        data_dev = json.load(f)
        for d in data_dev:
            d['query_pred'] = d['prediction']['query']
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
    data_demo_pairs = [(data, demos, args, schemas_train, schemas_dev, EXAMPLE, PROMPT, pack_db_path, database_to_string, pack_alignment, max_position_embeddings, tokenzier)
                       for data, demos in zip(data_dev, demonstrations)]
    prompts = process_map(generate_prompt, data_demo_pairs,
                          desc='Generating Prompt', chunksize=1)
    print(prompts[0])

    predictions = generate_with_llm(
        args.llm_name_or_path, prompts, config, 'chat')
    data_and_predictions = [(d, p, schemas_dev[d['db_id']])
                            for d, p in zip(data_dev, predictions)]
    data_dev = process_map(unpack_single_generation, data_and_predictions,
                           chunksize=1, desc="Unpacking Generation")

    if 'gold' in d['alignment']:
        print(json.dumps(evaluate(data_dev), ensure_ascii=False, indent=4))
    with open(args.dump_file, 'w', encoding='utf-8') as f:
        json.dump(data_dev, f, ensure_ascii=False, indent=4)
