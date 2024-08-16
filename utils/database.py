import re
import os
import json
import nltk
import time
import sqlite3
import sqlparse

from func_timeout import func_timeout
from typing import List, Dict, Any, Tuple
from sqlparse.tokens import Token
from sqlparse.sql import IdentifierList, Identifier


def extract_schema(sql: str, schema: Dict[str, Any] = None) -> List[str]:
    parsed = sqlparse.parse(sql)[0]
    identifiers = set()

    def parse_tokens(tokens, previous_token=None, aliases=set()):
        for token in tokens:
            if token.is_group:
                parse_tokens(token.tokens, previous_token, aliases)
            elif isinstance(token, Identifier):
                if previous_token and previous_token.ttype is sqlparse.tokens.Keyword and previous_token.value.upper() == 'AS':
                    aliases.add(token.get_real_name())
                else:
                    name = token.get_real_name()
                    if name and name not in aliases:
                        identifiers.add(name)
            elif token.ttype in sqlparse.tokens.Name:
                if previous_token and previous_token.ttype is sqlparse.tokens.Keyword and previous_token.value.upper() == 'AS':
                    aliases.add(token.value)
                elif token.value not in aliases:
                    identifiers.add(token.value)
            previous_token = token

    parse_tokens(parsed.tokens, set())
    results = [x for x in identifiers]
    if not schema:
        return results
    schema_names = schema['table_names_original'] + [x[1]
                                                     for x in schema['column_names_original']]
    schema_names = [x.lower() for x in schema_names]
    results = [x for x in results if x.lower() in schema_names]
    return results


def database_to_string(
    db_file: str,
    granularity: str = 'none',
    sql: str = None,
    question: str = None,
    schema: Dict[str, Any] = None,
    additional_entities: List[str] = [],
    add_value_lines: int = 16384
) -> str:
    schema_used = extract_schema(sql, schema) if sql else []
    schema_used = [x.lower() for x in schema_used] + additional_entities
    schema_used = list(set(schema_used))

    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(f"{e} : {db_file}")
        raise e
    conn.text_factory = lambda b: b.decode(errors='ignore')
    conn.isolation_level = None
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    output = ""
    for table in tables:
        table_name = table[0]
        if table_name == 'sqlite_sequence':
            continue
        if granularity in ['table', 'column'] and table_name.lower() not in schema_used:
            continue
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        columns = cursor.fetchall()

        create_statement = f"create table {table_name} (\n"
        primary_key = ""
        foreign_keys = []

        for col in columns:
            col = list(col)
            if ' ' in col[1]:
                col[1] = f"`{col[1]}`"
            if granularity == 'column' and col[1].lower() not in schema_used:
                continue
            col_desc = f"{col[1]} {col[2] if col[2] else 'TEXT'},\n"
            if col[5]:
                primary_key = f"primary key ({col[1]});\n"
            create_statement += col_desc

        cursor.execute(f'PRAGMA foreign_key_list("{table_name}");')
        foreign_keys_info = cursor.fetchall()
        for fk in foreign_keys_info:
            if not fk[3] or not fk[4]:
                continue
            fk = list(fk)
            if ' ' in fk[3]:
                fk[3] = f"`{fk[3]}`"
            if ' ' in fk[4]:
                fk[4] = f"`{fk[4]}`"
            if granularity == 'column' and (fk[3].lower() not in schema_used or fk[4].lower() not in schema_used):
                continue
            if granularity == 'table' and fk[2].lower() not in schema_used:
                continue
            fk_desc = f"foreign key ({fk[3]}) references {fk[2]}({fk[4]}),\n"
            foreign_keys.append(fk_desc)

        create_statement += "".join(foreign_keys)
        create_statement += primary_key
        output += create_statement + ')\n'

        if add_value_lines > 0:
            output += "/*\nColumns in " + table_name + \
                " and 3 distinct examples in each column:\n"
            col_names = [f"`{col[1]}`" if ' ' in col[1] else col[1]
                         for col in columns]
            cursor.execute(f'SELECT * FROM ' + '"' +
                           table_name.strip('"') + '"' + ' ORDER BY RANDOM() LIMIT ' + str(add_value_lines) + ';')
            rows = cursor.fetchall()
            col_data = {col: [] for col in col_names}
            for row in rows:
                for col_name, value in zip(col_names, row):
                    if value is not None:
                        col_data[col_name].append(
                            str(value).replace('\n', ' '))
            for col_name in col_names:
                if granularity == 'column' and col_name.lower() not in schema_used:
                    continue
                examples = col_data[col_name]
                examples_mentioned = [example for example in examples if example.lower(
                ) in question.lower()][:3] if question else []
                if len(examples_mentioned) < 3:
                    examples_mentioned += [
                        example for example in examples if example not in examples_mentioned][:3 - len(examples_mentioned)]
                examples_str = ", ".join(examples_mentioned)
                if len(examples_str) > 128:
                    examples_str = ""
                output += f"{col_name}: {examples_str};\n"
            output += "*/\n"
        output += "\n"
    cursor.close()
    conn.close()
    return output.strip()


def pack_db_path(db_path: str, db_name: str) -> str:
    return os.path.join(db_path, db_name, f"{db_name}.sqlite")


def fix_sql(sql: str, sql_order: int = -1) -> str:
    sql = sql.split(';')[0]

    # unpack markdown format
    if '```sql' in sql:
        sql = sql.split('```sql')[sql_order].split('```')[0].strip()
    if '```' in sql:
        sql = sql.split('```')[1].strip()

    sql = sql.split('<|eot_id|>')[0]
    sql = sql.strip().strip(';').strip()
    if not sql.lower().startswith('select') and not sql.lower().startswith('with'):
        sql = f"SELECT {sql}"
    sql = sql.replace('\n', ' ').replace('\t', ' ')

    # replace multiple spaces with single space
    sql = re.sub(r'\s+', ' ', sql)

    def fix_join(sql: str) -> str:
        # Helper function to replace commas with JOIN in a FROM clause
        def replace_commas(start: int) -> None:
            nonlocal sql
            paren_count = 0
            i = start
            while i < len(sql):
                if sql[i] == '(':
                    paren_count += 1
                elif sql[i] == ')':
                    paren_count -= 1
                # Replace comma with JOIN if not within parentheses
                elif sql[i] == ',' and paren_count == 0:
                    sql = sql[:i] + ' JOIN' + sql[i+1:]
                elif sql[i:i+4] == 'FROM' and paren_count == 0:
                    # Nested FROM, skip it
                    i += 4
                    continue
                elif sql[i:i+5] == 'WHERE' and paren_count == 0:
                    # Reached the end of the FROM clause
                    break
                i += 1

        # Main function logic
        i = 0
        while i < len(sql):
            # Find the FROM keyword
            from_index = sql.find('FROM', i)
            if from_index == -1:
                break  # No more FROM clauses found
            # Call the helper to replace commas starting from the end of 'FROM '
            replace_commas(from_index + 5)
            i = from_index + 5  # Update the index to continue searching

        return sql

    # sql = fix_join(sql)

    return sql


def extract_table_aliases(sql: str) -> Dict[str, str]:
    def process_identifier(identifier, aliases):
        if not isinstance(identifier, Identifier):
            return
        name = identifier.get_real_name()
        alias = identifier.get_alias()
        if name and alias:
            aliases[alias] = name

    parsed = sqlparse.parse(sql)[0]
    aliases = {}
    capture = False
    for token in parsed.tokens:
        if token.ttype is Token.Keyword and token.value.upper() in ["FROM", "JOIN"]:
            capture = True
        elif token.ttype is Token.Keyword:
            capture = False
        if capture:
            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    process_identifier(identifier, aliases)
            elif isinstance(token, Identifier):
                process_identifier(token, aliases)
    return aliases


def execute_sql(sql: str, database_path: str, timeout: float = 60) -> str:
    def execute(sql: str, database_path: str):
        try:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [description[0]
                       for description in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            result = {col: [] for col in columns}
            for row in rows:
                for col, value in zip(columns, row):
                    result[col].append(value)

            if not columns:
                result = f"Error: No Column Included in the SQL"
            else:
                result = json.dumps(result, ensure_ascii=False, indent=4)
        except Exception as e:
            result = f"Error executing SQL \"{sql}\": \"{e}\""
        finally:
            cursor.close()
            conn.close()
        return result

    try:
        return func_timeout(timeout, execute, args=(sql, database_path,))
    except:
        return "Error executing SQL: Timeout"


def remove_on_clause(tokens: list) -> Tuple[str, bool]:
    on_flag = False
    have_keyword = False
    modified_tokens: List[str] = []
    for token in tokens:
        if (token.is_keyword and token.value.lower() not in ['and', 'or']) or token.value.lower() in [')']:
            have_keyword = True
            on_flag = False

        if token.is_group:
            clause_sql, clause_on_flag = remove_on_clause(token.tokens)
            token_append = clause_sql
            if clause_on_flag:
                on_flag = False
        else:
            token_append = token.value

        if token.value.lower() == "on":
            on_flag = True
        if not on_flag:
            modified_tokens.append(token_append)
    return ''.join(modified_tokens), have_keyword


def extract_skeleton(sql: str, holder: str = '_') -> str:
    def should_replace(token):
        """Determine if the token should be replaced with '*'."""
        if token.value.upper() in ['MIN', 'MAX', 'SUM', 'AVG', 'COUNT', 'LENGTH', 'CAST']:
            return False
        if token.ttype in [Token.Name, Token.String, Token.Number] or token.ttype in Token.Literal:
            return True
        return False

    def process_token(token):
        """Process each token and decide whether to replace it or keep it."""
        if token.is_group:
            # If the token is a group (like a subquery or a function), process its tokens
            return token.__class__([process_token(t) for t in token.tokens])
        elif should_replace(token):
            # Replace the token with a '*' token
            return sqlparse.sql.Token(Token.Wildcard, holder)
        else:
            return token

    def print_tokens(tokens):
        print(json.dumps(
            [f"{token.ttype} {token.value} {token.is_group}" for token in tokens], ensure_ascii=False, indent=4))
        for token in tokens:
            if token.is_group:
                print_tokens(token.tokens)

    """Extract the skeleton from an SQL query by replacing certain tokens"""
    sql = sql.lower()
    sql, _ = remove_on_clause(sqlparse.parse(sql)[0].tokens)
    # Parse the SQL statement
    parsed = sqlparse.parse(sql)[0]
    # Process and replace tokens as needed
    modified_tokens = [process_token(token) for token in parsed.tokens]
    # Rebuild and return the modified SQL statement
    skeleton = ''.join(str(token) for token in modified_tokens)
    skeleton = ' '.join(nltk.tokenize.word_tokenize(skeleton))

    for keyword in ['current_date', 'template', 'location', 'language', 'null', 'match', 'size', 'year', 'temp', 'engine', 'owner', 'version', 'date', 'section', 'share', 'package', 'show', 'false', 'true', 'text']:
        skeleton = skeleton.replace(keyword, holder)
        skeleton = skeleton.replace(f" ( {keyword} ) ", f' ( {holder} ) ')
    for item_word in [f'( {holder} , {holder} )', f'{holder} . {holder}', f'{holder} {holder}', f'{holder}.{holder}']:
        skeleton = skeleton.replace(item_word, holder)
    for join_word in ['left join', 'right join', 'inner join', 'cross join']:
        skeleton = skeleton.replace(join_word, 'join')
    for remove_word in [f' as {holder}', ' as count']:
        skeleton = skeleton.replace(remove_word, ' ')
    skeleton = skeleton.replace('  ', ' ')
    skeleton = skeleton.replace(' is not ', ' is ')
    skeleton = skeleton.replace(
        ' order by count desc ', f' order by {holder} desc ')

    return skeleton.strip()


if __name__ == '__main__':
    with open('./dataset/KaggleDBQA/tables.json', 'r', encoding='utf-8') as f:
        schemas = {s['db_id']: s for s in json.load(f)}
    with open('./dataset/KaggleDBQA/dev.json', 'r', encoding='utf-8') as f:
        data = [d for d in json.load(f) if d['db_id'] == "USWildFires"][0]

    database = database_to_string(pack_db_path('./dataset/KaggleDBQA/database',
                                               data['db_id']), 'table', data['query'], data['question'], schemas[data['db_id']])
    print(database)
