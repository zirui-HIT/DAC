import os
import time
import subprocess

from typing import Dict, List
from func_timeout import func_timeout, FunctionTimedOut


def fix_answer(function: str, table: List[List[str]] = None) -> str:
    function_sentences = function.split('\n')
    if function.strip().startswith("def"):
        function_sentences = function_sentences[1:]
    if not function_sentences[0].startswith(" "):
        function_sentences = [f"    {x}" for x in function_sentences]
    for i, line in enumerate(function_sentences):
        if not line.startswith(' '):
            function_sentences = function_sentences[:i]
            break
    function = '\n'.join(function_sentences)
    function = f'def solver(table):\n{function}'
    if table:
        function = f"""
{function}

table = {repr(table)}
function_result = solver(table)
""".strip()
    return function


def parse_answer(program: str, time_out: float = 5) -> str:
    def run_exec(program: str) -> str:
        try:
            local_scope = {}
            exec(program, {}, local_scope)
            return str(local_scope.get("function_result", None))
        except Exception as e:
            return f"Error occurred: {e}"

    try:
        return func_timeout(time_out, run_exec, args=(program,))
    except FunctionTimedOut:
        return "Error: Execution time exceeded the limit"
    except Exception as e:
        return f"Error occurred: {e}"


def run_command_wait_file(cmd: str, dump_file: str):
    def get_modification_time(filepath):
        return os.path.getmtime(filepath)

    def wait_for_file_modification(filepath, initial_mod_time=None):
        if initial_mod_time is None:
            initial_mod_time = get_modification_time(filepath)
        count = 0
        while count < 60:
            current_mod_time = get_modification_time(filepath)
            if current_mod_time != initial_mod_time:
                print("File has been modified.")
                break
            time.sleep(1)
            count += 1

    if not os.path.exists(dump_file):
        open(dump_file, 'w').close()
    initial_mod_time = get_modification_time(dump_file)
    subprocess.run(cmd, shell=True)
    wait_for_file_modification(dump_file, initial_mod_time)


def execute_command(cmd: str):
    """
    Execute a Linux command and print the output in real-time.

    Parameters:
    cmd (str): The Linux command to execute.
    """
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    rc = process.poll()
    return rc
