import os
import time
import openai
import torch

from tqdm import tqdm
from copy import deepcopy
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor


class GPTAzureChatGenerator(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.error_types = {
            "continue_error": [
                "timed out",
                "Connection error",
                "Connection reset by peer",
                "Remote end closed connection without response",
                "occurred in violation of protocol",
                "Failed to resolve",
                "TLSV1_ALERT_INTERNAL_ERROR",
                "Error communicating",
                "The server is overloaded or not ready yet",
                "upstream_error",
                "new_api_error",
                "当前分组上游负载已饱和",
                "Lock wait timeout exceeded"
            ],
            "sleep_error": [
                "call rate limit",
                "token rate limit"
            ],
            "ignore_error": [
                "content",
                "reduce the length"
            ]
        }

    def generate_single(self, packed_data: List[tuple]) -> List[Tuple[str, float]]:
        from openai import AzureOpenAI
        from openai.types.chat import ChatCompletion

        sentence, engine, config = packed_data
        client = AzureOpenAI(
            api_version="your_api_version",
            azure_endpoint="your_checkpoint",
            api_key="your_api_key"
        )

        while True:
            try:
                completion: ChatCompletion = client.chat.completions.create(
                    model=engine,
                    messages=[{"role": "user", "content": sentence}],
                    **config)
                return [(x.message.content, 1.0) for x in completion.choices]
            except Exception as e:
                continue_flag = False
                sleep_flag = False
                ignore_flag = False
                for x in self.error_types['continue_error']:
                    if x in str(e):
                        continue_flag = True
                for x in self.error_types['sleep_error']:
                    if x in str(e):
                        sleep_flag = True
                        continue_flag = True
                for x in self.error_types['ignore_error']:
                    if x in str(e):
                        ignore_flag = True
                if sleep_flag:
                    time.sleep(5)
                if continue_flag:
                    continue
                if not ignore_flag:
                    print(e)
                return [""]

    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        config = deepcopy(config)
        if config['parallel']:
            config.pop('parallel')
            if 'batch_size' in config:
                config.pop('batch_size')
            packed_data = [(x, self.model_name, config) for x in source]
            with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as _:
                result: List[List[str]] = list(process_map(
                    self.generate_single, packed_data, max_workers=os.cpu_count() // 2, chunksize=1))
        else:
            config.pop('parallel')
            result: List[List[str]] = [self.generate_single(
                (x, self.model_name, config)) for x in tqdm(source)]
        return result


class GPTOpenAIChatGenerator(GPTAzureChatGenerator):
    def generate_single(self, packed_data: List[tuple]) -> List[Tuple[str, float]]:
        openai.api_key = "your_api_key"
        openai.api_base = "your_api_base"

        sentence, engine, config = packed_data
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[{"role": "user", "content": sentence}],
                    **config
                )
                return [(c.message['content'].strip(), 1.0) for c in response.choices]
            except Exception as e:
                continue_flag = False
                sleep_flag = False
                ignore_flag = False
                for x in self.error_types['continue_error']:
                    if x in str(e):
                        continue_flag = True
                for x in self.error_types['sleep_error']:
                    if x in str(e):
                        sleep_flag = True
                        continue_flag = True
                for x in self.error_types['ignore_error']:
                    if x in str(e):
                        ignore_flag = True
                if sleep_flag:
                    time.sleep(5)
                if continue_flag:
                    continue
                if not ignore_flag:
                    print(e)
                return [("", 0.0)]


class LlamaGenerator(object):
    def __init__(self, model_name_or_path: str):
        def check_cuda_gt_8() -> bool:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_properties = torch.cuda.get_device_properties(i)
                compute_capability = float(
                    f"{device_properties.major}.{device_properties.minor}")
                if compute_capability < 8.0:
                    return False
            return True

        self.llm = LLM(model=model_name_or_path,
                       tensor_parallel_size=torch.cuda.device_count(),
                       dtype="auto" if check_cuda_gt_8() else "float",
                       trust_remote_code=True)
        self.tokenizer = self.llm.get_tokenizer()

    def batch_data(self, data_list: List[str], batch_size: int) -> List[List[str]]:
        n = len(data_list) // batch_size
        batch_data = []
        for i in range(n-1):
            start = i * batch_size
            end = (i+1) * batch_size
            batch_data.append(data_list[start:end])
        last_start = (n-1) * batch_size
        batch_data.append(data_list[last_start:])
        return batch_data

    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        too_long_data_count = 0
        source_filtered = []
        for i, x in tqdm(enumerate(source), total=len(source), desc="filtering too long input"):
            if len(self.tokenizer(x)['input_ids']) > self.llm.llm_engine.model_config.max_model_len:
                source[i] = "TL;NR"
                too_long_data_count += 1
            else:
                source_filtered.append(x)
        print(f"too long input count: {too_long_data_count}")
        if config['ignore_too_long']:
            source = source_filtered

        sampling_params = SamplingParams(
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_tokens=config['max_tokens'],
            n=config['n'],
            logprobs=1,
            stop=config['stop']
        )

        res_completions = []
        batch_size = config['batch_size']
        while batch_size > 0:
            try:
                res_completions = []
                batch_instances = self.batch_data(
                    source, batch_size=batch_size)
                for _, prompt in tqdm(enumerate(batch_instances), total=len(batch_instances), desc="generating"):
                    if not isinstance(prompt, list):
                        prompt = [prompt]
                    completions = self.llm.generate(
                        prompt, sampling_params, use_tqdm=False)
                    for output in completions:
                        generated_text = []
                        for x in output.outputs:
                            total_logprob = 0.0
                            for t in x.logprobs:
                                max_logprob_token = max(
                                    t.items(), key=lambda x: x[1].logprob)
                                if max_logprob_token[0] == 13:
                                    break
                                total_logprob += max_logprob_token[1].logprob
                            generated_text.append(
                                (x.text.lstrip('\n'), total_logprob))
                        res_completions.append(generated_text)
                break  # If generation is successful, break out of the loop
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Halve the batch size if out of memory
                    batch_size = int(batch_size / 2)
                    print(
                        f"Reducing batch size due to memory constraints: new batch size is {batch_size}")
                    torch.cuda.empty_cache()  # Clear memory cache if using PyTorch
                else:
                    raise e  # Reraise if it's an unrelated error
            if batch_size < 1:
                raise ValueError("Batch size cannot be reduced further")

        return res_completions


class LlamaChatGenerator(LlamaGenerator):
    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        messages_list = [[{"role": "user", "content": x}] for x in source]
        source = [self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False) for messages in messages_list]
        return super().generate(source, config)


MODEL_MAP: Dict[str, object] = {
    "llama": {
        'text': LlamaGenerator,
        'chat': LlamaChatGenerator
    },
    "deepseek": {
        'text': LlamaGenerator,
        'chat': LlamaChatGenerator
    },
    "glm": {
        'text': LlamaGenerator,
        'chat': LlamaChatGenerator
    },
    "qwen": {
        'text': LlamaGenerator,
        'chat': LlamaChatGenerator
    },
    "gpt": {
        'text': GPTOpenAIChatGenerator,
        'chat': GPTOpenAIChatGenerator
    }
}


def generate_with_llm(model_name_or_path: str, source: List[str], config: Dict[str, Any], mode: str = 'text') -> List[List[Tuple[str, float]]]:
    generator = detect_generator(model_name_or_path, mode)
    results = generator.generate(source, config)
    del generator
    return results


def detect_generator(model_name_or_path: str, mode: str = 'text') -> object:
    for token in MODEL_MAP:
        if token in model_name_or_path.lower():
            return MODEL_MAP[token][mode](model_name_or_path)


def consistency(answers: List[Tuple[str, Any, float]]) -> Tuple[str, Any]:
    count: Dict[str, float] = {}
    record: Dict[str, Tuple[str, str]] = {}
    for a, b, c in answers:
        x = str(b)
        if "error" in x.lower():
            continue
        if x not in count:
            count[x] = 0
            record[x] = (a, b)
        count[x] += c
    if not count:
        return "", ""
    return record[max(count, key=lambda x: count[x])]


def consistency_with_error(answers: List[Tuple[str, Any, float]]) -> Tuple[str, Any]:
    count: Dict[str, float] = {}
    record: Dict[str, Tuple[str, str]] = {}
    for a, b, c in answers:
        x = str(b)
        # if "error" in x.lower():
        #     continue
        if x not in count:
            count[x] = 0
            record[x] = (a, b)
        count[x] += c
    if not count:
        return "", ""
    return record[max(count, key=lambda x: count[x])]
