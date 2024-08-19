# DAC: Decomposed Automation Correction for Text-to-SQL

This repository contains code for the paper ["DAC: Decomposed Automation Correction for Text-to-SQL"](https://arxiv.org/abs/2408.08779).

## Build Environment
```bash
conda create -n dac python=3.9 -y
conda activate dac
pip install requirements.txt
```

Download and put the [Spider](https://drive.google.com/u/0/uc?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download), [Bird](https://bird-bench.github.io) and [KaggleDBQA](https://github.com/Chia-Hsuan-Lee/KaggleDBQA) databases in ./dataset

Implement your openai-key in [utils/generator.py](utils/generator.py) if you want to use openai to generate demonstrations.

## Text-to-SQL
Use the script of [run.sh](./run.sh) to generate SQL with our DAC method.

## Evaluate
It is recommanded to evaluate the result with [https://github.com/taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval).
