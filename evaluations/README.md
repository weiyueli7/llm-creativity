# Evaluation Pipelines

This directory contains two evaluation pipelines: LLM-as-a-Judge and Rule-based-Evaluation.

## LLM-as-a-Judge

We use the `gemini-1.5-flash` model to evaluate the generated responses.

### Usage
```bash
cd evaluations/LLM-as-a-Judge
sh run.sh
```


## Rule-based-Evaluation

We define a rule-based evaluation pipeline to evaluate the generated responses.

### Usage

See [Rule-Based Evaluation](Rule_based/README.md) for more details.

