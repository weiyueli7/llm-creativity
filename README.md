# Towards Artistic Intelligence: Enhancing LLM Creativity Through Multi-Agent Feedback

<a href='https://drive.google.com/file/d/1gQ4WJI7yYqI93ohbpQYXv0qV49RNGB0g/view?usp=sharing'>< img src='https://img.shields.io/badge/Poster-PDF-red'> <a href='https://github.com/weiyueli7/llm-creativity/blob/main/datasets/SciFi/scientific_writing.json'>< img src='https://img.shields.io/badge/SciFi-Dataset-yellow'>


This repository contains the code for the paper "Towards Artistic Intelligence: Enhancing LLM Creativity Through Multi-Agent Feedback". We propose a novel approach to enhance the creativity of large language models (LLMs) by incorporating multi-agent feedback. We evaluate our approach on the task of scientific writing using using LLM-as-a-judge and our rule-based evaluation framework and show that our approach significantly improves the creativity of LLMs compared to the baseline frameworks.



## Abstract
Large Language Models (LLMs) have achieved remarkable success in various downstream tasks but often struggle with generating creative responses to open-ended questions. To improve LLM creativity in writing tasks, we propose frameworks inspired by structured human discussions, such as lectures and peer reviews, to foster creativity. Our two multi-phase discussion frameworks: 1) *LLM Teacher*, where a teacher agent provides feedback and fosters collaboration by sharing student responses, and 2) *LLM Review*, where each round of peer review sessions provide fresh insights and constructive feedback. Additionally, we present the first dataset for evaluating LLMs' creative writing in science fiction genre and assess our methods using LLM-based and rule-based evaluations. The results demonstrate that our frameworks outperform existing multi-LLM frameworks and single-LLM approaches in creativity metrics.




## Getting Started🚀

#### Installation
To install the required dependencies, run:

```bash
conda create -n llm-creativity python=3.10
conda activate llm-creativity
pip install -r requirements.txt
```

#### Pre-requisites
We used Google Cloud Platform (GCP) bucket for storing the generated responses. To use GCP, you need to:
- create a GCP account
- create a bucket called 'nlp-results' or any other name you prefer (need to replace the bucket name in the code)
- create a service account and download the JSON key file saved as `secrets.json` in the root directory of this project
Your project structure should look like this:
```
    |-Datasets
    |-Evaluation
    |-Experiments
    secrets.json
    requirements.txt
```

## Run

- To run experiments associated with LLM Teacher and LLM Review, refer to [experiments/README.md](experiments/README.md).

- To evaluate the generated responses using LLM-as-a-Judge and Rule-based-Evaluation, refer to [evaluations/README.md](evaluations/README.md).

- To get the results of the experiments from GCP bucket, simply run:
```bash
python get_results.py
```

## Acknowledgement🙏

We built our project on top of the [LLM Discussion](https://github.com/lawraa/LLM-Discussion) repository. Using their code means adhering to their license.

