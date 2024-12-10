# Rule-Based Evaluation

This folder provides tools for performing rule-based evaluation on various results, including single agent results, multi-discussion results, teacher framework results, and review framework results.

## Workflow

### Step 1: Load Reference Distribution
Run the `get_q_dist.py` script to generate the reference distribution.  
This will produce the `q_dist.json` file.

```bash
python get_q_dist.py
```
### Step 2: Load Book Embeddings

Run the `load_book_embeddings.py` script to load the book embeddings from the SFGram Dataset.  
This process will produce the `all_embeddings.npy` file.

```bash
python load_book_embeddings.py
```

### Step 3 load Experiments Results
run the `get_results.py` to get the experiments stored on the GCP buckets. The source folder and the destination folder for each experiments are specified in the `get_results.py`.
```bash
python get_q_dist.py
```

### Step 4: Perform Evaluation

Run the `eval.py` script to compute and generate the evaluation results.  
The output will be saved in the `statistics.json` file, which summarizes the rule-based evaluation results.

```bash
python eval.py
```
### Decoding Experiments
#### Step1 Evaluation
Run the `eval_{Rep_Pen, temp, topk, topp}.py` scripts for evaluation with various parameters (`Rep_Pen`, `temp`, `topk`, `topp`).  
The results will be stored in `statistics_{Rep_Pen, temp, topk, topp}.json`.

```bash
python eval_Rep_Pen.py
python eval_temp.py
python eval_topk.py
python eval_topp.py
```
#### Step2 Plot the evaluation result
Run the `draw_{Rep_Pen, temp, topk, topp}.py` scripts to visualize the rule-based evaluation results.  
The plots will depict the evaluation metrics for different parameter settings (`Rep_Pen`, `temp`, `topk`, `topp`).

```bash
python draw_Rep_Pen.py
python draw_temp.py
python draw_topk.py
python draw_topp.py
```