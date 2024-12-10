import os
import json
import re
from google.cloud import storage
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import argparse
import logging
import time
import google.generativeai as genai

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../secrets.json"

logging.basicConfig(level=logging.INFO)

class GeminiJudge:
    def __init__(self, model_name):
        """
        Initialize the LLM judge using the Gemini API.

        :param model_name: Name of the model to use.
        """
        self.model_name = model_name
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(self.model_name)

    def evaluate(self, prompt, temperature=1.0):
        """
        Generate a judgment for the given prompt using the model.

        :param prompt: The evaluation prompt to pass to the model.
        :param temperature: Sampling temperature for response generation.
        :return: Generated response as text.
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            )
            return response.text
        except Exception as e:
            logging.exception("Exception occurred during response generation: " + str(e))
            time.sleep(1)
            return self.evaluate(prompt)

def find_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    logging.info(f"Using device: {device}")
    return device

def initialize_local_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.float16)
    model.to(device)
    return tokenizer, model

def evaluate_with_local_model(story, criterion_description, tokenizer, model):
    prompt = f"""
    You are an expert evaluator for creative writing. The evaluation rubric is as follows:
    {criterion_description}
    Score the following story on a scale from 1 to 10, return the score in the format of SCORE: [[score]]. Provide your reasoning for the score.
    
    STORY:
    {story}

    SCORE:
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs.to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        max_length=1024
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def parse_score(response):
    score_patterns = [
        r"SCORE:\s*\[\[(\d+\.?\d*)\]\]",  # [[score]]
        r"SCORE:\s*\[(\d+\.?\d*)\]",      # [score]
        r"SCORE:\s*(\d+\.?\d*)",          # score
        r"Final score:\s*(\d+\.?\d*)"     # Final score
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    return None

def evaluate_story(story, rubric, use_gemini, llm_judge=None, tokenizer=None, model=None, rounds=3):
    results = {}
    for criterion in rubric["criteria"]:
        prompt = f"""
        You are an expert evaluator for creative writing. The evaluation rubric is as follows:
        {criterion["description"]}
        Score the following story on a scale from 1 to 10, return the score in the format of SCORE: [[score]]. Provide your reasoning for the score.
        
        STORY:
        {story}

        SCORE:
        """
        scores = []
        for _ in range(rounds):
            if use_gemini:
                response = llm_judge.evaluate(prompt)
            else:
                response = evaluate_with_local_model(story, criterion["description"], tokenizer, model)
            score = parse_score(response)
            if score is not None:
                scores.append(score)
        score = sum(scores) / len(scores) if scores else None
        results[criterion["id"]] = score
    return results

def download_json_files(bucket_name, prefix, local_dir):

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    file_paths = []
    for blob in blobs:
        if blob.name.endswith('.json'):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            file_paths.append(local_path)
    return file_paths

def upload_results(bucket_name, prefix, results, original_path, local_dir, results_dir="Eval_Results"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    original_path = original_path.replace(local_dir, "")[1:]
    original_path = os.path.join(prefix, original_path)
    prefix_dir = prefix.split("/")[0]
    output_path = original_path.replace(prefix_dir, results_dir)

    blob = bucket.blob(output_path)
    blob.upload_from_string(json.dumps(results, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Evaluate stories using local model or Gemini API.")
    parser.add_argument("--use_gemini", default=True, help="Use Gemini API for evaluation (default: False).")
    parser.add_argument("--gemini_model", default="gemini-1.5-flash", help="Gemini model name to use (default: gemini-1.5-turbo).")
    parser.add_argument("--rounds", default=3, type=int, help="Number of evaluation rounds to average scores over (default: 3).")
    parser.add_argument("--bucket_name", default="nlp_result", help="Name of the GCS bucket to download files from.")
    parser.add_argument("--prefix", default="Results_v3/Single_Agent/Output", help="Prefix of the GCS bucket to download files from.")
    parser.add_argument("--local_dir", default="./temp_jsons", help="Local directory to download files to.")
    parser.add_argument("--results_dir", default="Eval_Results_v2", help="Local directory to save evaluated results to.")
    args = parser.parse_args()

    use_gemini = args.use_gemini
    tokenizer, model, llm_judge = None, None, None

    rounds = args.rounds

    if use_gemini:
        logging.info("Using Gemini API for evaluation...")
        llm_judge = GeminiJudge(model_name=args.gemini_model)
    else:
        logging.info("Initializing local LLaMA model...")
        tokenizer, model = initialize_local_model()

    bucket_name = args.bucket_name
    prefix = args.prefix
    local_dir = args.local_dir
    results_dir = args.results_dir

    # delete the local directory if it already exists
    if os.path.exists(local_dir):
        os.system(f"rm -rf {local_dir}")

    rubric = {
        "criteria": [
            {"id": 1, "description": "Exploration of futuristic or scientific concepts, including their plausibility and integration into the story world."},
            {"id": 2, "description": "Use of speculative elements to push boundaries of imagination while maintaining internal consistency and logical coherence."},
            {"id": 3, "description": "Development of compelling and multi-dimensional characters that embody or respond to scientific or technological themes."},
            {"id": 4, "description": "Crafting of vivid and immersive settings that convincingly portray an advanced, alternate, or speculative reality."},
            {"id": 5, "description": "Integration of moral, ethical, or philosophical questions related to science and technology within the narrative."}
        ]
    }

    json_files = download_json_files(bucket_name, prefix, local_dir)
    # print(type(json_files))
    # print(json_files)
    # # stop the execution
    # os.system("exit")
    for file_path in tqdm(json_files):
        with open(file_path, 'r') as f:
            data = json.load(f)

        evaluated_data = []
        for entry in data:
            story = entry["answer"][0]
            results = evaluate_story(story, rubric, use_gemini, llm_judge, tokenizer, model, rounds)
            entry["scores"] = results
            evaluated_data.append(entry)

        upload_results(bucket_name, prefix, evaluated_data, file_path, local_dir, results_dir)

if __name__ == "__main__":
    device = find_device()
    main()
