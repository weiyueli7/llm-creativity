import os
import json
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import google.generativeai as genai
import time
from tqdm import tqdm
from google.cloud import storage
import torch.nn.functional as F
import datetime
import io

# Load configuration from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

os.environ['HF_DATASETS_OFFLINE'] = '1'

OUTPUT_DIR = config["output_dir"]
LOG_FILE = config["log_file"]
BUCKET_NAME = config["bucket_name"]  
SECRETS_PATH = config["secrets_path"] 


def get_storage_client():
    return storage.Client.from_service_account_json(SECRETS_PATH)

def upload_to_gcp(bucket_name, blob_name, data, is_binary=False):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data, indent=4), content_type='application/json')
    logging.info(f"Uploaded file to GCS: {bucket_name}/{blob_name}")

# Ensure the output directory exists
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# Set up logging
# log_file_path = os.path.join(OUTPUT_DIR, LOG_FILE)
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler(log_file_path)
#     ]
# )
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def find_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:2'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")
    return device

# Set device to MPS if available, otherwise fallback to CPU
device = find_device() #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = 'cuda'
logging.info(f"Using device: {device}")

class LLMAgent:
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt,format_string):
        self.model_name = model_name
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.format_string = format_string
        # self.agent_role_prompt = f"You are a {self.agent_role} whose specialty is {self.agent_speciality}. {agent_role_prompt}. {format_string}"
        self.agent_role_prompt = f"You are a {self.agent_role} whose specialty is {self.agent_speciality}. {agent_role_prompt}."

        logging.info(f"Initializing agent: {self.agent_name}, Role: {self.agent_role}, Specialty: {self.agent_speciality}")
        
        if self.agent_name == "Gemini Agent - Creative Writer":
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.model = genai.GenerativeModel(self.model_name)
            self.tokenizer = None
            logging.info(f"Gemini model '{self.model_name}' initialized successfully.")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
            self.model.to(device)
            logging.info(f"Model '{self.model_name}' initialized successfully on {device} with float16 precision.")

    # def declare_role(self):
    #     logging.info(f"{self.agent_name}: {self.agent_role}")
    #     logging.info(f"Specialty: {self.agent_speciality}")
    #     logging.info(f"Role Prompt: {self.agent_role_prompt}")

    def generate_response(self, prompt, max_token=config["max_token"]):
        # self.declare_role()
        prompt = f"Write a science fiction story with the following prompt: {prompt}"
        logging.info(f"Prompt: {prompt}")
        # prompt = self.agent_role_prompt + prompt
        inputs = self.tokenizer(prompt, return_tensors="pt") if self.tokenizer else prompt

        # Move inputs to the same device as the model (MPS if available)
        inputs = {key: value.to(device) for key, value in inputs.items()} if self.tokenizer else inputs

        logging.info(f"Generating response using {self.agent_name}")

        try:
            if self.agent_name == "Gemini Agent - Creative Writer":
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.6, max_output_tokens=max_token),
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                    ]
                )
                response_text = response.text
                logits = None
            else:
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_token,
                    num_return_sequences=1,
                    top_p=0.9,
                    temperature=0.6,
                    pad_token_id=self.model.config.eos_token_id[0],
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_logits=True
                )
                
                input_length = inputs["input_ids"].shape[1]
                generated_ids = outputs.sequences[:, input_length:]
                response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                if isinstance(outputs.logits, tuple):
                    logits = torch.cat(outputs.logits, dim=0)  # Replace `dim` based on your needs
                else:
                    logits = outputs.logits
                    
                probabilities = F.softmax(torch.tensor(logits), dim=-1)  # Shape: (sequence_length, vocab_size)
                max_probs, _ = torch.max(probabilities, dim=-1)  # choose max_probs
                entropy_max = -torch.sum(max_probs * torch.log2(max_probs)).item()
                
                entropy_per_time_step = -torch.sum(probabilities * torch.log2(probabilities), dim=-1)  # (sequence_length)
                entropy_mean = entropy_per_time_step.mean().item()
                
                # if isinstance(outputs.logits, tuple):
                #     outputs_tensor = torch.cat(outputs.logits, dim=0)  # Replace `dim` based on your needs
                # else:
                #     outputs_tensor = outputs.logits
                # response_text = self.tokenizer.decode(outputs[0][0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                # logits = outputs_tensor.cpu().detach().numpy()

            # logging.info(f"Generated response: {response_text[:100]}...")
            # if logits is not None:
            #     logging.info(f"Logits shape: {logits.shape if isinstance(logits, np.ndarray) else 'Not available'}")

            # return response_text, logits
            return response_text, logits, entropy_max, entropy_mean

        except Exception as e:
            logging.error(f"Error during response generation: {str(e)}")
            return None, None

class SciFiStoryGenerator:
    def __init__(self, prompts_file, output_dir=OUTPUT_DIR):
        self.prompts_file = prompts_file
        self.output_dir = output_dir
        self.prompts = self.load_prompts()
        
        llama_agent = LLMAgent(
            model_name=config["llama_model_name"],
            agent_name=config["agent_name"], 
            agent_role=config["agent_role"], 
            agent_speciality=config["agent_speciality"], 
            agent_role_prompt=config["agent_role_prompt"],
            format_string=config['format_string']
        )
        
        self.agents = [llama_agent]

    def load_prompts(self):
        try:
            with open(self.prompts_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Error: The file '{self.prompts_file}' was not found.")
            return {}

    def generate_stories(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for agent in self.agents:
            agent_folder = os.path.join(self.output_dir, agent.agent_name.replace(" ", "_"))
            if not os.path.exists(agent_folder):
                os.makedirs(agent_folder)

            for category, prompts in tqdm(self.prompts.items()):
                category_folder = os.path.join(agent_folder, category.replace(" ", "_"))
                if not os.path.exists(category_folder):
                    os.makedirs(category_folder)

                for i, prompt in tqdm(enumerate(prompts)):
                    logging.info(f"Processing prompt {i+1} under category '{category}' for agent {agent.agent_name}")
                    response_text, logits, entropy_max, entropy_mean = agent.generate_response(prompt)
                    
                    final_result = {
                        "question": prompt,
                        "answer": [response_text],
                        "entropy_max": entropy_max.item() if isinstance(entropy_max, torch.Tensor) else entropy_max,
                        "entropy_mean": entropy_mean.item() if isinstance(entropy_mean, torch.Tensor) else entropy_mean
                    }

                    # response_filename = os.path.join(category_folder, f"story_{i+1}.txt")
                    # with open(response_filename, "w") as f:
                    #     f.write(f"Prompt: {prompt}\n\n")
                    #     f.write(f"Generated Story: {response}\n")

                    # logits_filename = os.path.join(category_folder, f"story_{i+1}_logits.npy")
                    # np.save(logits_filename, logits)

                    # logging.info(f"Story saved to {response_filename}")
                    # logging.info(f"Logits saved to {logits_filename}")
                    
                    response_filename = f"../../Results/Single_Agent/Output/{category.replace(' ', '_')}_story_{i+1}.json"
                    logits_filename = f"../../Results/Single_Agent/Logits/{category.replace(' ', '_')}_story_{i+1}_logits.npy"
                    
                    final_result['logits_file'] = logits_filename

                    # response_filename = response_filename[6:]
                    # logits_filename = logits_filename[6:]
                    
                    storage_client = get_storage_client()
                    bucket = storage_client.bucket(BUCKET_NAME)

                    # save logits
                    # Move logits to CPU and convert to NumPy before saving
                    logits_np = np.array(logits.detach().cpu().numpy())
                    
                    with io.BytesIO() as byte_io:
                        np.save(byte_io, logits_np)  # Save the NumPy array into a byte stream
                        byte_io.seek(0)  # Rewind the byte stream to the beginning

                        # Upload the byte stream to GCS
                        blob = bucket.blob(logits_filename[6:])  # Remove the "../../" prefix from the filename
                        blob.upload_from_file(byte_io, content_type='application/octet-stream')

                    print(f"Saved Logits to GCS bucket nlp_result with filename {logits_filename[6:]}")

                    blob = bucket.blob(response_filename[6:])
                    blob.upload_from_string(json.dumps([final_result], indent=4), content_type='application/json')
                    
                    print(f"Saved Logits to GCS bucket nlp_result with filename {response_filename[6:]}")
            
                    # if logits is not None:
                    #     logits_np = np.array(logits)
                    #     upload_to_gcp(BUCKET_NAME, logits_filename, logits_np.tobytes(), is_binary=True)

                    # logging.info(f"Story saved to GCS: {response_filename}")
                    # logging.info(f"Logits saved to GCS: {logits_filename}")
    
if __name__ == "__main__":
    story_generator = SciFiStoryGenerator(config["prompts_file"])
    story_generator.generate_stories()
