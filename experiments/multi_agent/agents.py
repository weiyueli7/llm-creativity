import json
import os
import logging
import subprocess
import time
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch.nn.functional as F

class Agent:
    def generate_answer(self, answer_context):
        raise NotImplementedError("This method should be implemented by subclasses.")
    def construct_assistant_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")
    def construct_user_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")

    
def find_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:1'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")
    return device
        
class LlamaAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, speaking_rate, agent_type,
                 agent_role_prompt=None, agent_role_prompt_initial=None, agent_role_prompt_followup=None, general_instruction=None):
        self.model_name = model_name
        self.agent_name = agent_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        
        self.device = find_device()
        if torch.cuda.device_count() > 1:
            # print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        # self.device = 'cuda'
        self.model.to(self.device)
        self.agent_role = agent_role
        self.agent_type = agent_type
        self.agent_speciality = agent_speciality
        self.speaking_rate = speaking_rate
        self.agent_role_prompt = agent_role_prompt
        self.agent_role_prompt_initial = agent_role_prompt_initial
        self.agent_role_prompt_followup = agent_role_prompt_followup 
        self.general_instruction = general_instruction
        
    def generate_answer(self, answer_context, token_limit=1000, final_round=False, 
                        temp=0.6, top_p=0.9, top_k=50, rep_pen=1.0):
        inputs = self.tokenizer(answer_context[0]['content'], return_tensors="pt")

        # Move inputs to the same device as the model (MPS if available)
        inputs = {key: value.to(self.device) for key, value in inputs.items()} if self.tokenizer else inputs
        
        # Check if model is wrapped in DataParallel
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        # inputs = self.tokenizer(answer_context, return_tensors="pt")
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=token_limit,
            num_return_sequences=1,
            pad_token_id=model.config.eos_token_id[0],
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            # temperature=temp,
            # top_p=top_p,
            # top_k=top_k,
            # repetition_penalty=rep_pen,
        )
        
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[:, input_length:]
        response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Save logits if it's the final round and the agent is a student writer
        if final_round and "Writer" in self.agent_role:
            if isinstance(outputs.logits, tuple):
                logits = torch.cat(outputs.logits, dim=0)  # Replace `dim` based on your needs
            else:
                logits = outputs.logits
            
            probabilities = F.softmax(torch.tensor(logits), dim=-1)  # Shape: (sequence_length, vocab_size)
            max_probs, _ = torch.max(probabilities, dim=-1)  # choose max_probs
            entropy_max = -torch.sum(max_probs * torch.log2(max_probs)).item()
            
            entropy_per_time_step = -torch.sum(probabilities * torch.log2(probabilities), dim=-1)  # (sequence_length)
            entropy_mean = entropy_per_time_step.mean().item()
            return response_text, logits, entropy_max, entropy_mean
            
        return response_text
    
    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}
    

class MistralAgent(Agent):

    def __init__(self, model_name, agent_name, agent_role, agent_speciality, speaking_rate, agent_type,
                 agent_role_prompt=None, agent_role_prompt_initial=None, agent_role_prompt_followup=None, general_instruction=None):
        self.model_name = model_name
        self.agent_name = agent_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        # Set device
        self.device = 'cuda'  # Assuming you're using GPU
        self.model.to(self.device)
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.speaking_rate = speaking_rate
        self.agent_type = agent_type
        self.agent_role_prompt = agent_role_prompt
        self.agent_role_prompt_initial = agent_role_prompt_initial
        self.agent_role_prompt_followup = agent_role_prompt_followup
        self.general_instruction = general_instruction

    def generate_answer(self, answer_context, token_limit=1000, final_round=False):
        # Tokenize input
        # inputs = self.tokenizer(answer_context[0]['content'], return_tensors="pt")
        encoded = self.tokenizer.apply_chat_template(answer_context, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        inputs = encoded.to(self.device)
        self.model.to(self.device)
        
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate output using Mistral model
        outputs = self.model.generate(
            # inputs,
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=token_limit,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.6,
            pad_token_id=self.model.config.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            do_sample=True
        )

        # Return the decoded answer
        generated_ids = outputs.sequences.tolist()
        # print(len(generated_ids[0]))
        full_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        split_response = full_response.split("assistant\n", -1)
        response_text = split_response[1].strip() if len(split_response) > 1 else full_response
        print(response_text)
        # print(len(response_text))
        
        # Save logits if it's the final round and the agent is a student writer
        if final_round and "Writer" in self.agent_role:
            if isinstance(outputs.logits, tuple):
                logits = torch.cat(outputs.logits, dim=0)  # Replace `dim` based on your needs
            else:
                logits = outputs.logits
                
            # compute entropy
            probabilities = F.softmax(torch.tensor(logits), dim=-1)  # Shape: (sequence_length, vocab_size)
            max_probs, _ = torch.max(probabilities, dim=-1)  # choose max_probs
            entropy_max = -torch.sum(max_probs * torch.log2(max_probs)).item()
            
            entropy_per_time_step = -torch.sum(probabilities * torch.log2(probabilities), dim=-1)  # (sequence_length)
            entropy_mean = entropy_per_time_step.mean().item()
            
            return response_text, logits, entropy_max, entropy_mean

        return response_text

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}

    def construct_user_message(self, content):
        return {"role": "user", "content": content}