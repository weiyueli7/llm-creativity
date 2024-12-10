import json
import re
from agents import LlamaAgent, MistralAgent
import datetime
import os
from tqdm import tqdm
import numpy as np
import torch
from google.cloud import storage
import random
import multiprocessing
import io

class Discussion:
    PROMPTS = {
        1: "You are in a group discussion with other teammates; as a result, answer as diversely and creatively as you can.",
        2: "You're in a brainstorming session where each idea leads to the next. Embrace the flow of creativity without limits, encouraging one another to build on each suggestion for unexpected connections.",
        3: "Pretend your team is at a think tank where unconventional ideas are the norm. Challenge each other to think from different perspectives, considering the most unusual or innovative ideas.",
        4: "Engage in a collaborative discussion where each of you contributes a unique insight or query, aiming to delve into uncharted territories of thought. Throughout the discussion, focus on expanding the scope and depth of each contribution through constructive feedback, counterpoints, and further questioning. The objective is to achieve a broad spectrum of ideas and solutions, promoting a culture of continuous learning and innovation.",
        5: "Envision your group as a crew on a mission to solve a mystery using only your creativity and wit. How would you piece together clues from each member's ideas to find the solution? And this would be crucial to your member’s life."
    }

    def __init__(self, dataset_file, rounds, prompt, secrets_path="../../secrets.json"):
        self.dataset_file = dataset_file
        self.rounds = rounds
        self.discussion_prompt = self.PROMPTS.get(prompt, "Invalid prompt selected.")
        self.bucket_name = "nlp_result"  # Define your GCS bucket name here
        self.secrets_path = secrets_path  # Path to the service account key
        print("Discussion initialized with dataset: {} and {} rounds.".format(dataset_file, rounds))

    def get_storage_client(self):
        """Initialize the Google Cloud Storage client with a service account key."""
        return storage.Client.from_service_account_json(self.secrets_path)

    def save_conversation(self, filename, conversation_data):
        """Save conversation data to a GCP bucket."""
        storage_client = self.get_storage_client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(filename[6:])  # Remove the "../../" prefix from the filename

        # Convert conversation data to JSON and upload
        blob.upload_from_string(json.dumps(conversation_data, indent=4), content_type='application/json')
        print(f"Saved Conversation Data to GCS bucket {self.bucket_name} with filename {filename[6:]}")

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

        
    def extract_response(self, content):
        lines = content.split('\n')
        uses = [line.strip() for line in lines if line.strip() and re.match(r"^\d+\.", line)]
        uses = [use[use.find('.') + 2:] for use in uses]
        return uses

class LLM_Debate(Discussion):
    def __init__(self, agents_config, dataset_file, rounds, task, prompt):
        super().__init__(dataset_file, rounds, prompt)
        self.task_type = task
        self.agents = self.initialize_agents(agents_config, task)
        print(f"LLM_Debate initialized for task: {task} with {len(self.agents)} agents.")
    
    def initialize_agents(self, agents_config, task_type):
        agents = []
        for config in agents_config:
            # print(task_type, config['agent_name'])
            agent_class = {"llama": LlamaAgent, "mistral": MistralAgent}.get(config['type'])
            if not agent_class:
                raise ValueError(f"Unsupported agent type: {config['type']}")

            common_args = {
                "model_name": config['model_name'],
                "agent_name": config['agent_name'],
                "agent_role": config['agent_role'],
                "agent_speciality": config['agent_speciality'],
                "speaking_rate": config['speaking_rate'],
                "agent_type": config['type'],
            }

            if task_type == 'SciFi-Teacher' and 'Teacher' in config['agent_name']:
                agent = agent_class(
                    **common_args, 
                    agent_role_prompt_initial=config['agent_role_prompt_initial'], 
                    agent_role_prompt_followup=config['agent_role_prompt_followup']
                )
            elif task_type == 'SciFi-Teacher':
                agent = agent_class(
                    **common_args, 
                    agent_role_prompt=config['agent_role_prompt'], 
                    general_instruction=config.get('general_instruction')
                )
            elif task_type == "SciFi-Dis":
                agent = agent_class(
                    **common_args, 
                    agent_role_prompt=config['agent_role_prompt'], 
                    general_instruction=config.get('general_instruction')
                )
            elif task_type == "SciFi-Review":
                agent = agent_class(
                    **common_args, 
                    agent_role_prompt=config['agent_role_prompt'], 
                    general_instruction=config.get('general_instruction')
                )
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            agents.append(agent)
        return agents

    
    def construct_response(self, question, most_recent_responses, current_agent, is_last_round=False, is_first_round=False, baseline=False):
        format_string = """
        Please follow these formatting rules for your response:

        1. **Do not repeat the prompt**: Directly start generating the requested story without restating or summarizing the given prompt.
        2. **Start with the phrase**: "Here's my story:".
        3. **End with the phrase**: "End of Story".
        4. **Only include the story text between these phrases**: Avoid adding any commentary, explanation, or notes before or after the story.

        Do not ask questions or wait for guidance. Your response must strictly adhere to this format.\n
        """
        
        # deal with first round
        if is_first_round:
            return format_string

        # deal with middle rounds
        prefix_string =  "Here are responses from other students:\n"
        
        for agent_name, responses in most_recent_responses.items():
            if agent_name == current_agent.agent_name:
                continue  
            if responses and 'parts' in responses[-1]:
                response_content = responses[-1]['parts'][0]
            else:
                response_content = responses[-1]['content']
            
            other_agent_response = f"One student's response: {response_content}\n"
            prefix_string += other_agent_response
        
        suffix_string = f"To remind you again of the story prompt: {question}\n"
        if is_last_round:
            suffix_string += " This is the final round of the discussion. Write a fully developed 300-word science fiction story with a clear beginning, middle, and end. Use engaging characters, vivid settings, and meaningful conflicts to craft a cohesive and imaginative narrative. Focus solely on storytelling, not analysis or brainstorming.\n"

        return prefix_string + suffix_string + format_string  
    
    def construct_response_teacher(self, question, most_recent_responses, teacher_response, current_agent, is_last_round=False, is_first_round=False, baseline=False, teacher=False):
        # generation format
        if teacher:
            format_string = """
            Please format your response as follows:

            1. Start with: "Here's my critique:"
            2. Provide constructive feedback focusing on creativity, plot coherence, and character depth.
            3. End with: "End of critique"

            Do not include any unrelated commentary or repeat the prompt. Your response must strictly adhere to this format.\n
            """
        else:
            format_string = """
            Please follow these formatting rules for your response:

            1. **Do not repeat the prompt**: Directly start generating the requested story without restating or summarizing the given prompt.
            2. **Start with the phrase**: "Here's my story:".
            3. **End with the phrase**: "End of Story".
            4. **Only include the story text between these phrases**: Avoid adding any commentary, explanation, or notes before or after the story.

            Do not ask questions or wait for guidance. Your response must strictly adhere to this format.\n
            """

        # deal with first round
        if is_first_round:
            if teacher:
                return f"{format_string}"
            else:
                prefix_string = "Here are the key insights provided by your teacher to guide you:\n"
                return f"{prefix_string} {teacher_response} {format_string}"

        # deal with middle rounds
        if teacher:
            prefix_string = "These are the responses to the prompt from all students from the previous round of open discussion:\n"
        else:
            prefix_string =  "As the student, here is the feedback from your teacher and responses from other students:\n"
            
        for agent_name, responses in most_recent_responses.items():
            if agent_name == current_agent.agent_name:
                continue
            if "Teacher" in agent_name:
                # if responses and 'parts' in responses[-1]:
                #     response_content = responses[-1]['parts'][0]
                # else:
                #     response_content = responses[-1]['content']
                response_content = teacher_response
                other_agent_response = f"Teacher's feedback: {response_content}\n"
            else:
                if responses and 'parts' in responses[-1]:
                    response_content = responses[-1]['parts'][0]
                else:
                    response_content = responses[-1]['content']
                other_agent_response = f"One student's response: {response_content}\n"
            prefix_string += other_agent_response

        suffix_string = f"To remind you again of the story prompt: {question}\n"
        if is_last_round:
            suffix_string += " This is the final round of the discussion. Write a fully developed 300-word science fiction story with a clear beginning, middle, and end. Use engaging characters, vivid settings, and meaningful conflicts to craft a cohesive and imaginative narrative. Focus solely on storytelling, not analysis or brainstorming.\n"

        return prefix_string + suffix_string + format_string
    
    def format_response(self, agent, response):
        return response
        # if agent.agent_type == "mistral":
        #     try:
        #         formatted_response = response.split("assistant")[-1]
        #     except:
        #         formatted_response = "The response has invalid format, ignore and proceed."
        #     # formatted_response = response
        # elif agent.agent_type == "llama":
        #     if "Teacher" in agent.agent_name:
        #         try:
        #             formatted_response = response.split("Your response must strictly adhere to this format.\n")[-1]
        #         except:
        #             try:
        #                 formatted_response = "Here's my critique:" + response.split("Here's my critique:")[-1]
        #             except:
        #                 formatted_response = "The response has invalid format, ignore and proceed."
        #     else:
        #         try:
        #             formatted_response = response.split("Your response must strictly adhere to this format.\n")[-1]
        #         except:
        #             try:
        #                 formatted_response = "Here's my story:" + response.split("Here's my story:")[-1]
        #             except:
        #                 formatted_response = "The response has invalid format, ignore and proceed."
        # return formatted_response

    def save_debate_conversations(self, agents, all_responses, init_results, final_results, amount_of_data, 
                                  final_logits=None, task_type="AUT", baseline=False):
        current_date, formatted_time = self.get_current_datetime()
        model_names_concatenated = self.concatenate_model_names(agents)
        role_names_concatenated = self.concatenate_role_names(agents)
        subtask = self.determine_subtask(agents, baseline)

        output_filename = self.generate_filename(task_type, subtask, "chat_log", model_names_concatenated, role_names_concatenated, current_date, formatted_time, amount_of_data, len(agents), self.rounds)
        final_ans_filename = self.generate_final_filename(task_type, subtask, "multi_agent", model_names_concatenated, role_names_concatenated, current_date, formatted_time, amount_of_data, len(agents), self.rounds)
        init_ans_filename = self.generate_filename(task_type, subtask, "init", model_names_concatenated, role_names_concatenated, current_date, formatted_time, amount_of_data, len(agents), self.rounds)

        # Save logits separately as .npy files to GCS
        if final_logits is not None:
            storage_client = self.get_storage_client()
            bucket = storage_client.bucket(self.bucket_name)
            for logit_entry in final_logits:
                agent_name = logit_entry["agent_name"]
                logits = logit_entry["logit"]

                # Generate a filename for each agent's logits
                logit_filename = self.generate_filename(
                    task_type, subtask, f"logits_{agent_name}", model_names_concatenated, role_names_concatenated,
                    current_date, formatted_time, amount_of_data, len(agents), self.rounds
                )
                logit_filename = logit_filename.replace(".json", ".npy")  # Save logits as a .npy file
                
                # Move logits to CPU and convert to NumPy before saving
                logits_np = np.array(logits.detach().cpu().numpy())
                
                with io.BytesIO() as byte_io:
                    np.save(byte_io, logits_np)  # Save the NumPy array into a byte stream
                    byte_io.seek(0)  # Rewind the byte stream to the beginning

                    # Upload the byte stream to GCS
                    blob = bucket.blob(logit_filename[6:])  # Remove the "../../" prefix from the filename
                    blob.upload_from_file(byte_io, content_type='application/octet-stream')

                print(f"Saved Logits to GCS bucket {self.bucket_name} with filename {logit_filename[6:]}")
                
                # Save to GCS
                # blob = bucket.blob(logit_filename[6:])  # Remove the "../../" prefix from the filename
                # blob.upload_from_string(logits_np.tobytes(), content_type='application/octet-stream')
                # print(f"Saved Logits to GCS bucket {self.bucket_name} with filename {logit_filename[6:]}")

                # Add the filename to the corresponding agent's entry in final_results
                for result in final_results:
                    if result["Agent"] == agent_name:
                        result["logits_file"] = logit_filename  # Add the logit filename to the final_results entry

        # Save the conversations, including the logit filenames, to GCS
        self.save_conversation(output_filename, all_responses)
        self.save_conversation(final_ans_filename, final_results)
        self.save_conversation(init_ans_filename, init_results)

        return final_ans_filename
    
    @staticmethod
    def get_current_datetime():
        current_time = datetime.datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        formatted_time = current_time.strftime("%H-%M-%S")
        return current_date, formatted_time
    
    @staticmethod
    def concatenate_model_names(agents):
        if all(agent.model_name == agents[0].model_name for agent in agents):
            return agents[0].model_name.replace(".", "-")
        return "-".join(agent.model_name.replace(".", "-") for agent in agents)

    @staticmethod
    def concatenate_role_names(agents):
        if all(agent.agent_role == "None" for agent in agents):
            return "None"
        return "-".join(agent.agent_role.replace(" ", "") for agent in agents)

    def determine_subtask(self, agents, baseline):
        if baseline:
            return "baseline"
        if all(agent.agent_role == "None" for agent in agents):
            return "FINAL"
        return "roleplay"
    
    @staticmethod
    def generate_filename(task_type, subtask, data_type, model_names_concatenated, role_names_concatenated, current_date, formatted_time, amount_of_data, num_agents, num_rounds):
        return f"../../Results/{task_type}/{data_type}/{task_type}_multi_teacher_{subtask}_{num_agents}_{num_rounds}_{model_names_concatenated}_{role_names_concatenated}_{data_type}_{current_date}-{formatted_time}_{amount_of_data}.json"

    @staticmethod
    def generate_final_filename(task_type, subtask, data_type, model_names_concatenated, role_names_concatenated, current_date, formatted_time, amount_of_data, num_agents, num_rounds):
        return f"../../Results/{task_type}/Output/{data_type}/{task_type}_multi_teacher_{subtask}_{num_agents}_{num_rounds}_{model_names_concatenated}_{role_names_concatenated}_{data_type}_{current_date}-{formatted_time}_{amount_of_data}.json"

class LLM_Discussion_SciFi(LLM_Debate):
    def run(self):
        with open(self.dataset_file, 'r') as f:
            dataset = json.load(f)
        all_responses = {}
        init_results = []
        final_results = []
        final_logits = []
        all_examples = [d for key, val in dataset.items() for d in val]
        amount_of_data = 0
        
        # normal runs
        for example in all_examples:
            amount_of_data += 1      
            self.process_example(example, amount_of_data)
        
        # decoding experiment section
        # temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
        # top_ps = [0.7, 0.8, 0.9, 1.0]
        # top_ks = [5,10,50,100]
        # rep_penalties = [1.0, 1.2, 1.5]
        
        # all_sampled_groups = []
        # for rep_pen in rep_penalties:
        #     sampled_examples = []
        #     for key, val in dataset.items():
        #         # Sample 2 examples from each 'val' with replacement
        #         sampled_examples.extend(random.sample(val, k=2))

        #     # Ensure we have 20 examples in total for each group
        #     assert len(sampled_examples) == 20
        #     all_sampled_groups.append((sampled_examples, rep_pen))
        
        # for examples, rep_pen in all_sampled_groups:
        #     for example in examples:
        #         amount_of_data += 1      
        #         self.process_example(example, amount_of_data, rep_pen=rep_pen)
    
    def process_example(self, example, amount_of_data, temp=0.6, top_p=0.9, top_k=50, rep_pen=1.0):
        all_responses = {}
        init_results = []
        final_results = []
        final_logits = []
        
        print('processing prompt:', amount_of_data)
        chat_history = {agent.agent_name: [] for agent in self.agents}
        question = example
        initial_prompt = f"Welcome to today's creative writing class! You'll work together with your classmates to craft an engaging science fiction story with the following prompt: {question} Ensure the story is logically consistent, creative, and written as a full narrative with a beginning, middle, and end. Do not add notes or placeholders—only produce the final story.\n"
        
        # # # mistral prompt
        # mistral_prefix = "<s>[INST]"
        # mistral_suffix = "[/INST]"
        
        most_recent_responses = {}
        for round in range(self.rounds):
            is_last_round = (round == self.rounds - 1)
            is_first_round = (round == 0)
            round_responses = {agent.agent_name: [] for agent in self.agents}
            print(f"Round {round + 1}: Discussion on {question}")
            
            for agent in self.agents:

                if agent.agent_role != "None":
                    agent_role_prompt = f"You are a {agent.agent_role} whose specialty is {agent.agent_speciality}. Here's a breakdown of your role: {agent.agent_role_prompt} Besides that specialty, you should also follow some general instructions: {agent.general_instruction} Remember to claim your role in the beginning of each conversation.\n"
                    # print(f"agent_role = {agent.agent_role}")
                else:
                    agent_role_prompt = ""

                if is_first_round:
                    combined_prompt = self.construct_response(question, most_recent_responses, agent, is_last_round=is_last_round, is_first_round=is_first_round)
                    formatted_initial_prompt = agent.construct_user_message(initial_prompt + agent_role_prompt + combined_prompt)
                    # formatted_initial_prompt = agent.construct_user_message(mistral_prefix + initial_prompt + agent_role_prompt + combined_prompt + mistral_suffix) # prompt for mistral
                    # print("-----------------------\n\n")
                    # print(f"1st round formatted_initial_prompt = {formatted_initial_prompt}")
                    # print("-----------------------")
                    chat_history[agent.agent_name].append(formatted_initial_prompt)
                    raw_response = agent.generate_answer([formatted_initial_prompt])
                    response = self.format_response(agent, raw_response)
                    # print("-----------------------\n\n")
                    # print(f"1st round student response = {response}")
                    # print("-----------------------")
                    init_result = {"question": question, "answer": [response], "Agent": agent.agent_name}
                    init_results.append(init_result)
                else:
                    combined_prompt = self.construct_response(question, most_recent_responses, agent, is_last_round=is_last_round)
                    formatted_combined_prompt = agent.construct_user_message(agent_role_prompt + combined_prompt)
                    # formatted_combined_prompt = agent.construct_user_message(mistral_prefix + agent_role_prompt + combined_prompt + mistral_suffix)
                    # print("-----------------------\n\n")
                    # print(f"formatted_combined_prompt = {formatted_combined_prompt}")
                    # print("-----------------------")
                    chat_history[agent.agent_name].append(formatted_combined_prompt)

                    # Save Final Result
                    if is_last_round:
                        raw_response, logits, entropy_max, entropy_mean = agent.generate_answer([formatted_combined_prompt], final_round=is_last_round)
                        response = self.format_response(agent, raw_response)
                        # print("-----------------------\n\n")
                        # print(f"Last round student response = {response}")
                        # print("-----------------------")
                        final_result = {"question": question, "answer": [response], "Agent": agent.agent_name, 
                                        "entropy_max": entropy_max.item() if isinstance(entropy_max, torch.Tensor) else entropy_max,
                                        "entropy_mean": entropy_mean.item() if isinstance(entropy_mean, torch.Tensor) else entropy_mean}
                        final_results.append(final_result)
                        final_logit = {"agent_name": agent.agent_name, "logit": logits}
                        final_logits.append(final_logit)
                    else:
                        raw_response = agent.generate_answer([formatted_combined_prompt], final_round=is_last_round)
                        response = self.format_response(agent, raw_response)
                        # print("-----------------------\n\n")
                        # print(f"Middle round student response = {response}")
                        # print("-----------------------")

                formatted_response = agent.construct_assistant_message(response)
                chat_history[agent.agent_name].append(formatted_response)  # Update the agent's chat history
                round_responses[agent.agent_name].append(formatted_response)
                
            most_recent_responses = round_responses
        all_responses[question] = chat_history
        self.save_debate_conversations(self.agents, all_responses, init_results, final_results,  amount_of_data, final_logits, task_type=self.task_type)


class LLM_Teacher_SciFi(LLM_Debate):
    def run(self):
        with open(self.dataset_file, 'r') as f:
            dataset = json.load(f)
        all_responses = {}
        init_results = []
        final_results = []
        final_logits = []
        all_examples = [d for key, val in dataset.items() for d in val]
        amount_of_data = 0
        
        # normal runs
        for example in all_examples:
            amount_of_data += 1      
            self.process_example(example, amount_of_data)
        
        # decoding experiment section
        # temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
        # top_ps = [0.7, 0.8, 0.9, 1.0]
        # top_ks = [5,10,50,100]
        # rep_penalties = [1.0, 1.2, 1.5]
        
        # all_sampled_groups = []
        # for rep_pen in rep_penalties:
        #     sampled_examples = []
        #     for key, val in dataset.items():
        #         # Sample 2 examples from each 'val' with replacement
        #         sampled_examples.extend(random.sample(val, k=2))

        #     # Ensure we have 20 examples in total for each group
        #     assert len(sampled_examples) == 20
        #     all_sampled_groups.append((sampled_examples, rep_pen))
        
        # for examples, rep_pen in all_sampled_groups:
        #     for example in examples:
        #         amount_of_data += 1      
        #         self.process_example(example, amount_of_data, rep_pen=rep_pen)

    def process_example(self, example, amount_of_data, temp=0.6, top_p=0.9, top_k=50, rep_pen=1.0):
            
        print('processing prompt:', amount_of_data)
                    
        all_responses = {}
        init_results = []
        final_results = []
        final_logits = []

        chat_history = {agent.agent_name: [] for agent in self.agents}
        question = example
        initial_prompt_teacher = f"Welcome to today's creative writing class! You'll work together with your students to craft an engaging science fiction story with the following prompt: {question}\n"
        initial_prompt_student = f"{question} Ensure the story is logically consistent, creative, and written as a full narrative with a beginning, middle, and end. Do not add notes or placeholders—only produce the final story.\n"
        most_recent_responses = {}
        teacher_response = ""
        
        # mistral prompt
        # mistral_prefix = "<s>[INST]"
        # mistral_suffix = "[/INST]"

        for round in range(self.rounds):
            is_last_round = (round == self.rounds - 1)
            is_first_round = (round == 0)
            round_responses = {agent.agent_name: [] for agent in self.agents}
            print(f"Round {round + 1}: Discussion on {question}")

            for idx, agent in enumerate(self.agents):
                # check for valid definition of agent role
                if agent.agent_role != "None":
                    
                    # teacher agent
                    if idx == 0:
                        # first round - teacher prompt (analyze story prompt)
                        if is_first_round:
                            agent_role_prompt = f"You are a {agent.agent_role} whose specialty is {agent.agent_speciality}. {agent.agent_role_prompt_initial}\n"
                        else: # followup rounds - teacher prompt (instruct student writers)
                            agent_role_prompt = f"You are a {agent.agent_role} whose specialty is {agent.agent_speciality}. {agent.agent_role_prompt_followup}\n"
                    else: # student agent
                        agent_role_prompt = f"You are a {agent.agent_role} whose specialty is {agent.agent_speciality}. Here's a breakdown of your role:{agent.agent_role_prompt} Besides that specialty, you should also follow some general instructions: {agent.general_instruction} Remember to claim your role in the beginning of each conversation.\n"
                else:
                    agent_role_prompt = ""

                if is_first_round and idx == 0:  # Teacher's initial lecture
                    combined_prompt = self.construct_response_teacher(question, most_recent_responses, teacher_response, agent, is_last_round, is_first_round=True, teacher=True)
                    formatted_initial_prompt = agent.construct_user_message(initial_prompt_teacher + agent_role_prompt + combined_prompt)
                    # formatted_initial_prompt = agent.construct_user_message(mistral_prefix + initial_prompt_teacher + agent_role_prompt + combined_prompt + mistral_suffix) # prompt for mistral
                    # print("-----------------------\n\n")
                    # print(f"1st round formatted_initial_prompt = {formatted_initial_prompt}")
                    # print("-----------------------")
                    chat_history[agent.agent_name].append(formatted_initial_prompt)
                    raw_response = agent.generate_answer([formatted_initial_prompt], token_limit=300)
                    teacher_response = self.format_response(agent, raw_response)
                    # print("-----------------------\n\n")
                    # print(f"1st round teacher response = {teacher_response}")
                    # print("-----------------------")
                    init_result = {"question": question, "answer": [teacher_response], "Agent": agent.agent_name}
                    init_results.append(init_result)
                elif is_first_round and idx != 0:  # Students' initial responses
                    combined_prompt = self.construct_response_teacher(question, most_recent_responses, teacher_response, agent, is_last_round, is_first_round=True, teacher=False)
                    formatted_combined_prompt = agent.construct_user_message(initial_prompt_student + agent_role_prompt + combined_prompt)
                    # formatted_combined_prompt = agent.construct_user_message(mistral_prefix + initial_prompt_student + agent_role_prompt + combined_prompt + mistral_suffix) # mistral prompt
                    # print("-----------------------\n\n")
                    # print(f"1st round formatted_combined_prompt = {formatted_combined_prompt}")
                    # print("-----------------------")
                    chat_history[agent.agent_name].append(formatted_combined_prompt)
                    raw_response = agent.generate_answer([formatted_combined_prompt])
                    response = self.format_response(agent, raw_response)
                    # print("-----------------------\n\n")
                    # print(f"1st round student response = {response}")
                    # print("-----------------------")
                elif not is_first_round and idx == 0: # Teacher's middle response
                    combined_prompt = self.construct_response_teacher(question, most_recent_responses, teacher_response, agent, is_last_round, teacher=True)
                    formatted_combined_prompt = agent.construct_user_message(agent_role_prompt + combined_prompt)
                    # formatted_combined_prompt = agent.construct_user_message(mistral_prefix + agent_role_prompt + combined_prompt + mistral_suffix)
                    # print("-----------------------\n\n")
                    # print(f"formatted_combined_prompt  = {combined_prompt}")
                    # print("-----------------------")
                    chat_history[agent.agent_name].append(formatted_combined_prompt)
                    raw_response = agent.generate_answer([formatted_combined_prompt], token_limit=300)
                    response = self.format_response(agent, raw_response)
                    teacher_response = response
                    # print("-----------------------\n\n")
                    # print(f"Middle round teacher response = {response}")
                    # print("-----------------------")
                else: # Students' middle responses + all agents final round
                    combined_prompt = self.construct_response_teacher(question, most_recent_responses, teacher_response, agent, is_last_round, teacher=(idx == 0))
                    formatted_combined_prompt = agent.construct_user_message(agent_role_prompt + combined_prompt)
                    # formatted_combined_prompt = agent.construct_user_message(mistral_prefix + agent_role_prompt + combined_prompt + mistral_suffix)
                    # print("-----------------------\n\n")
                    # print(f"formatted_combined_prompt = {formatted_combined_prompt}")
                    # print("-----------------------")
                    chat_history[agent.agent_name].append(formatted_combined_prompt)
                    if is_last_round and idx != 0:  # Save only student final responses
                        # raw_response, logits = agent.generate_answer([formatted_combined_prompt], final_round=is_last_round)
                        raw_response, logits, entropy_max, entropy_mean = agent.generate_answer([formatted_combined_prompt], final_round=is_last_round)
                        # raw_response, entropy = agent.generate_answer([formatted_combined_prompt], final_round=is_last_round)
                        response = self.format_response(agent, raw_response)
                        # print("-----------------------\n\n")
                        # print(f"Middle round student response = {response}")
                        # print("-----------------------")
                        final_result = {"question": question, "answer": [response], "Agent": agent.agent_name, 
                                        "entropy_max": entropy_max.item() if isinstance(entropy_max, torch.Tensor) else entropy_max,
                                        "entropy_mean": entropy_mean.item() if isinstance(entropy_mean, torch.Tensor) else entropy_mean}
                        final_results.append(final_result)
                        final_logit = {"agent_name": agent.agent_name, "logit": logits}
                        final_logits.append(final_logit)
                    else:
                        raw_response = agent.generate_answer([formatted_combined_prompt], final_round=is_last_round)
                        response = self.format_response(agent, raw_response)
                        # print("-----------------------\n\n")
                        # print(f"Middle round student response = {response}")
                        # print("-----------------------")
                try:
                    formatted_response = agent.construct_assistant_message(response)
                except:
                    formatted_response = agent.construct_assistant_message(teacher_response)
                chat_history[agent.agent_name].append(formatted_response)
                round_responses[agent.agent_name].append(formatted_response)

            most_recent_responses = round_responses
        all_responses[question] = chat_history

        self.save_debate_conversations(self.agents, all_responses, init_results, final_results, amount_of_data, final_logits, task_type=self.task_type)


class LLM_Review_SciFi(LLM_Debate):
    def run(self):
        with open(self.dataset_file, 'r') as f:
            dataset = json.load(f)
        all_responses = {}
        init_results = []
        final_results = []
        final_logits = []
        all_examples = [d for key, val in dataset.items() for d in val]
        amount_of_data = 0
        
        # normal runs
        for example in all_examples:
            amount_of_data += 1      
            self.process_example(example, amount_of_data)
        
        # decoding experiment section
        # temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
        # top_ps = [0.7, 0.8, 0.9, 1.0]
        # top_ks = [5,10,50,100]
        # rep_penalties = [1.0, 1.2, 1.5]
        
        # all_sampled_groups = []
        # for rep_pen in rep_penalties:
        #     sampled_examples = []
        #     for key, val in dataset.items():
        #         # Sample 2 examples from each 'val' with replacement
        #         sampled_examples.extend(random.sample(val, k=2))

        #     # Ensure we have 20 examples in total for each group
        #     assert len(sampled_examples) == 20
        #     all_sampled_groups.append((sampled_examples, rep_pen))
        
        # for examples, rep_pen in all_sampled_groups:
        #     for example in examples:
        #         amount_of_data += 1      
        #         self.process_example(example, amount_of_data, rep_pen=rep_pen)
        
    def process_example(self, example, amount_of_data, temp=0.6, top_p=0.9, top_k=50, rep_pen=1.0):
        
        print('Processing prompt:', amount_of_data)
                
        all_responses = {}
        init_results = []
        final_results = []
        final_logits = []
        
        chat_history = {agent.agent_role: [] for agent in self.agents}
        question = example

        agent_stories = {agent.agent_role: [] for agent in self.agents}
        agent_reviews = {
            agent.agent_role: {
                other_agent.agent_role: []
                for other_agent in self.agents
                if other_agent.agent_role != agent.agent_role
            }
            for agent in self.agents
        }

        for round in range(self.rounds):
            is_last_round = (round == self.rounds - 1)
            is_first_round = (round == 0)
            print(f"Round {round + 1}: Discussion on '{question}'")

            # each agent generates a story first
            for agent in self.agents:
                if agent.agent_role != "None":
                    agent_role_prompt = (
                        f"You are a {agent.agent_role} whose specialty is "
                        f"{agent.agent_speciality}. Here's a breakdown of your role: "
                        f"{agent.agent_role_prompt} Besides that specialty, you should "
                        f"also follow some general instructions: {agent.general_instruction} "
                        "Remember to claim your role at the beginning of each conversation.\n"
                    )
                else:
                    agent_role_prompt = ""

                if is_first_round:
                    combined_prompt = self.construct_story_prompt_first_round(question, agent)
                else:
                    combined_prompt = self.construct_story_prompt(
                        question, agent, agent_stories, agent_reviews
                    )

                formatted_prompt = agent.construct_user_message(agent_role_prompt + combined_prompt)
                chat_history[agent.agent_role].append(formatted_prompt)

                # print(f"\n----- Agent {agent.agent_role} Prompt -----\n")
                # print(formatted_prompt)
                # print("\n------------------------------------------\n")

                if is_last_round:
                    raw_response, logits, entropy_max, entropy_mean = agent.generate_answer([formatted_prompt], final_round=True)
                    response = self.format_response(agent, raw_response)
                    final_result = {
                        "question": question,
                        "answer": [response],
                        "Agent": agent.agent_name,
                        "entropy_max": entropy_max.item() if isinstance(entropy_max, torch.Tensor) else entropy_max,
                        "entropy_mean": entropy_mean.item() if isinstance(entropy_mean, torch.Tensor) else entropy_mean
                    }
                    final_results.append(final_result)
                    final_logit = {"agent_name": agent.agent_name, "logit": logits}
                    final_logits.append(final_logit)
                else:
                    raw_response = agent.generate_answer([formatted_prompt])
                    response = self.format_response(agent, raw_response)
                    if is_first_round:
                        # save initial result
                        init_result = {
                            "question": question,
                            "answer": [response],
                            "Agent": agent.agent_name,
                        }
                        init_results.append(init_result)

                formatted_response = agent.construct_assistant_message(response)
                chat_history[agent.agent_role].append(formatted_response)
                agent_stories[agent.agent_role].append(response)

                # print(f"\n----- Agent {agent.agent_role} Response -----\n")
                # print(response)
                # print("\n---------------------------------------------\n")

            # Then, each agent reviews other agents' stories
            for agent in self.agents:
                if agent.agent_role != "None":
                    agent_role_prompt = (
                        f"You are a {agent.agent_role} whose specialty is "
                        f"{agent.agent_speciality}. Here's a breakdown of your role: "
                        f"{agent.agent_role_prompt} Besides that specialty, you should "
                        f"also follow some general instructions: {agent.general_instruction} "
                        "Remember to claim your role at the beginning of each conversation.\n"
                    )
                else:
                    agent_role_prompt = ""

                combined_prompt = self.construct_review_prompt(question, agent, agent_stories)
                formatted_prompt = agent.construct_user_message(agent_role_prompt + combined_prompt)
                chat_history[agent.agent_role].append(formatted_prompt)

                # print(f"\n----- Agent {agent.agent_role} Review Prompt -----\n")
                # print(formatted_prompt)
                # print("\n------------------------------------------------\n")

                raw_response = agent.generate_answer([formatted_prompt])
                response = self.format_response(agent, raw_response)

                formatted_response = agent.construct_assistant_message(response)
                chat_history[agent.agent_role].append(formatted_response)

                # print(f"\n----- Agent {agent.agent_role} Review Response -----\n")
                # print(response)
                # print("\n----------------------------------------------------\n")

                # parse and store the reviews
                reviews = self.parse_reviews(agent, response)
                # print(f"\n----- Extract Agent {agent.agent_role} Reviews -----\n")
                # print(reviews)
                # print("\n----------------------------------------------------\n")
                # print('----------agent role------------')
                # print(agent.agent_role)
                # print(reviews.keys())
                # print(agent_reviews)
                for other_agent_name, review in reviews.items():
                    if other_agent_name in agent_reviews:
                        if agent.agent_role in agent_reviews[other_agent_name]:
                            agent_reviews[other_agent_name][agent.agent_role].append(review)

        all_responses[question] = chat_history

        # save the conversations and results
        self.save_debate_conversations(
            self.agents,
            all_responses,
            init_results,
            final_results,
            amount_of_data,
            final_logits,
            task_type=self.task_type,
            )

    def construct_story_prompt_first_round(self, question, agent):
        format_string = f"""
        Please write a science fiction story based on the following prompt:
        {question}

        Please follow these formatting rules for your response:

        1. **Do not repeat the prompt**: Directly start generating the requested story without restating or summarizing the given prompt.
        2. **Start with the phrase**: "Here's my story:".
        3. **End with the phrase**: "End of Story".
        4. **Only include the story text between these phrases**: Avoid adding any commentary, explanation, or notes before or after the story.

        Do not ask questions or wait for guidance. Your response must strictly adhere to this format.\n
        """
        return format_string

    def construct_story_prompt(self, question, agent, agent_stories, agent_reviews):
        previous_story = agent_stories[agent.agent_role][-1]  # Get the agent's previous story
        reviews = [
            f"Feedback from {reviewer}:\n{agent_reviews[agent.agent_role][reviewer][-1]}"
            for reviewer in agent_reviews[agent.agent_role]
            if agent_reviews[agent.agent_role][reviewer]
        ]

        review_text = "\n".join(reviews) if reviews else "No feedback received yet."

        format_string = f"""
        Based on your previous story and the feedback from your peers, please improve your science fiction story.

        Your previous story:
        {previous_story}

        Feedback from your peers:
        {review_text}

        Please write a new version of your story, addressing the feedback. Follow the same formatting rules:

        1. **Do not repeat the prompt**: Directly start generating the requested story without restating or summarizing the given prompt.
        2. **Start with the phrase**: "Here's my story:".
        3. **End with the phrase**: "End of Story".
        4. **Only include the story text between these phrases**: Avoid adding any commentary, explanation, or notes before or after the story.

        Do not ask questions or wait for guidance. Your response must strictly adhere to this format.\n
        """
        return format_string

    def construct_review_prompt(self, question, agent, agent_stories):
        other_agents = [other_agent for other_agent in self.agents if other_agent.agent_role != agent.agent_name]
        other_stories = [
            (other_agent.agent_role, agent_stories[other_agent.agent_role][-1]) for other_agent in other_agents
        ]

        stories_text = "\n".join([f"{name}'s story:\n{story}" for name, story in other_stories])

        format_string = f"""
        Please review the following stories from your peers and provide constructive feedback for each one.

        {stories_text}

        Please format your feedback as follows:

        For each story, start with "Feedback for [Agent Name]:" on a new line, then provide your feedback.

        Focus your feedback on emphasizing creativity by encouraging the use of rare vocabulary, vivid imagery, and surprising plot elements. Highlight opportunities where the story could benefit from more imaginative settings, unique word choices, and unexpected twists. Ensure feedback also covers plot coherence, character development, and areas for improvement.

        Do not include any unrelated commentary or repeat the prompt. Your response must strictly adhere to this format.\n
        """
        return format_string

    def parse_reviews(self, agent, response):
        # Updated pattern to capture "Feedback for LLM Agent - xxx Writer's story"
        reviews = {}
        pattern = r"Feedback for ([\w\s\-]+):\s*(.+?)(?=(?:\n\s*)+Feedback for [\w\s\-]+:|$)"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for agent_role, feedback in matches:
            reviews[agent_role.strip()] = feedback.strip()
        
        return reviews
