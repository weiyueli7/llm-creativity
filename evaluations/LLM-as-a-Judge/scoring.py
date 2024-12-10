import os
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

bucket_name = "nlp_result"
def download_json_files(bucket_name, prefix, local_dir):

    from google.cloud import storage
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../secrets.json"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    else:
        os.system(f"rm -rf {local_dir}/*")
    

    print(f"Downloading from {bucket_name}/{prefix} to {local_dir}")

    file_paths = []
    for blob in blobs:
        if blob.name.endswith('.json'):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            file_paths.append(local_path)
    return file_paths


prefix = "Eval_Results_v2/Single_Agent/Output"
local_dir = "eval_results_single"
single_paths = download_json_files(bucket_name, prefix, local_dir)


prefix = "Eval_Results_v2/SciFi-Dis/Output/multi_agent"
local_dir = "eval_results_discuss"
discuss_paths = download_json_files(bucket_name, prefix, local_dir)


prefix = "Eval_Results_v2/SciFi-Teacher/Output/multi_agent"
local_dir = "eval_results_teacher"
teacher_paths = download_json_files(bucket_name, prefix, local_dir)


prefix = "Eval_Results_v2/SciFi-Review/Output/multi_agent"
local_dir = "eval_results_review"
review_paths = download_json_files(bucket_name, prefix, local_dir)

def calculate_avg(paths):
    agent1 = []
    agent2 = []
    agent3 = []
    for p in paths:
        data = json.load(open(p))
        agent1.append(data[0]['scores'])
        try:
            agent2.append(data[1]['scores'])
            agent3.append(data[2]['scores'])
        except:
            pass
    agent1 = np.array([list(a.values()) for a in agent1])
    agent1 = pd.DataFrame(agent1).fillna(0).values
    try:
        agent2 = np.array([list(a.values()) for a in agent2])
        agent3 = np.array([list(a.values()) for a in agent3])
        
        agent2 = pd.DataFrame(agent2).fillna(0).values
        agent3 = pd.DataFrame(agent3).fillna(0).values
    except:
        pass
    try:
        return np.array([np.mean(agent1, axis=0), np.mean(agent2, axis=0), np.mean(agent3, axis=0)]).mean(axis=0), np.array([np.std(agent1, axis=0), np.std(agent2, axis=0), np.std(agent3, axis=0)]).mean(axis=0)
    except:
        return np.mean(agent1, axis=0), np.std(agent1, axis=0)

single_mean, single_std = calculate_avg(single_paths)
print("Single Agent:")
print(f"Mean: {single_mean}\nStd: {single_std}")

discuss_mean, discuss_std = calculate_avg(discuss_paths)
print("LLM-Discussion:")
print(f"Mean: {discuss_mean}\nStd: {discuss_std}")

teacher_mean, teacher_std = calculate_avg(teacher_paths)
print("LLM-Teacher:")
print(f"Mean: {teacher_mean}\nStd: {teacher_std}")

review_mean, review_std = calculate_avg(review_paths)
print("LLM-Review:")
print(f"Mean: {review_mean}\nStd: {review_std}")
