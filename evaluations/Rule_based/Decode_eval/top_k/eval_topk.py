import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import requests
import random
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, norm
#from google.colab import drive
import json

def extract_generated_story(text):
    split_text = text.split("Generated Story:", 1)
    if len(split_text) > 1:
        return split_text[1].strip()
    else:
        return

def load_data_single(base_path):
    all_data = {}

    for sub_folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, sub_folder)
        if os.path.isdir(folder_path):
            story_data = []

            txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
            npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

            for i in range(1, 11):
                story_id = f"story_{i}"
                story_entry = {"id": story_id, "text": None, "entropy": None}

                txt_file = f"{story_id}.txt"
                txt_path = os.path.join(folder_path, txt_file)
                if txt_file in txt_files:
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            full_text = f.read()
                        story_entry["text"] = extract_generated_story(full_text)
                    except UnicodeDecodeError:
                        pass
                else:
                    pass

                npy_file = f"story_{story_id}_entropy.npy"
                npy_path = os.path.join(folder_path, npy_file)
                if npy_file in npy_files:
                    try:
                        entropy = np.load(npy_path).item()
                        story_entry["entropy"] = entropy
                    except Exception as e:
                        pass
                else:
                    pass

                story_data.append(story_entry)

            all_data[sub_folder] = story_data

    return all_data

def load_data_multi(folder_path):

    all_data = {}

    for file_idx, file_name in enumerate(sorted(os.listdir(folder_path)), start=1):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".json"):
            category_name = f"category_{file_idx}"
            all_data[category_name] = []

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for agent_idx, item in enumerate(data, start=1):
                        if isinstance(item, dict):
                            answers = item.get("answer")
                            entropy = item.get("entropy")

                            if answers is not None and entropy is not None:
                                if isinstance(answers, str):
                                    answers_cleaned = answers.replace("\n", " ")
                                    story_id = f"story_{file_idx}.{agent_idx}"
                                    all_data[category_name].append(
                                        {"id": story_id, "text": answers_cleaned, "entropy": entropy}
                                    )

                                elif isinstance(answers, list):
                                    for sub_idx, ans in enumerate(answers, start=1):
                                        if isinstance(ans, str):
                                            ans_cleaned = ans.replace("\n", " ")
                                            story_id = f"story_{file_idx}.{agent_idx}.{sub_idx}"  # 简洁的 ID 格式
                                            all_data[category_name].append(
                                                {"id": story_id, "text": ans_cleaned, "entropy": entropy}
                                            )
                                        else:
                                            pass
                else:
                    pass

            except json.JSONDecodeError:
                pass
            except Exception as e:
                pass

    return all_data

import os
import json

def load_data_multi_decode(folder_path):
    all_data = {}

    for file_idx, file_name in enumerate(sorted(os.listdir(folder_path)), start=1):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".json"):
            category_name = f"category_{file_idx}"
            all_data[category_name] = []

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for agent_idx, item in enumerate(data, start=1):
                        if isinstance(item, dict):
                            answers = item.get("answer")
                            #entropy_max = item.get("entropy_Max")
                            entropy_mean = item.get("entropy_mean")
                            Score = item.get("top_k")
                            cosine_similarity=item.get('cosine_similarity')
                            kl_divergence=item.get('kl_divergence')
                            if answers is not None and isinstance(answers, list):
                                for sub_idx, ans in enumerate(answers, start=1):
                                    if isinstance(ans, str):
                                        ans_cleaned = ans.replace("\n", " ").strip()
                                        story_id = f"story_{file_idx}.{agent_idx}.{sub_idx}"
                                        all_data[category_name].append(
                                            {
                                                "id": story_id,
                                                "text": ans_cleaned,
                                                "top_k": Score,
                                                "entropy": entropy_mean,
                                                "cosine_similarity":cosine_similarity,
                                                "kl_divergence":kl_divergence
                                            }
                                        )
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file_name}")
            except Exception as e:
                print(f"Unexpected error in file {file_name}: {e}")

    return all_data



def extract_entropy(data):

    results = []

    for category, stories in data.items():
        for story in stories:
            text = story.get("text")
            entropy = story.get("entropy")
            results.append({"text": text, "entropy": entropy})
    return results

def calculate_cosine_similarity(generated_sentence, reference_embedding):
   
    generated_embedding = model.encode(generated_sentence)

    cosine_similarities = util.cos_sim(generated_embedding, reference_embedding)

    max_cos_similarity = cosine_similarities.max().item()

    return max_cos_similarity

def calculate_similarity_for_all_stories(data, reference_embedding):

    for category, stories in data.items():
        for story in stories:
            story_text = story.get("text")
            if story_text:
                try:
                    cos_sim = calculate_cosine_similarity(story_text, reference_embedding)
                    story["cosine_similarity"] = 1-cos_sim
                except Exception as e:
                    story["cosine_similarity"] = None
            else:
                story["cosine_similarity"] = None
    return data

def calculate_word_frequencies(text):

    words = text.split()
    word_counts = Counter(words)
    total_count = sum(word_counts.values())
    word_probabilities = {word: count / total_count for word, count in word_counts.items()}
    return word_probabilities

def calculate_kl_divergence(text1, q_dist):
    p_dist = calculate_word_frequencies(text1)

    all_words = set(p_dist.keys()).union(set(q_dist.keys()))
    p_probs = np.array([p_dist.get(word, 0) for word in all_words])
    q_probs = np.array([q_dist.get(word, 0) for word in all_words])

    mask = p_probs > 0
    p_probs = p_probs[mask]
    q_probs = q_probs[mask]

    kl_divergence = np.sum(p_probs * np.log(p_probs / (q_probs + 1e-12)))
    return kl_divergence

def calculate_kl_divergence_for_all_stories(data, q_dist):

    for category, stories in data.items():
        for story in stories:
            story_text = story.get("text")
            if story_text:
                try:
                    kl_divergence = calculate_kl_divergence(story_text, q_dist)
                    story["kl_divergence"] = kl_divergence
                except Exception as e:
                    story["kl_divergence"] = None
            else:
                story["kl_divergence"] = None
    return data

def filter_stories(data):

    filtered_data = {}

    for category, stories in data.items():
        filtered_stories = []
        for story in stories:
            #entropy = story.get("entropy")
            cosine_similarity = story.get("cosine_similarity")
            kl_divergence = story.get("kl_divergence")
            #if entropy is not None and cosine_similarity is not None and kl_divergence is not None and entropy >= 10:
            if cosine_similarity is not None and kl_divergence is not None:
                filtered_stories.append(story)
            else:
                print(f"Filted ID: {story['id']}")

        if filtered_stories:
            filtered_data[category] = filtered_stories

    return filtered_data

def merge_data(named_dicts):
    merged_data = {}

    for dict_name, data in named_dicts:
        for category, stories in data.items():
            if category not in merged_data:
                merged_data[category] = []

            for story in stories:
                story["id"] = f"{dict_name}_{story['id']}"
                merged_data[category].append(story)

    return merged_data


def standardize_values(data):
    entropy_values = [story["entropy"] for stories in data.values() for story in stories if story["entropy"] is not None]
    #entropymean_values = [story["entropy_mean"] for stories in data.values() for story in stories if story["entropy_mean"] is not None]
    cosine_values = [story["cosine_similarity"] for stories in data.values() for story in stories if story["cosine_similarity"] is not None]
    kl_values = [story["kl_divergence"] for stories in data.values() for story in stories if story["kl_divergence"] is not None]
    entropy_min, entropy_max = min(entropy_values), max(entropy_values)
    #entropymean_min, entropymean_max = min(entropymean_values), max(entropymean_values)
    cosine_min, cosine_max = min(cosine_values), max(cosine_values)
    kl_min, kl_max = min(kl_values), max(kl_values)

    for category, stories in data.items():
        for story in stories:
            if story["entropy"] is not None:
                if entropy_max > entropy_min:
                    normalized_entropy = (story["entropy"] - entropy_min) / (entropy_max - entropy_min)
                else:
                    normalized_entropy = 0.5
                story["rescaled_entropy"] = normalized_entropy * 100
            else:
                story["rescaled_entropy"] = None
            #if story["entropy_mean"] is not None:
            #    if entropymean_max > entropymean_min:
            #        normalized_entropy = (story["entropy_mean"] - entropymean_min) / (entropymean_max - entropymean_min)
            #    else:
            #        normalized_entropy = 0.5
            #    story["rescaled_entropy_mean"] = normalized_entropy * 100
            #else:
            #    story["rescaled_entropy_mean"] = None
            if story["cosine_similarity"] is not None:
                if cosine_max > cosine_min:
                    normalized_cosine = (story["cosine_similarity"] - cosine_min) / (cosine_max - cosine_min)
                else:
                    normalized_cosine = 0.5
                story["rescaled_cosine_similarity"] = normalized_cosine * 100
            else:
                story["rescaled_cosine_similarity"] = None

            if story["kl_divergence"] is not None:
                if kl_max > kl_min:
                    normalized_kl = (story["kl_divergence"] - kl_min) / (kl_max - kl_min)
                else:
                    normalized_kl = 0.5
                story["rescaled_kl_divergence"] = normalized_kl * 100
            else:
                story["rescaled_kl_divergence"] = None

    return data

def calculate_averages(data):
    source_sums = {}
    source_counts = {}
    for category, stories in data.items():
        for story in stories:
            source = story["id"].split("_")[0]
            if source not in source_sums:
                source_sums[source] = {"entropy": 0, "cosine_similarity": 0, "kl_divergence": 0,"rescaled_entropy": 0, "rescaled_cosine_similarity": 0, "rescaled_kl_divergence": 0}
                source_counts[source] = 0
            for key in ["entropy_max","entropy_mean", "cosine_similarity", "kl_divergence","rescaled_entropy", "rescaled_cosine_similarity", "rescaled_kl_divergence"]:
                if story.get(key) is not None:
                    source_sums[source][key] += story[key]
            source_counts[source] += 1
    averages = {}
    for source, sums in source_sums.items():
        averages[source] = {
            key: sums[key] / source_counts[source] for key in sums.keys()
        }

    return averages

if __name__ == "__main__":
    model = SentenceTransformer('all-mpnet-base-v2')
    base_path_temp = "./Results/Decoding/Top_k"
    #base_path_multi_decode='D:\LLM-Discussion\Results\\temp'
    data_multi = load_data_multi_decode(base_path_temp)
    #data_multi_discussion = load_data_multi(base_path_multi_discussion)
    reference_embedding = np.load("./Evaluation/Rule_based/all_embeddings.npy")
    cosine_similarity_dict_multi = calculate_similarity_for_all_stories(data_multi, reference_embedding)
    with open("./Evaluation/Rule_based/q_dist.json", "r") as json_file:
        q_dist = json.load(json_file)
    kl_divergence_dict_multi = calculate_kl_divergence_for_all_stories(data_multi, q_dist)
    filtered_result_multi = filter_stories(kl_divergence_dict_multi)
    merged_result = merge_data([("multi", filtered_result_multi)])
    standard_result = standardize_values(merged_result)
    file_name = "./Evaluation/Rule_based/Decode_eval/Statistics_top_k.json"
    # average_statistics=calculate_averages(standard_result)
    #print(kl_divergence_dict_multi)
    with open(file_name, "w", encoding="utf-8") as json_file:
         json.dump(standard_result, json_file, indent=4, ensure_ascii=False)  
 