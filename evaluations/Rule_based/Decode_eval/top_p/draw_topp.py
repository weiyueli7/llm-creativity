import json
import matplotlib.pyplot as plt
import numpy as np

# Attempting to read the file with a different encoding (ISO-8859-1) as a fallback for potential encoding issues
file_path = "./Evaluation/Rule_based/Decode_eval/Statistics_top_p.json"

with open(file_path, 'r', encoding='iso-8859-1') as file:
    data = json.load(file)

# Collect scores grouped by "top_p"
scores_by_top_p = {}
for category, entries in data.items():
    for entry in entries:
        top_k = entry["top_p"]
        rescaled_metrics = {
            "rescaled_entropy": entry["rescaled_entropy"],
            "rescaled_cosine_similarity": entry["rescaled_cosine_similarity"],
            "rescaled_kl_divergence": entry["rescaled_kl_divergence"],
        }
        if top_k not in scores_by_top_p:
            scores_by_top_p[top_k] = { "rescaled_entropy": [],
                                       "rescaled_cosine_similarity": [], 
                                       "rescaled_kl_divergence": []}
        for metric, value in rescaled_metrics.items():
            if value is not None:  # Exclude None values
                scores_by_top_p[top_k][metric].append(value)

# Calculate means for each metric grouped by "top_p"
means_by_top_p = {
    top_k: {metric: np.mean(values) if values else 0 for metric, values in metrics.items()}
    for top_k, metrics in scores_by_top_p.items()
}

# Sort by "top_p" for consistent plotting
sorted_top_p = sorted(means_by_top_p.keys())
metrics = ["rescaled_entropy", "rescaled_cosine_similarity", "rescaled_kl_divergence"]
means_sorted = {metric: [means_by_top_p[top_k][metric] for top_k in sorted_top_p] for metric in metrics}

# Plotting
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(sorted_top_p, means_sorted[metric], marker='o', label=metric)
plt.title("Mean Scores by top_p", fontsize=20, fontweight='bold')
plt.xlabel("Top_p", fontsize=16, fontweight='bold')
plt.ylabel("Mean Score", fontsize=16, fontweight='bold')
plt.xticks(sorted_top_p)
plt.legend(fontsize=16, title_fontsize=16)  # Legend font size adjustment
plt.grid(True)
plt.show()
