import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Load data
folder_path = Path(__file__).parent.parent / 'data'
df = pd.read_csv(folder_path / 'loss_results.csv')

# Apply seaborn style
sns.set_theme(style="whitegrid")

# # Plot Loss Percentage vs Number of Clusters for Different rho_min
# plt.figure(figsize=(12, 6))
# for rho, group in df.groupby('rho_min'):
#     plt.plot(
#         group['n_clusters'], 
#         group['misses_over_total_edges'] * 100, 
#         marker='o', linestyle='-', label=f"rho_min = {rho} (Total Edges)"
#     )
#     plt.plot(
#         group['n_clusters'], 
#         group['misses_over_total_high_corr'] * 100, 
#         marker='x', linestyle='--', label=f"rho_min = {rho} (High Corr)"
#     )

# plt.title("Loss Percentage vs Number of Clusters for Different rho_min", fontsize=14, pad=15)
# plt.xlabel("Number of Clusters", fontsize=12)
# plt.ylabel("Loss Percentage (%)", fontsize=12)
# plt.legend(title="rho_min", fontsize=10, title_fontsize=12, loc='upper right')
# plt.grid(visible=True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # Plot Loss Percentage vs Different rho_min for Different Number of Clusters
# plt.figure(figsize=(12, 6))
# for n_clusters, group in df.groupby('n_clusters'):
#     plt.plot(
#         group['rho_min'], 
#         group['misses_over_total_edges'] * 100, 
#         marker='o', linestyle='-', label=f"n_clusters = {n_clusters} (Total Edges)"
#     )
#     plt.plot(
#         group['rho_min'], 
#         group['misses_over_total_high_corr'] * 100, 
#         marker='x', linestyle='--', label=f"n_clusters = {n_clusters} (High Corr)"
#     )

# plt.title("Loss Percentage vs rho_min for Different Number of Clusters", fontsize=14, pad=15)
# plt.xlabel("Rho_min", fontsize=12)
# plt.ylabel("Loss Percentage (%)", fontsize=12)
# plt.legend(title="n_clusters", fontsize=10, title_fontsize=12, loc='upper left')
# plt.grid(visible=True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


# Convert the cumulative loss to density
for n_clusters, group in df.groupby('n_clusters'):
    group = group.sort_values('rho_min')
    group['misses_over_total_edges'] = group['misses_over_total_edges'].shift(1, fill_value=0) - group['misses_over_total_edges']
    group['misses_over_total_high_corr'] = group['misses_over_total_high_corr'].shift(1, fill_value=0) - group['misses_over_total_high_corr']
    plt.hist(group['misses_over_total_edges'], bins=10, alpha=0.5, label=f"n_clusters = {n_clusters} (Total Edges)")
    plt.hist(group['misses_over_total_high_corr'], bins=10, alpha=0.5, label=f"n_clusters = {n_clusters} (High Corr)")

plt.title("Density of Loss Percentage for Different Number of Clusters", fontsize=14, pad=15)
plt.xlabel("Loss Percentage", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(title="n_clusters", fontsize=10, title_fontsize=12, loc='upper right')
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
