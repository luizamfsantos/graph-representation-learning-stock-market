import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# load data
folder_path = Path(__file__).parent.parent / 'data'
df = pd.read_csv(folder_path / 'loss_results.csv')

# Plot Loss Percentage vs Number of Clusters for Different rho_min
# plt.rcParams['text.usetex'] = True
plt.figure(figsize=(10, 6))
for rho, group in df.groupby('rho_min'):
    plt.plot(group['n_clusters'], group['loss_percentage'], marker='o', label=f"rho_min = {rho}")

plt.title("Loss Percentage vs Number of Clusters for Different rho_min")
plt.xlabel("Number of Clusters")
plt.ylabel("Loss Percentage")
plt.legend(title="rho_min")
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot Loss Percentage vs Different rho_min for different Number of Clusters
plt.figure(figsize=(10, 6))
for n_clusters, group in df.groupby('n_clusters'):
    plt.plot(group['rho_min'], group['loss_percentage'], marker='', label=f"n_clusters = {n_clusters}")

plt.title("Loss Percentage at different levels for different number of clusters")
plt.xlabel("Rho_min")
plt.ylabel("Loss Percentage")
plt.legend(title="n_clusters")
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()