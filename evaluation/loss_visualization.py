import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# load data
folder_path = Path(__file__).parent.parent / 'data'
df = pd.read_csv(folder_path / 'loss_results.csv')

# Plot
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
