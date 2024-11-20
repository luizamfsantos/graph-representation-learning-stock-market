import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
folder_path = Path(__file__).parent.parent / 'data'
corr_matrix = pd.read_csv(folder_path / 'correlation_matrix.csv', index_col=0)

# Flatten the correlation matrix to get the density values
density_values = corr_matrix.values.flatten()

# Create a density plot of the density values
plt.figure(figsize=(10, 6))
sns.kdeplot(density_values, shade=True)
plt.title('Density Plot of Correlation Matrix Values')
plt.xlabel('Correlation Value')
plt.ylabel('Density')
plt.show()

# Create a cumulative distribution plot of the density values
plt.figure(figsize=(10, 6))
sns.kdeplot(density_values, cumulative=True, shade=True)
plt.title('Cumulative Distribution Plot of Correlation Matrix Values')
plt.xlabel('Correlation Value')
plt.ylabel('Cumulative Density')
plt.show()