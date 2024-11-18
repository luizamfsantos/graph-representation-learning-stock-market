import torch 
import numpy as np 
from sklearn.cluster import KMeans

def cluster_embeddings(
    embeddings: np.ndarray, 
    n_clusters:int=3
    ) -> np.ndarray:
    """ Perform KMeans clustering on node embeddings. """
    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=10, # multiple initializations
        random_state=0 # for reproducibility
    )

    cluster_labels = kmeans.fit_predict(embeddings)

    return cluster_labels, kmeans 