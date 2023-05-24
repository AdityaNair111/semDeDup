import os
import numpy as np
import faiss
from utils import get_all_data, NUM_CLUSTERS, N_ITERATIONS, N_REDO

def cluster_and_save_data(k=NUM_CLUSTERS, n_iter=N_ITERATIONS,
                     n_redo=N_REDO, save_path='./data/'):

    np.random.seed(0)  # For reproducibility

    data = get_all_data() # returns data as a numpy array
    print('Data shape:', data.shape)
    d = data.shape[1]  # Dimension of the vectors in the data

    # Initializing a CPU index
    index = faiss.IndexFlatL2(d)

    # Moving the index to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

    print('Training the index on GPU...')
    # Initialize Clustering on GPU index
    kmeans = faiss.Clustering(d, k)
    kmeans.verbose = True  # If True, the clustering process will output verbose messages
    kmeans.niter = n_iter  # Number of iterations for the clustering
    kmeans.spherical = True  # Normalizes the input vectors to unit length before clustering
    kmeans.nredo = n_redo  # Number of times the clustering algorithm will be run, with random initializations

    # Train the Clustering algorithm on the data using the GPU index
    kmeans.train(data, gpu_index)

    # Get the cluster centroids
    centroids = faiss.vector_to_array(kmeans.centroids)
    centroids = centroids.reshape(k, d)
    print('Cluster centroids:', centroids.shape)

    # Assign the data points to clusters using GPU index
    gpu_index.add(centroids)
    _, I = gpu_index.search(data, 1)

    # save the cluster centroids & assignments in input save_path
    np.save(os.path.join(save_path, 'cluster_centroids.npy'), centroids)
    np.save(os.path.join(save_path, 'cluster_assignments.npy'), I.flatten())

    print(f'Cluster centroids and assignments saved in:{save_path}')

if __name__ == '__main__':
    cluster_and_save_data()