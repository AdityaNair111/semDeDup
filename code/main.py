import os
import torch
import numpy as np
from tqdm import tqdm
from utils import (
    get_all_data, 
    get_all_metadata, 
    download_and_tile_images,
    NUM_CLUSTERS,
    EPSILON,
)

@torch.no_grad()
def sort_by_distance_to_cluster_centroid(cluster_embeddings, cluster_centroid):
    # L2 distance is used because this is the distance used during k-means
    cosine_distances = ((cluster_embeddings.float() @ cluster_centroid.float()))
    sorted_indices = cosine_distances.argsort(descending=True)
    return cluster_embeddings[sorted_indices], sorted_indices

@torch.no_grad()
def dedup_cluster(cluster_embeddings, cluster_centroid, epsilon, batch_size=10_000, verbose=False):
    # transfer to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cluster_embeddings, cluster_centroid = cluster_embeddings.to(device), cluster_centroid.to(device)

    # normalize and sort embeddings closest to cluster centroid
    cluster_embeddings = torch.nn.functional.normalize(cluster_embeddings, p=2, dim=-1)
    cluster_embeddings, sorted_indices = sort_by_distance_to_cluster_centroid(cluster_embeddings, cluster_centroid)

    # calcualte similairity batchwise to avoid blowing up GPU memory
    num_rows = cluster_embeddings.shape[0]
    points_to_keep_from_cluster = torch.ones(num_rows, dtype=torch.bool)
    for i in range(0, num_rows, batch_size):
        end_i = min(i+batch_size, num_rows) # last index of batch

        # calculate similarity matrix for batch
        pairwise_sim_matrix = cluster_embeddings @ cluster_embeddings[i:end_i].T
        triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal=1-i) # -i + 1 to get the correct diagonal (when the matrix is not square)
        M = torch.max(triu_sim_matrix, dim=0)[0]

        # keep points that are not too similar to other points in the cluster
        points_to_keep_from_cluster[i:end_i] = M <= 1 - epsilon

    if verbose:
        kept_points = points_to_keep_from_cluster.nonzero()
        print(f"Data Points before : {len(cluster_embeddings)}, data points after deduping: {len(kept_points)}")
        print(f"{round(len(kept_points) / len(cluster_embeddings) * 100,2)}% of data points kept from this cluster")

    # Also return the sorted indicies for tiling images later
    return points_to_keep_from_cluster[sorted_indices.argsort().cpu()], sorted_indices.cpu().flatten().numpy()

@torch.no_grad()
def dedup_all_clusters(save_example_images=False, data_folder='./data', epsilon=EPSILON):
    # All embedding and metadata are loaded here but they can also be loaded as need if memory is an issue
    all_embeddings = torch.from_numpy(get_all_data())
    all_metadata = get_all_metadata()
    cluster_centroids = torch.from_numpy(np.load(os.path.join(data_folder, 'cluster_centroids.npy')))
    cluster_assignments = torch.from_numpy(np.load(os.path.join(data_folder, 'cluster_assignments.npy')))

    print(f'All embeddings shape: {all_embeddings.shape}')
    print(f'Cluster centroids shape: {cluster_centroids.shape}')
    print(f'Cluster assignments shape: {cluster_assignments.shape}')

    # add new column to metadata dataframe
    all_metadata['dedup_retain'] = [True] * len(all_metadata)
    bins = cluster_assignments.bincount().sort(descending=True)[1].tolist()

    for it in tqdm(range(NUM_CLUSTERS)):
        i = bins[it] # from largest to smallest cluster just to ensure if there are any memory issues they surface early
        retained_points, sorted_indices = dedup_cluster(all_embeddings[cluster_assignments == i], cluster_centroids[i], epsilon=epsilon)
        all_metadata.loc[cluster_assignments.numpy() == i, 'dedup_retain'] = retained_points.cpu().numpy()
        if save_example_images:
            download_and_tile_images(retained_points, sorted_indices, cluster_assignments, i, all_metadata)
        
    print(f"Number of points retained before deduping: {len(all_metadata)}")
    print(f"Number of points retained after deduping: {len(all_metadata[all_metadata['dedup_retain'] == True])}")
    print(f"Percentage of points retained after deduping: {round(len(all_metadata[all_metadata['dedup_retain'] == True]) / len(all_metadata) * 100,2)}%")

    # save deduped metadata
    all_metadata[['image_path', 'dedup_retain']].to_csv(os.path.join(data_folder, 'deduped_metadata.csv'), index=False)

if __name__ == "__main__":
    dedup_all_clusters()