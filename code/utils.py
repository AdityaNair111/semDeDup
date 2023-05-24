import numpy as np
import pandas as pd
import requests
from PIL import Image
import io

NUM_OF_SHARDS = 32
NUM_CLUSTERS = 10_000
N_ITERATIONS = 50
N_REDO = 5
EPSILON = 0.05

def get_embd_file_paths():
    file_paths = []
    for i in range(NUM_OF_SHARDS):
        file_paths.append(f'./data/embds/img_emb_{str(i).zfill(4)}.npy')
    return file_paths

def get_all_data():
    data = []
    for file_path in get_embd_file_paths():
        data.append(np.load(file_path))
    data = np.concatenate(data)
    return data

def get_metadata_file_paths():
    file_paths = []
    for i in range(NUM_OF_SHARDS):
        file_paths.append(f'./data/metadata/metadata_{str(i).zfill(4)}.parquet')
    return file_paths

def get_all_metadata():
    df = []
    for file_path in get_metadata_file_paths():
        df.append(pd.read_parquet(file_path))
    df = pd.concat(df)
    return df

def download_and_tile_images(retained_points, sorted_indices, cluster_assignments, i, all_metadata):
    try:
        if retained_points[sorted_indices][:10].sum() == 10:
            raise Exception("All images in this cluster were retained")

        cluster_assigments_index = (cluster_assignments == i).nonzero()[sorted_indices]
        download_and_save_images_for_a_cluster(all_metadata.iloc[cluster_assigments_index.flatten().numpy()], name=f'images/cluster_{i}_before_dedup')
        retained_cluster_assigments_index = (cluster_assignments == i).nonzero()[sorted_indices][retained_points[sorted_indices]]
        download_and_save_images_for_a_cluster(all_metadata.iloc[retained_cluster_assigments_index.flatten().numpy()], name=f'images/cluster_{i}_after_dedup')
        print(f"Saved images for cluster {i}")
    except Exception as e:
        print(e)
        print(f"Could not download images for cluster {i}")

def download_and_save_images_for_a_cluster(cluster_metadata, num_images_to_download=10, name='cluster'):    

    num_images_to_download = min(num_images_to_download, len(cluster_metadata))
    
    # Download and save images
    images = []
    for i, row in cluster_metadata.iterrows():
        response = requests.get(row['url'])
        img = Image.open(io.BytesIO(response.content)).convert('RGB') # convert to RGB format
        # resize image to 224x224
        img = img.resize((224, 224))
        images.append(np.array(img))
        if len(images) == num_images_to_download:
            break
    
    # save this list of images as one image file with all images in one roww
    concatenated_images = np.concatenate(images, axis=1)
    Image.fromarray(concatenated_images).save(f'{name}_{num_images_to_download}.jpg')