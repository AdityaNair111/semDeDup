# semDeDup

## Downloading the Data

use the following command to download the (first 32 shards) image embedding data and the corresponding metadata from HuggingFace:

`python ./data/download_data.py`

This will download the data and save it in the `data` folder.
(it assumes that you have the embds and metadata folders in the `data` folder)

## Running K-means using faiss

`python ./code/run_kmeans.py`

This will run k-means on the data and save cluster assignments and centroids in the `data` folder as `.npy` files.

## Running SemDeDup

`python ./code/main.py`

This will run SemDeDup on the data and save the results `image_path` metadata and a new bool column `dedup_retain` as a `.csv` file in the `data` folder.

## Misc

All parameters for SemDeDup are specified in `code/utils.py` file.