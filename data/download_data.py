import urllib.request

NUM_SHARDS = 32

for i in range(NUM_SHARDS):
    print(f"Downloading image embeddings {i}/{NUM_SHARDS}")
    url = f"https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings/resolve/main/img_emb/img_emb_{str(i).zfill(4)}.npy"
    filename = f"data/embds/img_emb_{str(i).zfill(4)}.npy"
    urllib.request.urlretrieve(url, filename)

for i in range(NUM_SHARDS):
    print(f"Downloading image metadata {i}/{NUM_SHARDS}")
    url = f"https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings/resolve/main/metadata/metadata_{str(i).zfill(4)}.parquet"
    filename = f"data/metadata/metadata_{str(i).zfill(4)}.parquet"
    urllib.request.urlretrieve(url, filename)
    