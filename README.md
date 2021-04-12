# Multi-Modal Image Retrieval (MMIR)
This repo holds the code for my Master Thesis at Language Technology (LT) Group of the University of Hamburg  

## How to run locally
_Assuming your in the root folder of this repository_

#### Install pre-requisites
```
conda env create -v --file environment.yaml

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### Run the api locally
Make sure to change the path according to your setup!
For more control edit: `config.yaml` _(you can also set the data paths there and omit the environment variables)_

Start the image server with the minimal config:
```
docker-compose -f docker-compose_img_svr.yml up -d
```

Start retrieval API
```
conda activate mt
export UNITER_DATA_DIR=/home/p0w3r/gitrepos/UNITER_fork/downloads
export UNITER_MODEL_CONFIG=/home/p0w3r/gitrepos/UNITER_fork/config/uniter-base.json
uvicorn main:app --host 0.0.0.0 --port 8081
```
