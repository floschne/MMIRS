# Master Thesis Code
This repo holds the code for my Master Thesis at Language Technology (LT) Group of the University of Hamburg  

## How to run locally

_Assuming that your in the root folder of this repository_
```
conda env create -v --file environment.yaml
conda activate mt
uvicorn main:app --host 0.0.0.0 --port 8081
```