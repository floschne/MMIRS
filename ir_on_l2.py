import argparse
import os

import pandas as pd
from loguru import logger
from tqdm import tqdm

from backend.fineselection.data import ImageFeaturePoolFactory
from backend.fineselection.retriever import RetrieverFactory


def load_dataset(path: str):
    assert os.path.lexists(path), f"Cannot read dataframe at {path}"

    # load the dataset
    df = pd.read_feather(opts.dataset_path)

    assert 'caption' in df.columns and 'dataset' in df.columns, \
        "Dataframe does not contain 'caption' AND 'dataset' columns"

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        type=str,
                        help='Path to the dataset DataFrame that contains the captions.',
                        required=True)
    parser.add_argument('--image_batch_size', type=int, default=5000)
    parser.add_argument('--retriever_name', type=str, default='teran_wicsmmir',
                        choices=['teran_wicsmmir', 'teran_coco', 'teran_f30k'])
    parser.add_argument('--dataset', type=str, default='wicsmmir', choices=['wicsmmir', 'coco', 'f30k'])

    opts = parser.parse_args()

    df = load_dataset(opts.dataset_path)

    # instantiate retriever and image pool
    retriever_factory = RetrieverFactory()
    retriever = retriever_factory.create_or_get_retriever(opts.retriever_name)

    # setup and build image pools
    pool_factory = ImageFeaturePoolFactory()
    pool = pool_factory.create_or_get_pool(opts.dataset, retriever.retriever_name)

    # build and load the image search space (into memory!) containing ALL IMAGES OF THE POOL!
    iss = pool.get_image_search_space(img_ids=None)

    top_k_results = []
    # run IR for all captions
    for row in tqdm(df.iterrows(), desc="Running image retrieval on each caption", total=len(df)):
        # do the retrieval
        res = retriever.find_top_k_images(focus=None,
                                          context=row[1]['caption'],
                                          focus_weight=0.0,
                                          top_k=50,
                                          iss=iss)
        top_k_results.append(res)

    # save the top k in the dataframe
    df['top_k'] = top_k_results

    # persist
    fn = f'top_k_images_{opts.retriever_name}_{opts.dataset}.df.feather'
    logger.info(f"Persisting results at {fn}")
    df.to_feather(fn)
