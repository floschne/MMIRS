import argparse
import os

import pandas as pd
from loguru import logger
from tqdm import tqdm

from backend.fineselection.data import ImageFeaturePoolFactory
from backend.fineselection.retriever import RetrieverFactory


def load_dataset(path: str):
    logger.info(f"Reading DataFrame at {path}...")
    assert os.path.lexists(path), f"Cannot read dataframe at {path}"

    # load the dataset
    df = pd.read_feather(path)

    assert 'caption' in df.columns and 'dataset' in df.columns, \
        "Dataframe does not contain 'caption' AND 'dataset' columns"

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        type=str,
                        help='Path to the dataset DataFrame that contains the captions and focus.',
                        required=True)
    parser.add_argument('--image_dataset',
                        type=str,
                        help="The image dataset from which the best matching images are retrieved!",
                        choices=['wicsmmir', 'coco', 'f30k'],
                        required=True)
    parser.add_argument('--retriever_name',
                        type=str,
                        help="The retriever model that's used to find the best matching images",
                        choices=['teran_wicsmmir', 'teran_coco', 'teran_f30k'],
                        required=True)
    parser.add_argument('--use_focus', action='store_true', default=False)
    parser.add_argument('--focus_weight', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=50)

    opts = parser.parse_args()

    df = load_dataset(opts.dataset_path)
    if opts.use_focus:
        assert "focus" in df.columns, "Dataframe does not contain a `focus` column!"

    # instantiate retriever and image pool
    retriever_factory = RetrieverFactory()
    retriever = retriever_factory.create_or_get_retriever(opts.retriever_name)

    # setup and build image pools
    pool_factory = ImageFeaturePoolFactory()
    pool = pool_factory.create_or_get_pool(opts.image_dataset, retriever.retriever_name)

    # build and load the image search space (into memory!) containing ALL IMAGES OF THE POOL!
    iss = pool.get_image_search_space(img_ids=None)

    top_k_results = []
    # run IR for all captions
    for idx, row in tqdm(df.iterrows(), desc="Running image retrieval on each caption", total=len(df)):
        # do the retrieval
        res = retriever.find_top_k_images(focus=row['focus'] if opts.use_focus else None,
                                          context=row['caption'],
                                          focus_weight=0.0 if opts.use_focus else opts.focus_weight,
                                          top_k=50,
                                          iss=iss)
        top_k_results.append(res)

    # save the top k in the dataframe
    df['top_k'] = top_k_results

    # persist
    if opts.use_focus:
        fn = f'top_k_images_{opts.retriever_name}_{opts.dataset}_fw_{opts.focus_weight}.df.feather'
    else:
        fn = f'top_k_images_{opts.retriever_name}_{opts.dataset}.df.feather'

    logger.info(f"Persisting results at {fn}")
    df.to_feather(fn)
