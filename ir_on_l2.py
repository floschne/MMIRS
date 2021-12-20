import argparse
import glob
import os
import shutil
import sys
from typing import List, Optional

import pandas as pd
from loguru import logger
from pandas import DataFrame
from tqdm import tqdm

from api.model import RetrievalRequest
from backend import MMIRS
from backend.fineselection.data import ImageFeaturePoolFactory
from backend.fineselection.retriever import RetrieverFactory
from backend.imgserver.py_http_image_server import PyHttpImageServer
from config import conf


def load_dataset(path: str, use_focus: bool = False) -> DataFrame:
    logger.info(f"Reading DataFrame at {path}...")
    assert os.path.lexists(path), f"Cannot read dataframe at {path}"

    # load the dataset
    df = pd.read_feather(path)

    assert 'caption' in df.columns and 'dataset' in df.columns, \
        "Dataframe does not contain 'caption' AND 'dataset' columns"
    if use_focus:
        assert "focus" in df.columns, "Dataframe does not contain a `focus` column!"

    return df


def persist_top_k_results_in_result_dataframe(top_k_results: List[List[str]],
                                              df: DataFrame,
                                              opts: argparse.Namespace) -> str:
    # generate filename for results df
    if opts.use_focus:
        fn = f'top_k_images_{opts.retriever_name}_{opts.dataset}_fw_{opts.focus_weight}.df.feather'
    else:
        fn = f'top_k_images_{opts.retriever_name}_{opts.dataset}.df.feather'
    os.makedirs(opts.output_path, exist_ok=True)
    fn = os.path.join(opts.output_path, fn)

    # save the top k in the results dataframe
    new_df = df[:len(top_k_results)]
    new_df['top_k'] = top_k_results

    # persist
    logger.info(f"Persisting results at {fn}")
    new_df.reset_index(drop=True).to_feather(os.path.join(opts.output_path, fn))

    return fn


def prepare_config(opts: argparse.Namespace) -> None:
    logger.info("Preparing config...")
    not_selected_ds = []
    selected = opts.image_dataset
    for ds in conf.image_server.datasources:
        if ds != selected:
            not_selected_ds.append(ds)

    # remove all config for other image datasets than the selected one
    for nds in not_selected_ds:
        del conf.image_server.datasources[nds]
        del conf.preselection.focus.wtf_idf[nds]
        del conf.preselection.context.sbert.symmetric_embeddings[nds]
        del conf.preselection.context.sbert.asymmetric_embeddings[nds]
        del conf.preselection.context.faiss.symmetric_indices[nds]
        del conf.preselection.context.faiss.asymmetric_indices[nds]
        del conf.fine_selection.feature_pools[nds]
        del conf.fine_selection.feature_pools[selected][f"teran_{nds}"]
        del conf.fine_selection.retrievers[f"teran_{nds}"]
        del conf.fine_selection.max_focus_annotator.datasources[nds]

    # prefetch the selected image dataset in RAM
    conf.fine_selection.feature_pools[selected][opts.retriever_name].pre_fetch = True


def build_retrieval_requests(df: DataFrame, opts: argparse.Namespace) -> List[RetrievalRequest]:
    reqs = []
    for idx, row in tqdm(df.iterrows(), desc="Building RetrievalRequests for each sample!", total=len(df)):
        req = RetrievalRequest(context=row['caption'],
                               focus=row['focus'] if opts.use_focus else None,
                               top_k=opts.top_k,
                               retriever=opts.retriever_name,
                               dataset=opts.image_dataset,
                               annotate_max_focus_region=opts.annotate_max_focus_region,
                               ranked_by=opts.ranking_method,
                               focus_weight=0.0 if opts.use_focus else opts.focus_weight,
                               return_scores=opts.return_scores,
                               return_wra_matrices=opts.return_wra_matrices,
                               return_timings=opts.return_timings)
        reqs.append(req)
    return reqs


def run_no_pss_retrieval(df: DataFrame, opts: argparse.Namespace) -> str:
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
                                          top_k=opts.top_k,
                                          iss=iss)
        top_k_image_ids = res['top_k'][opts.ranking_method]
        top_k_results.append(top_k_image_ids)

        if idx % opts.persist_step:
            persist_top_k_results_in_result_dataframe(top_k_results=top_k_results,
                                                      df=df,
                                                      opts=opts)

    return persist_top_k_results_in_result_dataframe(top_k_results=top_k_results,
                                                     df=df,
                                                     opts=opts)


def safe_annotated_images(top_k_img_ids: List[str],
                          opts: argparse.Namespace,
                          id_prefix: Optional[str] = None) -> List[str]:
    dst_root_p = conf.fine_selection.max_focus_annotator.annotated_images_dst
    top_k_img_ids_with_prefix = []
    # copy images from root_p to output_path and append id_prefix
    for img_id in top_k_img_ids:
        img_files = glob.glob(f"{dst_root_p}/{img_id}*")
        for f in img_files:
            fn_with_prefix = f"{id_prefix}_{os.path.basename(f)}"
            dst = os.path.join(opts.output_path, fn_with_prefix)
            shutil.move(f, dst)
            top_k_img_ids_with_prefix.append(fn_with_prefix)
    return top_k_img_ids_with_prefix


def run_retrieval_with_pss(df: DataFrame, opts: argparse.Namespace) -> str:
    prepare_config(opts)
    # build the retrieval request list from the dataframe
    reqs = build_retrieval_requests(df, opts)

    # create the output path
    os.makedirs(opts.output_path, exist_ok=True)

    # start MMIRS
    mmirs = MMIRS()

    img_srv = PyHttpImageServer()

    logger.info(f"Starting retrieval of {len(reqs)} samples!")
    top_k_results = []
    for idx, req in tqdm(enumerate(reqs), desc="Retrieval progress: ", total=len(reqs)):
        if opts.return_wra_matrices:
            top_k_img_urls, top_k_wra_urls = MMIRS().retrieve_top_k_images(req)
        else:
            top_k_img_urls = mmirs.retrieve_top_k_images(req)

        top_k_img_ids = img_srv.get_image_ids(top_k_img_urls)
        top_k_img_ids = safe_annotated_images(top_k_img_ids, opts, id_prefix=str(idx))

        top_k_results.append(top_k_img_ids)

        if idx % opts.persist_step:
            persist_top_k_results_in_result_dataframe(top_k_results=top_k_results,
                                                      df=df,
                                                      opts=opts)

    return persist_top_k_results_in_result_dataframe(top_k_results=top_k_results,
                                                     df=df,
                                                     opts=opts)


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
    parser.add_argument('--output_path',
                        type=str,
                        help='Path where the result dataframe and annotated images are saved',
                        default="/tmp/mmirs_out")
    parser.add_argument('--ranking_method',
                        type=str,
                        help=("Method that is used to retrieve and rank the images. If no_pss is selected, only FSS"
                              " is used to retrieve the images, which can be more accurate but takes A LOT OF TIME."
                              " This currently does not support annotating focus regions etc."),
                        choices=['focus', 'context', 'combined', "no_pss"],
                        default="combined")
    parser.add_argument('--use_focus', action='store_true', default=False)
    parser.add_argument('--focus_weight', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--annotate_max_focus_region', action='store_true', default=True)
    parser.add_argument('--return_scores', action='store_true', default=False)
    parser.add_argument('--return_wra_matrices', action='store_true', default=False)
    parser.add_argument('--return_timings', action='store_true', default=False)
    parser.add_argument('--log_level',
                        type=str,
                        choices=['info', 'debug', 'error', "warning"],
                        default="info")
    parser.add_argument('--persist_step', type=int, default=100)
    opts = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=opts.log_level.upper())
    logger.add(opts.output_path + '/logs/{time}.log',
               rotation=f"{conf.logging.max_file_size} MB",
               level=opts.log_level.upper())

    df = load_dataset(opts.dataset_path, opts.use_focus)

    if opts.ranking_method == "no_pss":
        run_no_pss_retrieval(df, opts)
    else:
        run_retrieval_with_pss(df, opts)
