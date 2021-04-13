import argparse
import glob
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

from backend.preselection import load_sentence_embeddings, verify_embedding_structure


def load_corpus(dataset_path: str, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    dp = Path(dataset_path)
    assert dp.exists(), f"Cannot read {dataset_path}!"

    corpus = None
    if dp.is_dir():
        feathers = glob.glob(str(dp.joinpath("*.df.feather")))
        logger.info(f"Found {len(feathers)} DataFrames!")
        for feather in feathers:
            df = pd.read_feather(feather)
            assert 'caption' in df.columns, f"DataFrame {feather} has no column with name 'caption'!"
            if corpus is None:
                corpus = df
            else:
                corpus = pd.concat([corpus, df])
        logger.info(f"Loaded {len(corpus)} captions from {len(feathers)} DataFrames!")
    elif dp.is_file():
        corpus = pd.read_feather(dataset_path)
        assert 'caption' in corpus.columns, f"DataFrame {str(dp)} has no column with name 'caption'!"
    else:
        logger.error(f"Cannot read DataFrame from {dataset_path}!")
        sys.exit(1)

    # FIXME do not hard code this
    if dataset == 'wicsmmir':
        corpus['wikicaps_id'] = 'wikicaps_' + corpus['wikicaps_id'].astype(str)
    elif dataset == 'coco':
        corpus['image_id'] = 'coco_' + corpus['image_id'].astype(str)
    elif dataset == 'f30k':
        corpus['image_id'] = 'f30k_' + corpus['image_id'].astype(str)

    # image_id is for f30k and coco
    corpus_ids = corpus['wikicaps_id' if 'wikicaps_id' in corpus.columns else 'image_id'].to_numpy()

    return corpus['caption'].to_numpy(), corpus_ids


def generate_sentence_embeddings(dataset_name: str,
                                 dataset_path: str,
                                 out_path: str,
                                 symm_model: str,
                                 asym_model: str,
                                 max_seq_len: int) -> Dict[str, Dict[str, Any]]:
    corpus, corpus_ids = load_corpus(dataset_path)
    results = {'symmetric': None, 'asymmetric': None}

    op = Path(out_path)
    if not op.exists():
        logger.info(f"Creating {str(op)}")
        op.mkdir(parents=True, exist_ok=False)

    symm_dst = op.joinpath(f'{dataset_name}_symm_embs.pkl')
    asym_dst = op.joinpath(f'{dataset_name}_asym_embs.pkl')
    if symm_dst.exists():
        logger.warning(f'Symmetric Sentence Embeddings already exists at {str(symm_dst)}')
        results['symmetric'] = load_sentence_embeddings(symm_dst)
    if asym_dst.exists():
        logger.warning(f'Asymmetric Sentence Embeddings already exists at {str(asym_dst)}')
        results['asymmetric'] = load_sentence_embeddings(asym_dst)

    models = {'symmetric': symm_model, 'asymmetric': asym_model}
    for typ, model in models.items():
        if results[typ] is None:
            dst = asym_dst if 'asym' in typ else symm_dst
            logger.info(f"Loading Sentence Embedding Model '{model}' into Memory...")
            embedder = SentenceTransformer(model)

            # generate sentence embeddings
            embedder.max_seq_length = max_seq_len
            pool = embedder.start_multi_process_pool()
            start = time.time()
            logger.info(
                f"Computing {len(corpus)} {typ} Sentence Embeddings with {model} model! This may take a while...")
            # noinspection PyTypeChecker,PydanticTypeChecker
            embs = embedder.encode_multi_process(corpus, pool)
            emb_struct = {'corpus_ids': corpus_ids,
                          'embeddings': embs,
                          'type': typ,
                          'model': model,
                          'dataset': dataset_name}
            results[typ] = emb_struct

            # persist the embeddings
            with open(str(dst), "wb") as fOut:
                logger.info(f"Persisting {typ} Sentence Embeddings at {str(dst)}")
                pickle.dump(emb_struct, fOut, protocol=pickle.HIGHEST_PROTOCOL)

            embedder.stop_multi_process_pool(pool)
            logger.info(
                f"Computed {len(corpus)} {typ} Sentence Embeddings with {model} model in {time.time() - start}secs")
    return results


def generate_faiss_indices(dataset_name: str,
                           out_path: str,
                           faiss_num_cluster_N: int,
                           embedding_structs: Dict[str, Dict[str, Any]]):
    # TODO https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU

    op = Path(out_path)
    if not op.exists():  # should be already created in the embeddings generation method. but just to be safe!
        logger.info(f"Creating {str(op)}")
        op.mkdir(parents=True, exist_ok=False)

    symm_dst = op.joinpath(f'{dataset_name}_symm.faiss')
    asym_dst = op.joinpath(f'{dataset_name}_asym.faiss')

    for typ, embs in embedding_structs.items():
        dst = asym_dst if 'asym' in typ else symm_dst
        if dst.exists():
            logger.warning(f'{typ} FAISS Index for {embs["dataset"]} already exists at {str(dst)}')
            continue
        verify_embedding_structure(embs)

        emb = embs['embeddings']
        embedding_size = emb[0].shape[0]
        logger.debug(f"FAISS Index embedding size: {embedding_size}")
        # Number of clusters or Voronoi cells. Select a value 4*sqrt(len(corpus)) to 16*sqrt(len(corpus))
        n_clusters = int(faiss_num_cluster_N * np.sqrt(len(emb)))
        logger.debug(f"FAISS Index number of clusters: {n_clusters}")

        # we need to normalize vectors to unit length so that we can use dot product as distance measure
        emb = emb / np.linalg.norm(emb, axis=1)[:, None]

        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        # dot-product index as quantizer (to assign vectors to Voronoi cells)
        quantizer = faiss.IndexFlatIP(embedding_size)
        # inverted file flat index (Voronoi cells) with dot product metric
        index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)

        # train index ( create the VCs)
        logger.info(
            f"Training {typ} FAISS IVF Cluster (Voronoi Cell) Index for {embs['dataset']}. This may take a while...")
        index.train(emb)
        assert index.is_trained

        # add all embeddings to the index (i.e., add them to their respective VCs)
        logger.info(
            f"Adding embeddings to {typ} FAISS (Voronoi Cell) Index for {embs['dataset']}. This may take a while...")
        index.add(emb)

        # persist
        logger.info(f"Persisting {typ} FAISS Index at {str(dst)}")
        faiss.write_index(index, str(dst))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset DataFrame that contains the captions.\n'
                                                         'If this is a directory, all *.df.feather files in the '
                                                         'directory will be loaded, concatenated to compute the '
                                                         'sentence embeddings from.',
                        required=True)
    parser.add_argument('--dataset_name', default='wicsmmir', type=str, choices=['wicsmmir', 'coco', 'f30k'],
                        help='The name of the dataset. This will be reflected in the name of the generated output '
                             'files.')
    parser.add_argument('--out_path', default='data/sembs', type=str, help='Output path')
    # parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'],
    #                     help='Device on which the sentence embeddings get computed')

    # params to compute sentence embeddings
    parser.add_argument('--symm_model', default='paraphrase-distilroberta-base-v1', type=str,
                        help='Model to compute the embeddings for symmetric semantic text similarity search')
    parser.add_argument('--asym_model', default='msmarco-distilbert-base-v2', type=str,
                        help='Model to compute the embeddings for asymmetric semantic text similarity search')
    parser.add_argument('--max_seq_len', default=200, type=str,
                        help='A common value for BERT & Co. are 512 word pieces, corresponding to about 300-400 words')

    # params to compute FAISS Index
    parser.add_argument('--faiss_num_cluster_N', default=10, type=str,
                        help='Defines the number of clusters of the FAISS Index. n_clusters = N * np.sqrt(len(corpus))')

    opts = parser.parse_args()

    embeddings = generate_sentence_embeddings(opts.dataset_name,
                                              opts.dataset_path,
                                              opts.out_path,
                                              opts.symm_model,
                                              opts.asym_model,
                                              opts.max_seq_len)

    generate_faiss_indices(opts.dataset_name,
                           opts.out_path,
                           opts.faiss_num_cluster_N,
                           embeddings)
