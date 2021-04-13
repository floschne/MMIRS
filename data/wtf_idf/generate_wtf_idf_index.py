import argparse
import concurrent.futures as cf
import glob
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from backend.preselection.focus.image_metadata import ImageMetadata
from backend.preselection import VisualVocab


def generate_metadata(feat_path: str,
                      octh: float,
                      acth: float,
                      alpha: float,
                      out_path: str,
                      persist: bool = False) -> ImageMetadata:
    # build metadata instance
    vocab = VisualVocab()
    im = ImageMetadata(feat_path, vocab, octh, acth, alpha)

    # persist
    if persist:
        with open(Path(out_path).joinpath(im.img_id + '.mdata'), 'wb') as out:
            pickle.dump(im, out)

    return im


def build_wtf_idf_index(docs: List[ImageMetadata], dst: Path) -> None:
    N = len(docs)
    # get all terms of all docs
    terms = []
    for doc in docs:
        terms.extend(doc.terms)
    terms = set(terms)

    # document frequency (how many docs contain t)
    df = {t: 0 for t in terms}
    for t in tqdm(terms, desc="Computing Document Frequencies"):
        for d in docs:
            if t in d.terms:
                df[t] += 1
    assert max(df.values()) <= N

    # smoothed, idf
    # idf(t) = log(N/(df + 1))
    idf = {t: np.log(N / (df[t] + 1)) for t in df.keys()}
    assert min(idf.values()) >= 0.

    # weighted term-frequency
    wtf = {}
    for d in tqdm(docs, desc="Collecting weighted term-frequencies"):
        for t, td in d.term_data.items():
            wtf[t, d.img_id] = td.wtf
    assert min(wtf.values()) >= 0.

    # weighted tf-idf
    # wtf-idf(t, d) = wtf(t, d) * idf(t)
    wtf_idf = {}
    for t_f in tqdm(wtf, desc="Computing weighted tf-idf"):
        wtf_idf[t_f] = wtf[t_f] * idf[t_f[0]]
    # sort alphabetically by terms
    wtf_idf = {k: v for k, v in sorted(wtf_idf.items(), key=lambda i: i[0][0])}

    # create DataFrame
    rows = []
    for td, v in tqdm(wtf_idf.items(), desc="Building wtf-idf Index Structure"):
        rows.append((*td, v))
    index = pd.DataFrame(data=rows, columns=['term', 'doc', 'wtf_idf'])
    index.set_index(['term'])

    # persist
    index.to_feather(dst)
    logger.info(f"Successfully persisted WTF-IDF Index at {dst}")


def generate_wtf_idf_index(feats_path: str,
                           out_path: str,
                           octh: float,
                           acth: float,
                           alpha: float,
                           persist: bool,
                           num_workers: int):
    op = Path(out_path)
    dst = op.joinpath(f'wtf_idf_octh_{octh:0.2f}_acth_{acth:0.2f}_alpha_{alpha:0.2f}.index')
    if dst.exists():
        logger.info(f'Index already exists at {str(dst)}')
        return

    if not op.exists():
        logger.info(f"Creating {str(op)}")
        op.mkdir(parents=True, exist_ok=False)

    assert Path(feats_path).exists(), f"Cannot read {feats_path}"
    feats = glob.glob(feats_path + '/*.npz')
    logger.info(f"Found {len(feats)} feature archives!")

    with cf.ProcessPoolExecutor(max_workers=num_workers) as ex:
        with tqdm(total=len(feats), desc="Generating Image Metadata") as pbar:
            futures = []
            for feat in feats:
                future = ex.submit(generate_metadata, feat, octh, acth, alpha, out_path, persist)
                future.add_done_callback(lambda p: pbar.update(1))
                futures.append(future)

            docs = [f.result() for f in cf.as_completed(futures)]

    build_wtf_idf_index(docs, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats_path', default='data/features', type=str, help='Path to the feature npz archives')
    parser.add_argument('--out_path', default='data/wtf_idf', type=str, help='Output path')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of parallel workers')
    parser.add_argument('--persist_metadata', default=False, action='store_true',
                        help='If True, ImageMetadata gets persisted.')

    # params to compute weighted term freqs
    parser.add_argument('--octh', default=.2, type=float, help='Object confidence threshold. Objects detected with '
                                                               'less confidence than octh get ignored as terms.')
    parser.add_argument('--acth', default=.15, type=float, help='Attribute confidence threshold. Attribute detected '
                                                                'with less '
                                                                'confidence than octh get ignored as terms.')
    parser.add_argument('--alpha', default=.95, type=float, help='Weight for weighted term-frequency. weight = alpha * '
                                                                 'conf + (1-alpha) * area')
    opts = parser.parse_args()

    generate_wtf_idf_index(opts.feats_path,
                           opts.out_path,
                           opts.octh,
                           opts.acth,
                           opts.alpha,
                           opts.persist_metadata,
                           opts.num_workers)
