import argparse
import glob
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

from cluster.focus_word import ImageMetadata
from cluster.focus_word.vocab import Vocab
from cluster.focus_word.wtfidf import build_wtf_idf_index


def generate_metadata(feat_path: str, vocab: Vocab, out_path: str, persist: bool = False) -> ImageMetadata:
    # build metadata instance
    im = ImageMetadata(feat_path, vocab)

    # persist
    if persist:
        with open(Path(out_path).joinpath(im.img_id + '.mdata'), 'wb') as out:
            pickle.dump(im, out)

    return im


def generate_wtf_idf_index(feats_path, vocab_path, out_path, octh, acth, alpha, num_workers):
    assert Path(feats_path).exists(), f"Cannot read {feats_path}"
    feats = glob.glob(feats_path + '/*.npz')
    logger.info(f"Found {len(feats)} feature archives!")

    op = Path(out_path)
    if not op.exists():
        logger.info(f"Creating {out_path}")
        op.mkdir(parents=True, exist_ok=False)

    vocab = Vocab(vocab_path=vocab_path)

    with mp.Pool(num_workers) as pool:
        docs = [pool.apply_async(generate_metadata, args=(f, vocab, out_path,)) for f in feats]
        docs = [r.get() for r in tqdm(docs, desc="Generating ImageMetadata...")]

    build_wtf_idf_index(docs, op)


# https://gist.github.com/wy193777/0e2a4932e81afc6aa4c8f7a2984f34e2
def download_magnitude_embeddings(out_path: str = None):
    url = "http://magnitude.plasticity.ai/fasttext/heavy/crawl-300d-2M.magnitude"

    if out_path is None:
        out_path = os.getcwd()
    if out_path[-1] != '/':
        out_path += '/'
    dst = out_path + url.split('/')[-1]

    file_size = int(requests.head(url).headers["Content-Length"])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(total=file_size,
                initial=first_byte,
                unit='B',
                unit_scale=True,
                position=0,
                leave=True,
                desc="Downloading Magnitude Embeddings")
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()

    return file_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', type=str, choices=['download_embeddings', 'generate_index'],
                        default='generate_index',
                        help='The operation to execute.')
    parser.add_argument('--feats_path', default='data/features', type=str, help='Path to the feature npz archives')
    parser.add_argument('--vocab_path', default='cluster/focus_word/', type=str, help='Path to the vocabulary files')
    parser.add_argument('--out_path', default='data/', type=str, help='Output path')
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

    if opts.operation == 'generate_index':
        generate_wtf_idf_index(opts.feats_path,
                               opts.vocab_path,
                               opts.out_path,
                               opts.octh,
                               opts.acth,
                               opts.alpha,
                               opts.num_workers)
    if opts.operation == 'download_embeddings':
        download_magnitude_embeddings(opts.out_path)
