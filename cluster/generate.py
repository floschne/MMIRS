import argparse
import glob
import multiprocessing as mp
import pickle
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from cluster.visual import ImageMetadata


def generate_metadata(feat_path, out_path):
    # build metadata instance
    im = ImageMetadata(feat_path)

    # persist
    with open(Path(out_path).joinpath(im.img_id + '.mdata'), 'wb') as out:
        pickle.dump(im, out)


def generate_image_metadata(feats_path, out_path, num_workers):
    assert Path(feats_path).exists(), f"Cannot read {feats_path}"
    feats = glob.glob(feats_path + '/*.npz')
    logger.info(f"Found {len(feats)} feature archives")

    op = Path(out_path)
    if not op.exists():
        logger.info(f"Creating {out_path}")
        op.mkdir(parents=True, exist_ok=False)

    with mp.Pool(num_workers) as pool:
        results = [pool.apply_async(generate_metadata, args=(f, out_path,)) for f in feats]
        results = [r.get() for r in tqdm(results)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', type=str, choices=['img_metadata'], default='img_metadata',
                        help='Operation to execute')
    parser.add_argument('--feats_path', type=str, help='Path to the feature npz archives', required=True)
    parser.add_argument('--out_path', default='data/cluster/visual/', type=str, help='Output path')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of parallel workers')

    opts = parser.parse_args()
    if opts.op == 'img_metadata':
        generate_image_metadata(opts.feats_path, opts.out_path, opts.num_workers)
