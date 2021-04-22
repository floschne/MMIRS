import argparse
import os

import requests
from tqdm import tqdm


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
    parser.add_argument('--out_path', default='data/magnitude', type=str, help='Output path')
    opts = parser.parse_args()

    download_magnitude_embeddings(opts.out_path)
