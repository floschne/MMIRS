import sys
from types import SimpleNamespace
from typing import List

from api.model import RetrievalRequest
from backend.retriever import Retriever
from logger import backend_logger

sys.path.append("models/uniter")
import image_retrieval


class UniterImageRetriever(Retriever):

    def __init__(self):
        super().__init__(retriever_name="uniter")

    def get_top_k(self, req: RetrievalRequest) -> List[str]:
        opts = self._build_retrieval_opts(req)
        backend_logger.info(opts)

        return image_retrieval.run_retrieval(opts)

    def _build_retrieval_opts(self, req: RetrievalRequest):
        opts = SimpleNamespace()

        opts.query = req.query
        opts.img_feat_db = self._conf.uniter_dir + "/img_db/flickr30k"
        opts.img_ds = self._conf.flickr_dir
        opts.checkpoint = self._conf.uniter_dir + "/pretrained/uniter-base.pt"
        opts.model_config = self._conf.model_config
        opts.meta_file = self._conf.uniter_dir + "/txt_db/itm_flickr30k_test.db/meta.json"

        opts.top_k = req.top_k
        opts.bs = self._conf.batch_size
        opts.num_imgs = self._conf.num_imgs

        opts.n_workers = self._conf.n_data_workers
        opts.fp16 = self._conf.fp16
        opts.pin_mem = self._conf.pin_mem

        return opts
