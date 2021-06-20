import sys
from types import SimpleNamespace

import numpy as np
from loguru import logger
from typing import List, Tuple, Optional

from backend.fineselection.data import ImageSearchSpace
from backend.fineselection.retriever import Retriever
from backend.fineselection.retriever.retriever import RetrieverType

UNITER_PATH = 'models/uniter'
sys.path.append(UNITER_PATH)


# noinspection PyUnresolvedReferences
# import image_retrieval


class UniterRetriever(Retriever):
    def __init__(self, retriever_name: str,
                 n_gpu: int,
                 uniter_dir: str,
                 model_config: str,
                 num_imgs: int,
                 batch_size: int,
                 n_data_workers: int,
                 fp16: bool,
                 pin_mem: bool):
        super().__init__(retriever_type=RetrieverType.UNITER,
                         retriever_name=retriever_name)
        raise NotImplementedError("UniterRetriever is currently not supported!")

        # self.n_gpu = n_gpu
        # self.uniter_dir = uniter_dir
        # self.model_config = model_config
        # self.num_imgs = num_imgs
        # self.batch_size = batch_size
        # self.n_data_workers = n_data_workers
        # self.fp16 = fp16
        # self.pin_mem = pin_mem

    @logger.catch
    def find_top_k_images(self, focus: str, context: str, top_k: int, iss: ImageSearchSpace, ) -> List[str]:
        raise NotImplementedError("UniterRetriever is currently not supported!")
        # opts = self._build_retrieval_opts(focus, context, top_k)
        # logger.info(opts)
        # return image_retrieval.run_retrieval(opts)

    def _build_retrieval_opts(self, focus: str, context: str, top_k: int):
        # TODO focus!
        opts = SimpleNamespace()

        opts.query = context
        # TODO remove hard coded flicker stuff (do similar to TERAN coco / wicsmmir / f30k)
        opts.img_feat_db = self.uniter_dir + "/img_db/flickr30k"
        opts.checkpoint = self.uniter_dir + "/pretrained/uniter-base.pt"
        opts.meta_file = self.uniter_dir + "/txt_db/itm_flickr30k_test.db/meta.json"
        opts.model_config = self.model_config

        opts.top_k = top_k
        opts.bs = self.batch_size
        opts.num_imgs = self.num_imgs

        opts.n_workers = self.n_data_workers
        opts.fp16 = self.fp16
        opts.pin_mem = self.pin_mem

        return opts

    def find_focus_span_in_context(self, context: str, focus: str) -> Tuple[int, int]:
        raise NotImplementedError("UniterRetriever is currently not supported!")

    def compute_focus_score(self, wra_matrix: np.ndarray, focus_span: Tuple[int, int], strategy: Optional[str]):
        raise NotImplementedError("UniterRetriever is currently not supported!")
