import os
import sys
from types import SimpleNamespace
from typing import List

from loguru import logger

from api.model import RetrievalRequest
from backend.retriever import Retriever
from logger import backend_logger

TERAN_PATH = 'models/teran'
sys.path.append(TERAN_PATH)

# noinspection PyUnresolvedReferences
from inference import prepare_model_checkpoint_and_config, load_teran, load_precomputed_image_embeddings, \
    compute_distances, get_image_names, QueryEncoder


class TeranImageRetriever(Retriever):

    def __init__(self):
        super().__init__(retriever_name="teran")
        opts = self._build_retrieval_opts()
        backend_logger.info(opts)

        # load teran config and checkpoint
        teran_config, model_checkpoint = prepare_model_checkpoint_and_config(opts)
        # load TERAN
        teran = load_teran(teran_config, model_checkpoint)
        self.teran = teran
        self.model_config = teran_config

        # load pre-computed image embeddings
        img_embs, img_lengths, dataset = load_precomputed_image_embeddings(teran_config, opts.num_data_workers)
        self.img_embs = img_embs
        self.img_lengths = img_lengths
        self.dataset = dataset

    @logger.catch
    def get_top_k(self, req: RetrievalRequest) -> List[str]:
        # compute query embedding
        query_encoder = QueryEncoder(self.model_config, self.teran)
        query_embs, query_lengths = query_encoder.compute_query_embedding(req.query)

        # compute the matching scores
        distance_sorted_indices = compute_distances(self.img_embs,
                                                    query_embs,
                                                    self.img_lengths,
                                                    query_lengths,
                                                    self.model_config)
        top_k_indices = distance_sorted_indices[:req.top_k]

        # get the image names
        top_k_images = get_image_names(top_k_indices, self.dataset)
        return top_k_images

    def _build_retrieval_opts(self):
        opts = SimpleNamespace()

        opts.config = os.path.join(os.getcwd(), TERAN_PATH, self._conf.model_config)
        opts.model = os.path.join(os.getcwd(), TERAN_PATH, self._conf.model)
        opts.num_data_workers = self._conf.num_data_workers
        opts.device = self._conf.device

        opts.dataset = self._conf.batch_size
        opts.num_images = self._conf.num_imgs
        return opts
