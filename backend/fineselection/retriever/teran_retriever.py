import os
import sys
from types import SimpleNamespace
from typing import List

from loguru import logger

from backend.fineselection.data import TeranISS
from backend.fineselection.retriever import Retriever
from backend.fineselection.retriever.retriever import RetrieverType

TERAN_PATH = 'models/teran'
sys.path.append(TERAN_PATH)

# noinspection PyUnresolvedReferences
from inference import prepare_model_checkpoint_and_config, load_teran, compute_distances, QueryEncoder


class TeranRetriever(Retriever):
    def __init__(self, retriever_name: str, device: str, model: str, model_config: str):
        super().__init__(retriever_type=RetrieverType.TERAN,
                         retriever_name=retriever_name)

        opts = self.__build_retrieval_opts(device, model, model_config)
        logger.debug(opts)

        # load teran config and checkpoint
        teran_config, model_checkpoint = prepare_model_checkpoint_and_config(opts)
        # load TERAN
        logger.info(f"Loading TERAN model {opts.model}!")
        teran = load_teran(teran_config, model_checkpoint)

        self.teran = teran
        self.model_config = teran_config

    @logger.catch
    def find_top_k_images(self, focus: str, context: str, top_k: int, iss: TeranISS) -> List[str]:
        # compute query embedding
        query_encoder = QueryEncoder(self.model_config, self.teran)
        query_embs, query_lengths = query_encoder.compute_query_embedding(context)

        # get the precomputed image embeddings and lengths tensors to compute the similarity with the query embedding
        img_embs, img_length = iss.get_images()

        # compute the matching scores
        # TODO weight bei focus signal!!!!
        distance_sorted_indices = compute_distances(img_embs,
                                                    query_embs,
                                                    img_length,
                                                    query_lengths,
                                                    self.model_config)

        # get the top-k image names / ids
        top_k_indices = distance_sorted_indices[:top_k]
        top_k_images = iss.get_image_ids(top_k_indices)
        return top_k_images

    @staticmethod
    def __build_retrieval_opts(device: str, model: str, model_config: str):
        opts = SimpleNamespace()
        opts.model = os.path.join(os.getcwd(), TERAN_PATH, model)
        opts.config = os.path.join(os.getcwd(), TERAN_PATH, model_config)
        opts.device = device
        return opts
