import os
import sys
from types import SimpleNamespace

import numpy as np
from loguru import logger
from typing import List, Union, Tuple

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
    def find_top_k_images(self,
                          focus: str,
                          context: str,
                          top_k: int,
                          iss: TeranISS,
                          return_alignment_matrices: bool = False) -> Union[List[str], Tuple[List[str], np.ndarray]]:
        # compute query embedding
        query_encoder = QueryEncoder(self.model_config, self.teran)
        query_embs, query_lengths = query_encoder.compute_query_embedding(context)

        # get the precomputed image embeddings and lengths tensors to compute the similarity with the query embedding
        img_embs, img_length = iss.get_images()

        # compute the matching scores
        wra_matrices: np.ndarray  # type hint
        distances: np.ndarray  # type hint
        distances, wra_matrices = compute_distances(img_embs,
                                                    query_embs,
                                                    img_length,
                                                    query_lengths,
                                                    self.model_config,
                                                    return_wra_matrices=True)

        # get the img indices descended sorted by the distance matrix
        distance_sorted_indices = np.argsort(distances)[::-1]

        # get the top-k image names / ids
        top_k_indices = distance_sorted_indices[:top_k]
        top_k_images = iss.get_image_ids(top_k_indices)

        # TODO weight by focus signal!!!!

        if return_alignment_matrices:
            # get the wra matrices of the top-k
            return top_k_images, wra_matrices[top_k_indices, ...]
        return top_k_images

    @staticmethod
    def __build_retrieval_opts(device: str, model: str, model_config: str):
        opts = SimpleNamespace()
        opts.model = os.path.join(os.getcwd(), TERAN_PATH, model)
        opts.config = os.path.join(os.getcwd(), TERAN_PATH, model_config)
        opts.device = device
        return opts
