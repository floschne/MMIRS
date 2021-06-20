from abc import abstractmethod

import numpy as np
from enum import Enum, unique
from loguru import logger
from typing import List, Union, Tuple, Optional, Dict

from backend.fineselection.data import ImageSearchSpace
from config import conf


@unique
class RetrieverType(str, Enum):
    UNITER = 'uniter'
    TERAN = 'teran'


class Retriever(object):
    def __init__(self, retriever_type: RetrieverType, retriever_name: str):
        logger.info(f"Instantiating {retriever_name} {retriever_type.upper()} Retriever...")
        self.retriever_type = retriever_type
        self.retriever_name = retriever_name

        self._conf = conf.fine_selection.retrievers[retriever_name]
        assert self._conf is not None, f"Cannot find config for Retriever with name {retriever_name}!"

    @abstractmethod
    def find_top_k_images(self,
                          focus: str,
                          context: str,
                          top_k: int,
                          iss: ImageSearchSpace,
                          focus_weight: float = 0.5,
                          return_scores: bool = False,
                          return_wra_matrices: bool = False,
                          return_separated_ranks: bool = False) -> Dict[str, Dict[str, Union[List[str], np.ndarray]]]:
        """
        finds the top-k matches from the pool of images in the search space

        :param focus: focus text
        :param context: context text
        :param top_k: how many images to return
        :param iss: image search space in which the retriever searchers for the best matching images according to req
        :param focus_weight: the weight of the focus when ranking the images (w * focus + (1-2) * context)
        :param return_scores: if true the scores of the top-k images are returned
        :param return_wra_matrices: if true the wra matrices of the top-k images are returned
        :param return_separated_ranks: if true, in addition to the combined ranked top-k image,
               also the context ranked and focus ranked top-k images are returned
        :return: A dict with the following structure: {'top_k': {'combined|focus|context': List[str]},
                                                       'scores' {'combined|focus|context': np.ndarray},
                                                       'wra' {'combined|focus|context': np.ndarray}}
        """
        pass

    @abstractmethod
    def find_focus_span_in_context(self, context: str, focus: str) -> Tuple[int, int]:
        """
        Finds the span of the focus in the context on a token level.
        :param focus: focus text
        :param context: context text
        :return: the span of the focus tokens in the context tokens. (start, end)
        """
        pass

    @abstractmethod
    def tokenize(self, text: str, remove_sep_cls: bool) -> Tuple[List[str], List[int]]:
        """
        Tokenizes and arbitrary text and return a tuple of the tokens and the token ids.
        :param text: arbitrary text to tokenize
        :param remove_sep_cls: if true, the CLS and SEP tokens are removed. (assuming a BERT Tokenizer is utilized)
        :return: tuple of the tokens and the token ids
        """
        pass

    @abstractmethod
    def compute_focus_score(self, wra_matrix: np.ndarray, focus_span: Tuple[int, int], strategy: Optional[str]):
        """
        computes the focus score by pooling the WRA matrix
        :param wra_matrix: wra matrix of the context tokens and the image region with shape: (num_roi, num_ctx_token)
        :param focus_span: span of the focus in the context on a token level
        :param strategy: depends on the implementation
        :return: a score of how strong the signal of the focus tokens in this wra matrix is
        """

    @abstractmethod
    def find_max_focus_region_index(self, focus_span: Tuple[int, int], wra_matrix: np.ndarray) -> int:
        pass
