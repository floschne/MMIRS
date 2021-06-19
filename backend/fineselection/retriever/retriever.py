from abc import abstractmethod

import numpy as np
from enum import Enum, unique
from loguru import logger
from pathlib import Path
from typing import List, Union, Tuple

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
                          return_alignment_matrices: bool = False) -> List[Union[Union[Path, str],
                                                                           Tuple[Union[Path, str], np.ndarray]]]:
        """
        finds the top-k matches from the pool of images in the search space
        :param focus:
        :param context:
        :param top_k:
        :param iss: image search space in which the retriever searchers for the best matching images according to req
        :param return_alignment_matrices: if true the alignment matrices get returned
        :return: Paths or IDs of the image(s) and the WRA matrices if set
        """
        pass
