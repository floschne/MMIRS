import os
from abc import abstractmethod
from typing import List

from loguru import logger

from backend.fineselection.data import ImageSearchSpace


class ImageFeaturePool(object):
    """
    This class represents a complete pool of image features for a specific retriever type.
     - contains all image features found in data_root.
     - can be in memory or on disk
    """

    def __init__(self,
                 source_dataset: str,
                 target_retriever_type: str,
                 pre_fetch: bool,
                 data_root: str):
        """
        :param source_dataset: The dataset the image features originate from
        :param target_retriever_type: Since different retriever types (i.e. models) require the features in different
        formats, the target retriever type of ImageFeaturePools has to be specified. Can be TERAN or UNITER as of
        this MMIR version
        :param pre_fetch: if True load the !complete! feature pool into memory
        :param data_root: the root directory where the features are located
        """
        logger.info(f"Instantiating {source_dataset} ImageFeaturePool for {target_retriever_type} Retriever...")

        if not (os.path.lexists(data_root) and os.path.isdir(data_root)):
            logger.error(f"Cannot read {data_root}!")
            raise SystemError(f"Cannot read {data_root}!")

        self.source_dataset = source_dataset
        self.target_retriever_type = target_retriever_type
        self.data_root = data_root
        self.pre_fetch = pre_fetch
        self.data = None

        if pre_fetch:
            self.load_data_into_memory()

    @abstractmethod
    def load_data_into_memory(self):
        raise NotImplementedError()

    @abstractmethod
    def get_image_search_space(self, img_ids: List[str]) -> ImageSearchSpace:
        raise NotImplementedError()
