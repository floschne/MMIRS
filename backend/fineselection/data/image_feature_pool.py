import os
from abc import abstractmethod

from loguru import logger
from typing import List, Optional

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
                 feats_root: str):
        """
        :param source_dataset: The dataset the image features originate from
        :param target_retriever_type: Since different retriever types (i.e. models) require the features in different
        formats, the target retriever type of ImageFeaturePools has to be specified. Can be TERAN or UNITER as of
        this MMIR version
        :param pre_fetch: if True load the !complete! feature pool into memory
        :param feats_root: the root directory where the features are located
        """
        logger.info(f"Instantiating {source_dataset} ImageFeaturePool for {target_retriever_type} Retriever...")

        if not (os.path.lexists(feats_root) and os.path.isdir(feats_root)):
            logger.error(f"Cannot read features at {feats_root}!")
            raise FileNotFoundError(f"Cannot read features at {feats_root}!")

        self.source_dataset = source_dataset
        self.target_retriever_type = target_retriever_type
        self.feats_root = feats_root
        self.pre_fetch = pre_fetch
        self.data = None

    @abstractmethod
    def load_data_into_memory(self):
        raise NotImplementedError()

    @abstractmethod
    def get_image_search_space(self, img_ids: Optional[List[str]]) -> ImageSearchSpace:
        raise NotImplementedError()
