from abc import abstractmethod
from typing import Any

from loguru import logger


class ImageSearchSpace(object):
    def __init__(self, target_retriever_type: str, images: Any):
        logger.debug(f"Instantiating ImageSearchSpace for {target_retriever_type} Retriever with {len(images)} images...")
        self.target_retriever_type = target_retriever_type
        self.images = images

    @abstractmethod
    def get_images(self):
        """
        This method returns the image (feature) data in the format necessary for the respective target retrieval model
        """
        pass
