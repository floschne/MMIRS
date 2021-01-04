from abc import abstractmethod

from omegaconf import OmegaConf

from backend.util import SingletonABCMeta
from logger import backend_logger


class ImageServer(object):
    __metaclass__ = SingletonABCMeta

    def __init__(self, image_srv_name: str):
        backend_logger.info(f"Instantiating {image_srv_name} Image Server...")
        self._conf = OmegaConf.load("config.yaml").image_server[image_srv_name]
        # FIXME very strange bug: can't create the dict via list or dict comprehension
        self.datasources = {}
        for i in range(len(self._conf.datasources)):
            self.datasources.update(self._conf.datasources[i])
        backend_logger.info(f"{image_srv_name} Image Server has datasources: {self.datasources}")



    @abstractmethod
    def get_img_url(self, img_id: str, datasource: str) -> str:
        """
        :param img_id: the ID of the image to be served
        :param datasource: the datasource to load the image from (e.g. teran, uniter, coco_val_14, etc)
        :return: URL to the image
        """
        pass
