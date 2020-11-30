from abc import abstractmethod

from omegaconf import OmegaConf

from backend.util import SingletonABCMeta
from logger import backend_logger


class ImageServer(object):
    __metaclass__ = SingletonABCMeta

    def __init__(self, image_srv_name: str):
        backend_logger.info(f"Instantiating {image_srv_name} Image Server...")
        self._conf = OmegaConf.load("config.yaml").image_server[image_srv_name]

    @abstractmethod
    def get_img_url(self, img_id: str) -> str:
        """
        :param img_id: the ID of the image to be served
        :return: URL to the image
        """
        pass
