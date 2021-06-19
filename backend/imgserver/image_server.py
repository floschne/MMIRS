from abc import abstractmethod

from loguru import logger

from backend.imgserver.image_datasource import ImageDatasource
from config import conf


class ImageServer(object):

    def __init__(self):
        # FIXME very strange bug: can't create the dict via list or dict comprehension
        self.datasources = {}
        datasources_conf = conf.image_server.datasources
        for ds in datasources_conf.keys():
            self.datasources[ds] = ImageDatasource(dataset=str(ds),
                                                   images_root=datasources_conf[ds].images_root,
                                                   image_prefix=datasources_conf[ds].image_prefix,
                                                   image_suffix=datasources_conf[ds].image_suffix)
        logger.info(f"Image Server has datasources: {self.datasources}")

    @abstractmethod
    def get_img_url(self, img_id: str, dataset: str, annotated: bool = False) -> str:
        """
        :param img_id: the ID of the image to be served
        :param dataset: the datasource to load the image from (e.g. teran, uniter, coco_val_14, etc)
        :param annotated: if true, returns the image URL for the image with the MaxFocusRegion annotated (if it exists)
        :return: URL to the image
        """
        raise NotImplementedError()

    @abstractmethod
    def get_image_path(self, img_id: str, dataset: str) -> str:
        """
        :param img_id: the ID of the image
        :param dataset: the datasource to load the image from (e.g. teran, uniter, coco_val_14, etc)
        :return: path to the image
        """
        raise NotImplementedError()

    @abstractmethod
    def register_annotated_image(self,
                                 img_id: str,
                                 dataset: str,
                                 annotated_path: str):
        """
        Registers, i.e., makes the annotated image at the provided path available so that it can be served via URL
        :param img_id: the ID of the image
        :param dataset: the datasource / dataset of the annotated image
        :param annotated_path: the path to the annotated image
        """
        raise NotImplementedError()
