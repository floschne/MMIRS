import os
from abc import abstractmethod

from loguru import logger

from config import conf


class ImageDatasource(object):
    def __init__(self, dataset: str, images_root: str, image_prefix: str, image_suffix: str):
        self.dataset = dataset
        self.images_root = images_root
        self.image_prefix = image_prefix
        self.image_suffix = image_suffix

        if not os.path.lexists(images_root) or not os.path.isdir(images_root):
            logger.error(f"Cannot read Image Datasource at {images_root}!")
            raise FileNotFoundError(f"Cannot read Image Datasource at {images_root}!")

    def get_image_file_name(self, img_id: str):
        return self.image_prefix + img_id + self.image_suffix

    def __repr__(self):
        return (f"ImageDatasource(dataset={self.dataset},\n"
                f"\timages_root={self.images_root},"
                f"\timage_prefix={self.image_prefix},"
                f"\timage_suffix={self.image_suffix})")


class ImageServer(object):
    def __init__(self, image_srv_name: str):
        logger.info(f"Instantiating {image_srv_name} Image Server...")
        self._conf = conf.image_server[image_srv_name]
        # FIXME very strange bug: can't create the dict via list or dict comprehension
        self.datasources = {}
        for ds in self._conf.datasources.keys():
            self.datasources[ds] = ImageDatasource(dataset=str(ds),
                                                   images_root=self._conf.datasources[ds].images_root,
                                                   image_prefix=self._conf.datasources[ds].image_prefix,
                                                   image_suffix=self._conf.datasources[ds].image_suffix)
        logger.info(f"{image_srv_name} Image Server has datasources: {self.datasources}")

    @abstractmethod
    def get_img_url(self, img_id: str, datasource: str) -> str:
        """
        :param img_id: the ID of the image to be served
        :param datasource: the datasource to load the image from (e.g. teran, uniter, coco_val_14, etc)
        :return: URL to the image
        """
        pass
