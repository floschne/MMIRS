import urllib.parse as url

from loguru import logger
from typing import List

from backend.imgserver.image_server import ImageServer
from config import conf


# TODO improve architecture wrt ImageServer superclass
class LighttpImgServer(ImageServer):
    __singleton = None
    _conf = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            cls.__singleton = super(LighttpImgServer, cls).__new__(cls)

            logger.info(f"Instantiating Lighttp Image Server...")
            cls._conf = conf.image_server['lighttp']

            cls._base_url = cls.__get_img_server_base_url()

        return cls.__singleton

    @classmethod
    def __get_img_server_base_url(cls) -> str:
        base_url = "https://" if cls._conf.https else "http://"
        base_url += cls._conf.host
        base_url += ":" + str(cls._conf.port)
        base_url += cls._conf.context_path
        return base_url

    def get_img_url(self, img_id: str, dataset: str, annotated: bool = False) -> str:
        if dataset not in self.datasources:
            logger.error(f"Images for Dataset {dataset} not available!")
            return 'NoImagesAvailable'
        return url.urljoin(self._base_url, self.datasources[dataset].get_image_file_name(img_id))

    def get_img_urls(self, img_ids: str, dataset: str, annotated: bool = False) -> List[str]:
        return [self.get_img_url(img_id, dataset, annotated) for img_id in img_ids]

    def get_image_path(self, img_id: str, dataset: str) -> str:
        if dataset not in self.datasources:
            logger.error(f"Images for Dataset {dataset} not available!")
            raise KeyError(f"Image Datasource for dataset {dataset} is not registered!")
        return self.datasources[dataset].get_image_path(img_id)

    def register_annotated_image(self, img_id: str, dataset: str, annotated_path: str):
        raise NotImplemented("This is not (yet) implemented!")
