import urllib.parse as url
from typing import List

from loguru import logger

from .image_server import ImageServer


class LighttpImgServer(ImageServer):
    def __init__(self):
        super().__init__(image_srv_name="lighttp")
        self._base_url = self.__get_img_server_base_url()

    def __get_img_server_base_url(self) -> str:
        base_url = "https://" if self._conf.https else "http://"
        base_url += self._conf.host
        base_url += ":" + str(self._conf.port)
        base_url += self._conf.context_path
        return base_url

    def get_img_url(self, img_id: str, dataset: str) -> str:
        if dataset not in self.datasources:
            logger.error(f"Images for Dataset {dataset} not available!")
            return 'NoImagesAvailable'
        return url.urljoin(self._base_url, self.datasources[dataset].get_image_file_name(img_id))

    def get_img_urls(self, img_ids: str, dataset: str) -> List[str]:
        return [self.get_img_url(img_id, dataset) for img_id in img_ids]
