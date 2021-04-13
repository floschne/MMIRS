import urllib.parse as url

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

    def get_img_url(self, img_id: str, datasource: str = 'teran') -> str:
        assert datasource in self.datasources
        return url.urljoin(self._base_url, self.datasources[datasource]) + img_id
