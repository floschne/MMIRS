from .image_server import ImageServer


class LighttpImgServer(ImageServer):
    def __init__(self):
        super().__init__(image_srv_name="lighttp")
        self._base_url = self.__get_img_server_base_url()

    def __get_img_server_base_url(self) -> str:
        base_url = "https://" if self._conf.https else "http://"
        base_url += self._conf.host
        base_url += ":" + str(self._conf.port)
        base_url += self._conf.endpoint
        return base_url

    def get_img_url(self, img_id: str) -> str:
        return self._base_url + img_id
