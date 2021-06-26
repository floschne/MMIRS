import glob
import os.path
import urllib.parse as url
from concurrent.futures import ThreadPoolExecutor

import http.server
import socketserver
from loguru import logger
from typing import List

from backend.imgserver.image_server import ImageServer
# TODO improve architecture wrt ImageServer superclass -> datasource instantiation
from config import conf


def http_server_task(http_server_root_dir: str, port: int, host: str):
    # https://stackoverflow.com/a/52531444
    def handler_from(directory):
        def _init(self, *args, **kwargs):
            return http.server.SimpleHTTPRequestHandler.__init__(self,
                                                                 *args,
                                                                 directory=self.directory,
                                                                 **kwargs)

        return type(f'HandlerFrom<{directory}>',
                    (http.server.SimpleHTTPRequestHandler,),
                    {'__init__': _init, 'directory': directory})

    # TODO SSL support: https://gist.github.com/dergachev/7028596#gistcomment-3708957
    with socketserver.TCPServer((host, port), handler_from(http_server_root_dir)) as httpd:
        logger.info(f"Serving {http_server_root_dir} at {host}:{port} ...")
        httpd.serve_forever()


class PyHttpImageServer(ImageServer):

    __singleton = None
    _conf = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            cls.__singleton = super(PyHttpImageServer, cls).__new__(cls)

            logger.info(f"Instantiating PyHttp Image Server...")
            cls._conf = conf.image_server['pyhttp']
            cls._base_url = cls.__get_img_server_base_url()

            cls.__link_root_dir = cls._conf.link_root_dir
            if not os.path.lexists(cls.__link_root_dir):
                os.makedirs(cls.__link_root_dir, exist_ok=True)

            if cls._conf.flush_link_dir:
                logger.info(f"Flushing link root directory {cls.__link_root_dir}!")
                for f in glob.glob(os.path.join(cls.__link_root_dir, "*")):
                    os.remove(f)

            cls.annotated_images_filename_cache = dict()
            cls.wra_plot_filename_cache = dict()

            # setup http server thread
            logger.info("Starting PyHttpImageServer Thread...")
            cls.http_server_thread = ThreadPoolExecutor(max_workers=1)
            cls.http_server_thread.submit(http_server_task,
                                          http_server_root_dir=cls.__link_root_dir,
                                          port=cls._conf.port,
                                          host=cls._conf.host)

        return cls.__singleton

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def shutdown(self):
        logger.info(f'Shutting down PyHttpImageServer!')
        # clear remaining futures
        # https://gist.github.com/clchiou/f2608cbe54403edb0b13
        self.http_server_thread._threads.clear()
        concurrent.futures.thread._threads_queues.clear()
        self.http_server_thread.shutdown(wait=False)

    @classmethod
    def __get_img_server_base_url(cls) -> str:
        base_url = "https://" if cls._conf.https else "http://"
        base_url += cls._conf.host
        base_url += ":" + str(cls._conf.port)
        base_url += cls._conf.context_path
        return base_url

    def __hard_link_into_link_root_dir(self, image_path: str, force: bool = False) -> str:
        link_dst = str(os.path.join(self.__link_root_dir, os.path.basename(image_path)))
        if not os.path.lexists(image_path):
            logger.warning(f"Cannot read {image_path}!")
        elif not os.path.lexists(link_dst):
            os.link(image_path, link_dst)
            logger.debug(f"Hard-linked {image_path} to {link_dst} !")
        elif os.path.lexists(link_dst) and os.path.isfile(link_dst) and force:
            logger.debug(f"Force overwrite {link_dst} with {image_path}!")
            os.remove(link_dst)
            os.link(image_path, link_dst)
            logger.debug(f"Hard-linked {image_path} to {link_dst} !")
        else:
            logger.warning(f"Cannot hard-link to {link_dst}! File already exists!")

        return link_dst

    def get_img_url(self, img_id: str, dataset: str, annotated: bool = False) -> str:
        # get the path of the image
        img_p = self.get_image_path(img_id, dataset)
        if not annotated:
            # hard link into serving directory
            link = self.__hard_link_into_link_root_dir(img_p)  # image is now available via url
            return url.urljoin(self._base_url, os.path.basename(link))
        else:
            # the annotated images get registered in the MaxFocusRegionAnnotator, so we do not have to link them again
            return url.urljoin(self._base_url, self.annotated_images_filename_cache[(img_id, dataset)])

    def get_wra_url(self, img_id: str) -> str:
        # the wra plots get registered in the WRAPlotter, so we do not have to link them again
        return url.urljoin(self._base_url, self.wra_plot_filename_cache[img_id])

    def get_img_urls(self, img_ids: List[str], dataset: str, annotated: bool = False) -> List[str]:
        return [self.get_img_url(img_id, dataset, annotated) for img_id in img_ids]

    def get_wra_urls(self, img_ids: List[str]) -> List[str]:
        return [self.get_wra_url(img_id) for img_id in img_ids]

    def get_image_path(self, img_id: str, dataset: str) -> str:
        if dataset not in self.datasources:
            logger.error(f"Images for Dataset {dataset} are not available!")
            raise KeyError(f"Images for Dataset {dataset} are not available!")
        return self.datasources[dataset].get_image_path(img_id)

    def register_annotated_image(self, img_id: str, dataset: str, annotated_image_path: str):
        self.__hard_link_into_link_root_dir(annotated_image_path, force=True)
        self.annotated_images_filename_cache[(img_id, dataset)] = os.path.basename(annotated_image_path)
        logger.debug(f"Registered annotated image for Image {img_id} of dataset {dataset}!")

    def register_wra_plot(self, img_id: str, wra_plot_path: str):
        self.__hard_link_into_link_root_dir(wra_plot_path, force=True)
        self.wra_plot_filename_cache[img_id] = os.path.basename(wra_plot_path)
        logger.debug(f"Registered WRA Plot for Image {img_id}!")