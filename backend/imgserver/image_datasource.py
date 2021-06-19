import os
import glob

from loguru import logger


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

    def get_image_path(self, img_id: str) -> str:
        img_p = os.path.join(self.images_root, self.get_image_file_name(img_id))
        if not (os.path.lexists(img_p) or os.path.isfile(img_p)):
            logger.error(f"Cannot read image at {img_p}!")
            raise FileNotFoundError(f"Cannot read image at {img_p}!")
        return img_p

    def get_number_of_images(self):
        return len(glob.glob(self.get_image_file_name("*")))

    def __repr__(self):
        return (f"ImageDatasource(dataset={self.dataset},\n"
                f"\timages_root={self.images_root},"
                f"\timage_prefix={self.image_prefix},"
                f"\timage_suffix={self.image_suffix}) with {self.get_number_of_images()} images!")
