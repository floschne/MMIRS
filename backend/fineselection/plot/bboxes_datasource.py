import glob
import os

from loguru import logger


# TODO make superclass or interface (see ImageDatasource)
class BBoxesDatasource(object):
    def __init__(self, dataset: str, bboxes_root: str, fn_prefix: str, fn_suffix: str):
        self.dataset = dataset
        self.bboxes_root = bboxes_root
        self.fn_prefix = fn_prefix if fn_prefix is not None else ''
        self.fn_suffix = fn_suffix if fn_suffix is not None else ''

        if not os.path.lexists(bboxes_root) or not os.path.isdir(bboxes_root):
            logger.error(f"Cannot read BBoxesDatasource at {bboxes_root}!")
            raise FileNotFoundError(f"Cannot read BBoxesDatasource at {bboxes_root}!")

    def get_bbox_file_name(self, img_id: str):
        return self.fn_prefix + img_id + self.fn_suffix

    def get_bbox_path(self, img_id: str) -> str:
        img_p = os.path.join(self.bboxes_root, self.get_bbox_file_name(img_id))
        if not (os.path.lexists(img_p) or os.path.isfile(img_p)):
            logger.error(f"Cannot read bbox at {img_p}!")
            raise FileNotFoundError(f"Cannot read bbox at {img_p}!")
        return img_p

    def get_number_of_bboxes(self):
        return len(glob.glob(os.path.join(self.bboxes_root, self.get_bbox_file_name("*"))))

    def __repr__(self):
        return (f"BBoxesDatasource(dataset={self.dataset},\n"
                f"\tbboxes_root={self.bboxes_root},"
                f"\tfn_prefix={self.fn_prefix},"
                f"\tfn_suffix={self.fn_suffix}) with {self.get_number_of_bboxes()} BBoxes!")
