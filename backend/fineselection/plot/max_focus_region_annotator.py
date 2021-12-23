import gc
import os
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from backend.fineselection.plot.bboxes_datasource import BBoxesDatasource
from backend.imgserver.py_http_image_server import PyHttpImageServer
from config import conf


class MaxFocusRegionAnnotator(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info(f"Instantiating MaxFocusRegionAnnotator...")
            cls.__singleton = super(MaxFocusRegionAnnotator, cls).__new__(cls)

            cls.__conf = conf.fine_selection.max_focus_annotator

            assert cls.__conf is not None, f"Cannot find config for MaxFocusAnnotator!"

            cls.__datasources = {}
            datasources_conf = cls.__conf.datasources
            for ds in datasources_conf.keys():
                cls.__datasources[ds] = BBoxesDatasource(dataset=str(ds),
                                                         bboxes_root=datasources_conf[ds].bbox_root,
                                                         fn_prefix=datasources_conf[ds].fn_prefix,
                                                         fn_suffix=datasources_conf[ds].fn_suffix)
            logger.debug(f"MaxFocusRegionAnnotator has BBoxes datasources: {cls.__datasources}")

            cls.__annotated_images_dst = cls.__conf.annotated_images_dst
            if not (os.path.lexists(cls.__annotated_images_dst) and os.path.isdir(cls.__annotated_images_dst)):
                try:
                    os.mkdir(cls.__annotated_images_dst)
                except:
                    raise FileNotFoundError(f"Cannot read annotated image destination at {cls.__annotated_images_dst}!")

            cls.img_server = PyHttpImageServer()

        return cls.__singleton

    def get_max_focus_annotated_image_path(self, image_id: str):
        return os.path.join(self.__annotated_images_dst, f"{image_id}_annotated.png")

    @logger.catch(reraise=True)
    def annotate_max_focus_regions(self,
                                   pool: ProcessPoolExecutor,
                                   image_ids: List[str],
                                   max_focus_region_indices: List[int],
                                   dataset: str,
                                   focus_text: str) -> List[Future]:
        if dataset not in self.__datasources:
            logger.error(f"BBoxes for Dataset {dataset} are not available!")
            raise KeyError(f"BBoxes for {dataset} are not available!")

        futures = []
        for iid, mfri in zip(image_ids, max_focus_region_indices):
            # submit all tasks and keep future
            futures.append(pool.submit(annotate_max_focus_region,
                                       image_id=iid,
                                       bbox_fn=self.__datasources[dataset].get_bbox_path(iid),
                                       max_focus_region_idx=mfri,
                                       dst=self.get_max_focus_annotated_image_path(iid),
                                       img_path=self.img_server.get_image_path(iid, dataset),
                                       dataset=dataset,
                                       focus_text=focus_text
                                       )
                           )
        return futures


def load_bboxes(bbox_fn: str) -> np.ndarray:
    feat = np.load(bbox_fn, allow_pickle=True)
    # noinspection PyUnresolvedReferences
    if 'bbox' not in feat.files:
        raise ValueError("Cannot find bbox in npz archive!")
    return feat['bbox']


@logger.catch(reraise=True)
def annotate_max_focus_region(image_id: str,
                              bbox_fn: str,
                              max_focus_region_idx: int,
                              dst: str,
                              img_path: str,
                              dataset: str,
                              focus_text: str) -> Tuple[str, str, str]:
    logger.debug(f"Annotating maximum focus region for image {image_id} of dataset {dataset}")

    # load bboxes
    bboxes = load_bboxes(bbox_fn)
    # find the bbox with maximum focus signal in the WRA matrix
    foc_bb = bboxes[max_focus_region_idx]
    # free bboxes memory
    del bboxes


    # setup matplotlib so that the image gets drawn on the axes canvas in full size
    # https://stackoverflow.com/a/53816322
    dpi = mpl.rcParams['figure.dpi']
    # FIXME Flo: better image io to be safe when sth is wrong with the image...
    im_data = plt.imread(img_path)
    height, width = im_data.shape[0], im_data.shape[1]
    # Size of the figure in inches to fit the image
    figsize = width / float(dpi), height / float(dpi)
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')
    # Display the image on the axes canvas
    ax.imshow(im_data)

    # get the position of the bbox
    x0, y0, x1, y1 = foc_bb[0], foc_bb[1], foc_bb[2], foc_bb[3]
    w, h = x1 - x0, y1 - y0

    # draw the bbox border rectangle
    # TODO color, etc in config
    bbox_border_lw = 2
    ax.add_patch(plt.Rectangle((x0, y0), w, h,
                               fill=False,
                               edgecolor='red',
                               linewidth=bbox_border_lw,
                               alpha=0.75))

    # draw the focus text in a rect on the upper right outside of the bbox
    # TODO color, etc in config
    fs = 15
    text_border_lw = 1
    # check if the text is outside of the image and re-position if so
    ty0 = y0 - bbox_border_lw - fs // 2
    if ty0 < fs // 2:  # outside on top
        ty0 = y1 + fs + bbox_border_lw + text_border_lw  # reposition below bbox
    if ty0 + fs + text_border_lw > height:  # outside on bottom
        ty0 = y1 - bbox_border_lw - fs // 2  # reposition inside the bbox at the bottom

    ax.text(x0 + text_border_lw + bbox_border_lw,
            ty0,
            focus_text,
            bbox=dict(facecolor='blue', alpha=0.5, linewidth=text_border_lw),
            fontsize=fs,
            color='white')

    fig.savefig(dst)
    plt.clf()
    plt.cla()
    plt.close("all")
    gc.collect()
    logger.debug(f"Persisted MaxFocus-annotated image at {dst}")

    return image_id, dst, 'anno'
