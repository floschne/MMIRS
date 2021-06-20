import os
from concurrent.futures import as_completed, ProcessPoolExecutor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from loguru import logger
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple

from backend.imgserver.py_http_image_server import PyHttpImageServer
from config import conf


class WRAPlotter(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info(f"Instantiating WRAPlotter...")
            cls.__singleton = super(WRAPlotter, cls).__new__(cls)

            cls.__conf = conf.fine_selection.wra_plotter
            assert cls.__conf is not None, f"Cannot find config for WRAPlotter!"
            cls.__wra_plots_dst = cls.__conf.wra_plots_dst
            if not (os.path.lexists(cls.__wra_plots_dst) and os.path.isdir(cls.__wra_plots_dst)):
                try:
                    os.mkdir(cls.__wra_plots_dst)
                except:
                    raise FileNotFoundError(f"Cannot read WRA Plot destination at {cls.__wra_plots_dst}!")

            cls.cell_size_px = cls.__conf.cell_size_px

            # init plotter pool
            cls.pool = ProcessPoolExecutor(max_workers=cls.__conf.max_workers)

            cls.img_server = PyHttpImageServer()

        return cls.__singleton

    def get_wra_plot_path(self, image_id: str):
        return os.path.join(self.__wra_plots_dst, f"{image_id}_wra.png")

    def generate_wra_plots(self,
                           image_ids: List[str],
                           wra_matrices: np.ndarray,
                           max_focus_region_indices: List[int],
                           focus_span: Tuple[int, int],
                           context_tokens: List[str], ):

        with tqdm.tqdm(desc='Generating WRA Plots', total=len(image_ids)) as pbar:
            futures = []
            for iid, wra, mfri in zip(image_ids, wra_matrices, max_focus_region_indices):
                # submit all plotting tasks and keep future
                futures.append(self.pool.submit(plot_wra,
                                                image_id=iid,
                                                wra=wra,
                                                context_tokens=context_tokens,
                                                max_focus_region_idx=mfri,
                                                focus_span=focus_span,
                                                cell_size_px=self.cell_size_px,
                                                dst=self.get_wra_plot_path(iid)
                                                )
                               )

            for future in as_completed(futures):
                iid, dst = future.result()
                # register wra plot at image server
                self.img_server.register_wra_plot(img_id=iid, wra_plot_path=dst)
                pbar.update(1)


def plot_wra(
        image_id: str,
        wra: np.ndarray,
        context_tokens: List[str],
        max_focus_region_idx: int,
        focus_span: Tuple[int, int],
        cell_size_px: int,
        dst: str) -> Tuple[str, str]:
    logger.info(f"Creating WRA Plot for image {image_id}!")

    # setup matplotlib
    # https://stackoverflow.com/a/53816322
    num_regions = wra.shape[0]
    num_tokens = wra.shape[1]
    if num_tokens != len(context_tokens) - 1:  # FIXME LAST TOKEN MISSING
        raise ValueError(f"The number of tokens does not match the WRA with shape={wra.shape}")

    dpi = mpl.rcParams['figure.dpi']
    # Size of the figure in inches to fit the wra
    figsize = num_tokens * cell_size_px / float(dpi), num_regions * cell_size_px / float(dpi)

    fig, ax = plt.subplots(1, figsize=figsize)

    # region IDs on y-Axis
    ax.set_ylabel("Region IDs")
    ax.set_yticks(np.arange(num_regions))
    # token text on x-axis
    ax.set_xticks(np.arange(num_tokens))
    ax.set_xticklabels(context_tokens[:-1])  # FIXME LAST TOKEN MISSING

    # Rotate the tokens and set their horizontal alignment (ha)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    im = ax.imshow(wra, interpolation='nearest')

    # annotated the max focus region
    y = max_focus_region_idx - 0.5
    x = focus_span[0] - 0.5
    w = focus_span[1] - focus_span[0] + 1
    h = 1
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', lw=2, clip_on=False))

    # colorbar -> https://stackoverflow.com/a/18195921
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # persist the annotated image
    fig.savefig(dst, bbox_inches='tight')
    plt.clf()
    logger.info(f"Persisted WRA Plot for image {image_id} at {dst}")

    return image_id, dst
