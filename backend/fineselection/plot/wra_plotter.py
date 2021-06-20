import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List

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

            cls.img_server = PyHttpImageServer()

        return cls.__singleton

    def get_wra_plot_path(self, image_id: str):
        return os.path.join(self.__wra_plots_dst, f"{image_id}_wra.png")

    def plot_wra(self,
                 image_id: str,
                 wra: np.ndarray,
                 context_tokens: List[str]) -> str:
        logger.info(f"Creating WRA Plot for image {image_id}!")

        # setup matplotlib
        # https://stackoverflow.com/a/53816322
        num_regions = wra.shape[0]
        num_tokens = wra.shape[1]
        if num_tokens != len(context_tokens) - 1:  # FIXME LAST TOKEN MISSING
            raise ValueError(f"The number of tokens does not match the WRA with shape={wra.shape}")

        dpi = mpl.rcParams['figure.dpi']
        # Size of the figure in inches to fit the wra
        figsize = num_tokens * self.cell_size_px / float(dpi), num_regions * self.cell_size_px / float(dpi)

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

        # annotated cells with wra value
        # for i in range(num_regions):
        #     for j in range(num_tokens):
        #         text = ax.text(j, i, f"{wra[i, j] : .2f}", ha="center", va="center", color="w")

        # colorbar -> https://stackoverflow.com/a/18195921
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # persist the annotated image
        dst = self.get_wra_plot_path(image_id)
        fig.savefig(dst, bbox_inches='tight')
        plt.clf()
        logger.info(f"Persisted WRA Plot for image {image_id} at {dst}")

        # FIXME get image server dynamically
        self.img_server.register_wra_plot(img_id=image_id,
                                          wra_plot_path=dst)

        return dst
