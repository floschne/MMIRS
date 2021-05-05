import sys
from typing import Tuple, List, Union

import numpy as np
import torch

from backend.fineselection.data import ImageSearchSpace

TERAN_PATH = 'models/teran'
sys.path.append(TERAN_PATH)
# noinspection PyUnresolvedReferences
from data import PreComputedImageEmbeddingsData


class TeranISS(ImageSearchSpace):
    def __init__(self, images: PreComputedImageEmbeddingsData):
        super().__init__(target_retriever_type='TERAN', images=images)

        self.cached_img_embs = None
        self.cached_img_lengths = None

    def get_images(self) -> Tuple[torch.Tensor, int]:
        if self.cached_img_embs is None:
            # get the img embeddings and convert them to Tensors
            np_img_embs = np.array(list(self.images.img_embs.values()))
            img_embs = torch.Tensor(np_img_embs)
            # TODO support image embeddings with various lengths (now this works because we always use fixed num of 36 rois)
            img_lengths = len(np_img_embs[0])
            self.cached_img_embs = img_embs
            self.cached_img_lengths = img_lengths

        return self.cached_img_embs, self.cached_img_lengths

    def get_image_id(self, idx: int) -> Union[str, int]:
        return self.images.image_ids[idx]

    def get_image_ids(self, indices: List[int]) -> List[Union[str, int]]:
        return [self.get_image_id(idx) for idx in indices]

    def __del__(self):
        del self.images
        del self.cached_img_embs

    def __len__(self):
        return len(self.images)
