import collections
import os
from typing import List, Tuple

import numpy as np


class ROI(object):
    def __init__(self, x0, y0, x1, y1, obj_id, obj_conf, attr_id, attr_conf):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.w = x1 - x0
        self.h = y1 - y0
        self.obj_id = obj_id
        self.obj_conf = obj_conf
        self.attr_id = attr_id
        self.attr_conf = attr_conf

    def __str__(self):
        return (
            "ROI={\n"
            f"\t(X, Y, W, H) -> ({self.x0:0.0f}, {self.y0:0.0f}, {self.w:0.0f}, {self.h:0.0f})\n"
            f"\t(obj_id, obj_conf) -> ({self.obj_id}, {self.obj_conf:0.4f})\n"
            f"\t(attr_id, attr_conf) -> ({self.attr_id}, {self.attr_conf:0.4f})\n"
            "}"
        )

    def __repr__(self):
        return self.__str__()


class ImageMetadata(object):
    def __init__(self, feat_path: str):
        # load npz
        assert os.path.lexists(feat_path), f"Cannot find feature at {feat_path}"
        f = np.load(feat_path, allow_pickle=True)

        info = f['info'].item()
        self.img_id = info['image_id']
        self.image_h = info['image_h']
        self.image_w = info['image_w']

        objects = zip(info['objects_id'], info['objects_conf'])
        attrs = zip(info['attrs_id'], info['attrs_conf'])

        # Sorted list of ROIs according to confidence of detected object categories
        bb = f['bbox']
        self.rois = sorted([
            ROI(x0=bb[i][0],
                y0=bb[i][1],
                x1=bb[i][2],
                y1=bb[i][3],
                obj_id=o[0],
                obj_conf=o[1],
                attr_id=a[0],
                attr_conf=a[1])
            for i, (o, a) in enumerate(zip(objects, attrs))
        ], key=lambda roi: roi.obj_conf)[::-1]
        self.num_rois = len(self.rois)

        # index from object id to roi
        self.obj_to_rois = {}
        for roi in self.rois:
            if roi.obj_id not in self.obj_to_rois:
                self.obj_to_rois[roi.obj_id] = [roi]
            else:
                self.obj_to_rois[roi.obj_id].append(roi)

        # index from attribute id to roi
        self.attr_to_rois = {}
        for roi in self.rois:
            if roi.attr_id not in self.attr_to_rois:
                self.attr_to_rois[roi.attr_id] = [roi]
            else:
                self.attr_to_rois[roi.attr_id].append(roi)

        # object and attribute frequencies
        self.obj_freqs = collections.Counter([roi.obj_id for roi in self.rois])
        self.attr_freqs = collections.Counter([roi.attr_id for roi in self.rois])

    def get_top_k_objects_freqs(self, k: int = 10) -> List[Tuple[int, int]]:
        return self.obj_freqs.most_common(k)

    def get_top_k_objects_rois(self, k: int = 10) -> List[Tuple[int, List[ROI]]]:
        mco = self.obj_freqs.most_common(k)
        return [(obj_id, self.obj_to_rois[obj_id]) for obj_id, _ in mco]

    def get_top_k_rois(self, k: int = 10) -> List[ROI]:
        # TODO define top k better. E.g. we could normalize the object confidences or incorporate
        #  the attribute confidences, too!
        return self.rois[:k]
