import collections
import os
from typing import List, Tuple, Dict

import numpy as np

from cluster.visual.vocab import Vocab


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


class TermData(object):
    def __init__(self, term: str, count: int = 0, area: float = 0., conf: float = 0., weight: float = 0.,
                 wtf: float = 0.):
        self.term = term
        self.count = count
        self.area = area
        self.conf = conf
        self.weight = weight
        self.wtf = wtf

    def __str__(self):
        return (
            "TermData={\n"
            f"\t term -> {self.term}\n"
            f"\t wtf -> {self.wtf}\n"
            f"\t count -> {self.count}\n"
            f"\t weight -> {self.weight}\n"
            f"\t area -> {self.area}\n"
            f"\t conf -> {self.conf}\n"
            "}"
        )

    def __repr__(self):
        return self.__str__()


class ImageMetadata(object):
    def __init__(self, feat_path: str, vocab: Vocab, octh: float = .2, acth: float = .15, alpha: float = .95):
        """
        :param feat_path: path to the feature.npz file
        :param vocab: object and attribute vocabulary. to lookup object and attribute names
        :param octh: object confidence threshold. objects with conf below the threshold get ignored
        :param acth: attribute confidence threshold. attributes with conf below the threshold get ignored
        :param alpha: weight for weighted term-frequency. weight = alpha * conf + (1-alpha) * area
        """
        # load npz
        assert os.path.lexists(feat_path)
        f = np.load(feat_path, allow_pickle=True)

        info = f['info'].item()
        self.img_id = info['image_id']

        self.img_w = info['image_w']
        self.img_h = info['image_h']

        bb = f['bbox']

        objects = zip(info['objects_id'], info['objects_conf'])
        attrs = zip(info['attrs_id'], info['attrs_conf'])

        # Sorted list of ROIs according to confidence of detected object categories
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
        ], key=lambda r: r.obj_conf)[::-1]
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

        self.vocab = vocab
        self.octh = octh
        self.acth = acth
        self.alpha = alpha

        self.term_data = self._collect_term_data()
        self.terms = set(self.term_data.keys())

    def get_most_common_objects(self, n: int = 10) -> List[Tuple[int, int]]:
        return self.obj_freqs.most_common(n)

    def get_most_common_object_rois(self, n: int = 10) -> List[Tuple[int, List[ROI]]]:
        mco = self.obj_freqs.most_common(n)
        return [(obj_id, self.obj_to_rois[obj_id]) for obj_id, _ in mco]

    def get_top_k_rois(self, k: int = None) -> List[ROI]:
        # TODO define top k better. E.g. we could normalize the object confidences or incorporate
        #  the attribute confidences, too!
        if k is None:
            k = self.num_rois
        return self.rois[:k]

    def _collect_term_data(self) -> Dict[str, TermData]:
        term_data = {}
        for r in self.rois:
            # object confidence threshold
            if r.obj_conf >= self.octh:
                # get the name of the object
                obj = self.vocab.get_obj_name(r.obj_id)
                # whitespace tokenize if necessary to get the terms
                for term in obj.split(' '):
                    if term not in term_data:
                        term_data[term] = TermData(term)

                    # increase term counter
                    term_data[term].count += 1

                    # accumulate term conf
                    term_data[term].conf += r.obj_conf

                    # accumulate term area
                    term_data[term].area += r.w * r.h

                # attribute confidence threshold (obj conf is dominant!)
                if r.attr_conf >= self.acth:
                    # get the name of the attribute
                    attr = self.vocab.get_attr_name(r.attr_id)
                    # whitespace tokenize if necessary
                    for term in attr.split(' '):
                        if term not in term_data:
                            term_data[term] = TermData(term)

                        # increase term counter
                        term_data[term].count += 1

                        # accumulate term conf
                        term_data[term].conf += r.attr_conf

                        # accumulate term area
                        term_data[term].area += r.w * r.h

        for td in term_data.values():
            # normalize conf by count of term
            td.conf /= td.count
            # normalize area by image area
            td.area /= (self.img_w * self.img_h)
            # compute weight
            td.weight = self.alpha * td.conf + (1 - self.alpha) * td.area
            # compute weighted term frequency (wtf)
            td.wtf = (td.count * td.weight) / len(term_data)

        # sort by wtf
        return {k: v for k, v in sorted(term_data.items(), key=lambda item: item[1].wtf)[::-1]}
