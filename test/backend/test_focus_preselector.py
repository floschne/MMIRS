import time

import pytest
from loguru import logger

from backend.preselection import FocusPreselector


@pytest.fixture
def fps():
    return FocusPreselector()


def test_top_k_similar_terms_out_of_vocab(fps):
    f = "gyroscope"
    assert f not in fps.vocab

    similar = fps.find_top_k_similar_focus_terms(f)

    assert type(similar) == dict
    assert len(similar) <= fps.max_similar
    assert len(similar) == fps.top_k_similar
    assert max(similar.values()) <= 1.0

    # test ordering
    sims = list(similar.values())
    for i in range(len(sims)):
        if i == 0:
            continue
        assert sims[i] <= sims[i - 1]


def test_top_k_similar_terms_single_focus(fps):
    similar = fps.find_top_k_similar_focus_terms("building")
    assert type(similar) == dict
    assert len(similar) <= fps.max_similar
    assert max(similar.values()) <= 1.0

    # test ordering
    sims = list(similar.values())
    for i in range(len(sims)):
        if i == 0:
            continue
        assert sims[i] <= sims[i - 1]


def test_top_k_similar_terms_multiple_focus(fps):
    similar = fps.find_top_k_similar_focus_terms("green building")
    assert type(similar) == dict
    assert len(similar) <= fps.max_similar
    assert max(similar.values()) <= 1.0

    # test ordering
    sims = list(similar.values())
    for i in range(len(sims)):
        if i == 0:
            continue
        assert sims[i] <= sims[i - 1]


def test_retrieve_top_k_relevant_images_out_of_vocab(fps):
    f = "gyroscope"
    assert f not in fps.vocab
    k = 100
    relevant = fps.retrieve_top_k_relevant_images(f, k, dataset="wicsmmir")
    assert len(relevant) == k


# TODO
#  - also test for different dataset (coco)
#  - generify similar to context tests

def test_retrieve_top_k_relevant_images(fps):
    k = 1000
    for i in range(3):
        start = time.time()
        relevant = fps.retrieve_top_k_relevant_images("green building", k, dataset="wicsmmir", weight_by_sim=False)
        logger.debug(f"{i}th run took {time.time() - start}s")
        assert len(relevant) == k


def test_retrieve_top_k_relevant_images_weight_by_sim(fps):
    k = 1000
    for i in range(3):
        start = time.time()
        relevant = fps.retrieve_top_k_relevant_images("green building", k, dataset="wicsmmir", weight_by_sim=True)
        logger.debug(f"{i}th run took {time.time() - start}s")
        assert len(relevant) == k
