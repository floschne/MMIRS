import time

import pytest
from loguru import logger

from backend.preselection import ContextPreselector


@pytest.fixture
def cps():
    return ContextPreselector()


@pytest.fixture
def ks():
    return [10, 100, 1000, 10000]


@pytest.fixture
def qs():
    return ["A brown dog is playing with a red ball",
            "Original white Wii standing upright on its stand next to a Wii Remote."]


@pytest.fixture
def ds():
    return ["wicsmmir", "coco"]


def test_retrieve_top_k_relevant_images(cps, ks, ds, qs):
    for d in ds:
        for k in ks:
            logger.debug(f'Running retrieve_top_k_relevant_images with k={k} and exact=False')
            for q in qs:
                for i in range(3):
                    start = time.time()
                    relevant = cps.retrieve_top_k_relevant_images(q, k, dataset=d, exact=False)
                    logger.debug(f"{i}th run with k={k} took {time.time() - start}s")
                    assert len(relevant) == k


def test_retrieve_top_k_relevant_images_exact(cps, ks, qs):
    for d in ds:
        for k in ks:
            logger.debug(f'Running retrieve_top_k_relevant_images with k={k} and exact=True')
            for q in qs:
                for i in range(3):
                    start = time.time()
                    relevant = cps.retrieve_top_k_relevant_images(q, k, dataset=d, exact=True)
                    logger.debug(f"{i}th run with k={k} took {time.time() - start}s")
                    assert len(relevant) == k
