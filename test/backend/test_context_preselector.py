import time

import pytest
from loguru import logger

from preselection.context.context_preselector import ContextPreselector


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


def test_retrieve_top_k_relevant_images(cps, ks, qs):
    for k in ks:
        logger.debug(f'Running retrieve_top_k_relevant_images with k={k} and exact=False')
        for q in qs:
            start = time.time()
            relevant = cps.retrieve_top_k_relevant_images(q, k, exact=False)
            logger.debug(f"First took {time.time() - start}s")
            # assert len(relevant) == k

            start = time.time()
            relevant = cps.retrieve_top_k_relevant_images(q, k, exact=False)
            logger.debug(f"Second took {time.time() - start}s")
            # assert len(relevant) == k

            start = time.time()
            relevant = cps.retrieve_top_k_relevant_images(q, k, exact=False)
            logger.debug(f"Third took {time.time() - start}s")
            # assert len(relevant) == k


def test_retrieve_top_k_relevant_images_exact(cps, ks, qs):
    for k in ks:
        logger.debug(f'Running retrieve_top_k_relevant_images with k={k} and exact=True')
        for q in qs:
            start = time.time()
            relevant = cps.retrieve_top_k_relevant_images(q, k, exact=True)
            logger.debug(f"First took {time.time() - start}s")
            assert len(relevant) == k

            start = time.time()
            relevant = cps.retrieve_top_k_relevant_images(q, k, exact=True)
            logger.debug(f"Second took {time.time() - start}s")
            assert len(relevant) == k

            start = time.time()
            relevant = cps.retrieve_top_k_relevant_images(q, k, exact=True)
            logger.debug(f"Third took {time.time() - start}s")
            assert len(relevant) == k
