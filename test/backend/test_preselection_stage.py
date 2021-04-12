import time

import pytest
from loguru import logger

from preselection.preselection_stage import PreselectionStage, MergeOp


@pytest.fixture
def ps():
    return PreselectionStage()


@pytest.fixture
def inp():
    ctx = "Stingrays exhibit a wide range of colors and patterns on their dorsal surface to help them camouflage with " \
          "the sandy bottom."
    focus = "Stingrays"
    return ctx, focus


def test_retrieve_relevant_images_defaults(ps: PreselectionStage, inp):
    max_rel = 5000
    ctx, foc = inp

    for i in range(3):
        start = time.time()
        relevant = ps.retrieve_relevant_images(context=ctx, focus=foc)
        logger.debug(f"Run {i} took {time.time() - start}s")
        assert len(relevant) != 0 and len(relevant) <= max_rel
        logger.info(f"Found {len(relevant)} images!")


def test_retrieve_relevant_images_merge_union(ps: PreselectionStage, inp):
    max_rel = 5000
    ctx, foc = inp

    for i in range(3):
        start = time.time()
        relevant = ps.retrieve_relevant_images(context=ctx, focus=foc, merge_op=MergeOp.union)
        logger.debug(f"Run {i} took {time.time() - start}s")
        assert len(relevant) != 0 and len(relevant) <= max_rel
        logger.info(f"Found {len(relevant)} images!")


def test_retrieve_relevant_images_exact_context(ps: PreselectionStage, inp):
    max_rel = 5000
    ctx, foc = inp

    for i in range(3):
        start = time.time()
        relevant = ps.retrieve_relevant_images(context=ctx, focus=foc, exact_context_retrieval=True)
        logger.debug(f"Run {i} took {time.time() - start}s")
        assert len(relevant) != 0 and len(relevant) <= max_rel
        logger.info(f"Found {len(relevant)} images!")


def test_retrieve_relevant_images_focus_weight_by_sim(ps: PreselectionStage, inp):
    max_rel = 5000
    ctx, foc = inp

    for i in range(3):
        start = time.time()
        relevant = ps.retrieve_relevant_images(context=ctx, focus=foc, focus_weight_by_sim=True)
        logger.debug(f"Run {i} took {time.time() - start}s")
        assert len(relevant) != 0 and len(relevant) <= max_rel
        logger.info(f"Found {len(relevant)} images!")


def test_retrieve_relevant_images_no_defaults(ps: PreselectionStage, inp):
    max_rel = 5000
    ctx, foc = inp

    for i in range(3):
        start = time.time()
        relevant = ps.retrieve_relevant_images(context=ctx, focus=foc,
                                               merge_op=MergeOp.union,
                                               focus_weight_by_sim=True,
                                               exact_context_retrieval=True)
        logger.debug(f"Run {i} took {time.time() - start}s")
        assert len(relevant) != 0 and len(relevant) <= max_rel
        logger.info(f"Found {len(relevant)} images!")
