import time

import pytest
from loguru import logger

from backend.preselection import PreselectionStage, MergeOp


@pytest.fixture
def ps():
    return PreselectionStage()


@pytest.fixture
def inp():
    ctx = ["Stingrays exhibit a wide range of colors and patterns on their dorsal surface to help them camouflage with "
           "the sandy bottom.",
           "The dog has been selectively bred over millennia for various behaviors, sensory capabilities, and physical "
           "attributes. Dogs are subclassified into breeds, which vary widely in shape, size and color."]
    focus = ["Stingrays", "dog"]
    return ctx, focus


def test_retrieve_relevant_images_defaults(ps: PreselectionStage, inp):
    max_rel = 5000

    for i in range(3):
        for c, f in inp:
            start = time.time()
            relevant = ps.retrieve_relevant_images(context=c, focus=f)
            logger.info(f"Run {f}.{i} took {time.time() - start}s")
            assert len(relevant) != 0 and len(relevant) <= max_rel
            logger.info(f"Found {len(relevant)} images!")


def test_retrieve_relevant_images_merge_union(ps: PreselectionStage, inp):
    max_rel = 5000
    ctx, foc = inp

    for i in range(3):
        for c, f in inp:
            start = time.time()
            relevant = ps.retrieve_relevant_images(context=c, focus=f, merge_op=MergeOp.UNION)
            logger.info(f"Run {f}.{i} took {time.time() - start}s")
            assert len(relevant) != 0 and len(relevant) <= max_rel
            logger.info(f"Found {len(relevant)} images!")


def test_retrieve_relevant_images_exact_context(ps: PreselectionStage, inp):
    max_rel = 5000
    ctx, foc = inp

    for i in range(3):
        for c, f in inp:
            start = time.time()
            relevant = ps.retrieve_relevant_images(context=c, focus=f, exact_context_retrieval=True)
            logger.info(f"Run {f}.{i} took {time.time() - start}s")
            assert len(relevant) != 0 and len(relevant) <= max_rel
            logger.info(f"Found {len(relevant)} images!")


def test_retrieve_relevant_images_focus_weight_by_sim(ps: PreselectionStage, inp):
    max_rel = 5000
    ctx, foc = inp

    for i in range(3):
        for c, f in inp:
            start = time.time()
            relevant = ps.retrieve_relevant_images(context=c, focus=f, focus_weight_by_sim=True)
            logger.info(f"Run {f}.{i} took {time.time() - start}s")
            assert len(relevant) != 0 and len(relevant) <= max_rel
            logger.info(f"Found {len(relevant)} images!")


def test_retrieve_relevant_images_no_defaults(ps: PreselectionStage, inp):
    max_rel = 5000
    ctx, foc = inp

    for i in range(3):
        for c, f in inp:
            start = time.time()
            relevant = ps.retrieve_relevant_images(context=c, focus=f,
                                                   merge_op=MergeOp.UNION,
                                                   focus_weight_by_sim=True,
                                                   exact_context_retrieval=True)
            logger.info(f"Run {f}.{i} took {time.time() - start}s")
            assert len(relevant) != 0 and len(relevant) <= max_rel
            logger.info(f"Found {len(relevant)} images!")
