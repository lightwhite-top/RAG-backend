from __future__ import annotations

from math import log2

import pytest

from baozhi_rag.services.retrieval_eval import (
    RetrievalMetrics,
    evaluate_retrieval,
    mean_retrieval_metrics,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_mrr_ndcg_single_query() -> None:
    predictions = ["intro", "anchor", "summary", "appendix"]
    relevant = {"anchor", "appendix"}
    relevance_map = {"anchor": 1.0, "summary": 0.2, "appendix": 1.0}

    assert recall_at_k(predictions, relevant, k=3) == pytest.approx(0.5)
    assert mrr_at_k(predictions, relevant, k=4) == pytest.approx(0.5)

    # 手动计算 ideal dcg 为 1 + 1/log2(3)，实际 dcg 为 1/log2(3) + 1/log2(4)
    actual_dcg = 1.0 / log2(3) + (2**0.2 - 1.0) / log2(4)
    ideal_dcg = 1.0 + 1.0 / log2(3) + (2**0.2 - 1.0) / log2(4)
    expected_ndcg = actual_dcg / ideal_dcg
    assert ndcg_at_k(predictions, relevance_map, k=3) == pytest.approx(expected_ndcg)


def test_evaluate_retrieval_defaults_to_binary_grades() -> None:
    metrics = evaluate_retrieval(
        predictions=["doc1", "doc2"],
        relevant={"doc2"},
        k=2,
    )

    assert metrics.recall == pytest.approx(1.0)
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.ndcg == pytest.approx(1.0 / log2(3))


def test_mean_retrieval_metrics_reduces_list() -> None:
    metrics = [
        RetrievalMetrics(recall=1.0, mrr=0.5, ndcg=1.0),
        RetrievalMetrics(recall=0.0, mrr=0.0, ndcg=0.2),
    ]

    averaged = mean_retrieval_metrics(metrics)

    assert averaged.recall == pytest.approx(0.5)
    assert averaged.mrr == pytest.approx(0.25)
    assert averaged.ndcg == pytest.approx(0.6)


def test_mean_retrieval_metrics_handles_empty_iterable() -> None:
    averaged = mean_retrieval_metrics([])
    assert averaged == RetrievalMetrics(recall=0.0, mrr=0.0, ndcg=0.0)
