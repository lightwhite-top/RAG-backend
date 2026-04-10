"""离线检索评测工具。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class RetrievalMetrics:
    """描述一次检索召回的评测值。"""

    recall: float
    mrr: float
    ndcg: float


def recall_at_k[T](predictions: Sequence[T], relevant: set[T], k: int) -> float:
    """计算 Recall@k, 结果在 [0,1] 之间, relevant 为空时返回 0.0。"""
    if not relevant:
        return 0.0
    hits = sum(1 for doc in predictions[:k] if doc in relevant)
    return hits / len(relevant)


def mrr_at_k[T](predictions: Sequence[T], relevant: set[T], k: int) -> float:
    """计算 MRR@k，遇到第一个相关文档就返回倒数排名。"""
    for rank, doc in enumerate(predictions[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k[T](
    predictions: Sequence[T],
    relevance_map: Mapping[T, float],
    k: int,
) -> float:
    """计算 NDCG@k，支持预先定义每个文档的相关度权重。"""
    if not relevance_map:
        return 0.0

    def dcg(values: Sequence[float]) -> float:
        total = 0.0
        for idx, value in enumerate(values):
            total += (2.0**value - 1.0) / log2(idx + 2)
        return total

    actual = [relevance_map.get(doc, 0.0) for doc in predictions[:k]]
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg(actual) / ideal_dcg


def evaluate_retrieval[T](
    predictions: Sequence[T],
    relevant: set[T],
    relevance_map: Mapping[T, float] | None = None,
    k: int = 10,
) -> RetrievalMetrics:
    """一次检索得分的整表评估，返回 recall/mrr/ndcg 三指标。"""
    if relevance_map is None:
        relevance_map = {doc: 1.0 for doc in relevant}
    return RetrievalMetrics(
        recall=recall_at_k(predictions, relevant, k),
        mrr=mrr_at_k(predictions, relevant, k),
        ndcg=ndcg_at_k(predictions, relevance_map, k),
    )


def mean_retrieval_metrics(metrics: Iterable[RetrievalMetrics]) -> RetrievalMetrics:
    """批量检索评测的均值，metrics 为空时默认 0。"""
    metrics_list = list(metrics)
    if not metrics_list:
        return RetrievalMetrics(0.0, 0.0, 0.0)
    count = len(metrics_list)
    return RetrievalMetrics(
        recall=sum(metric.recall for metric in metrics_list) / count,
        mrr=sum(metric.mrr for metric in metrics_list) / count,
        ndcg=sum(metric.ndcg for metric in metrics_list) / count,
    )
