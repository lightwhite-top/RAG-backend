"""FMM/BMM 词匹配服务测试。"""

from __future__ import annotations

from pathlib import Path

from baozhi_rag.services.term_matching import build_default_term_matcher


def test_maximum_matching_extracts_fmm_bmm_and_merged_terms() -> None:
    """FMM/BMM 应输出各自结果并合并去重。"""
    matcher = build_default_term_matcher()

    result = matcher.extract_terms("保险责任免除和免赔额说明")

    assert result.fmm_terms == ["保险责任", "免赔额"]
    assert result.bmm_terms == ["保险", "责任免除", "免赔额"]
    assert result.merged_terms == ["保险责任", "免赔额", "保险", "责任免除"]


def test_maximum_matching_loads_extra_terms_from_file(tmp_path: Path) -> None:
    """扩展词典应在默认词典基础上叠加新词。"""
    dictionary_path = tmp_path / "insurance_terms.txt"
    dictionary_path.write_text("# 注释\n保单贷款\n", encoding="utf-8")
    matcher = build_default_term_matcher(dictionary_path)

    result = matcher.extract_terms("客户申请保单贷款")

    assert result.fmm_terms == ["保单贷款"]
    assert result.bmm_terms == ["保单贷款"]
    assert result.merged_terms == ["保单贷款"]
