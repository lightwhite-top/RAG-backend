from __future__ import annotations

import os
from unittest.mock import patch

from baozhi_rag.core.config import Settings


def test_settings_accept_search_candidate_and_rerank_env_aliases() -> None:
    with patch.dict(
        os.environ,
        {
            "APP_DEBUG": "false",
            "LLM_RERANK_MODEL": "text-rerank-v1",
            "LLM_DEEP_RERANK_MODEL": "deep-rerank-v2",
            "LLM_IMAGE_RERANK_MODEL": "image-rerank-v1",
            "SEARCH_LEXICAL_CANDIDATE_SIZE": "55",
            "SEARCH_VECTOR_CANDIDATE_SIZE": "66",
            "SEARCH_RRF_LEXICAL_WEIGHT": "1.2",
            "SEARCH_RRF_VECTOR_WEIGHT": "1.4",
            "SEARCH_DEEP_RERANK_SCORE_THRESHOLD": "0.21",
            "SEARCH_DEEP_RERANK_MARGIN_THRESHOLD": "0.03",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)

    assert settings.rerank_model == "text-rerank-v1"
    assert settings.deep_rerank_model == "deep-rerank-v2"
    assert settings.image_rerank_model == "image-rerank-v1"
    assert settings.search_lexical_candidate_size == 55
    assert settings.search_vector_candidate_size == 66
    assert settings.search_rrf_lexical_weight == 1.2
    assert settings.search_rrf_vector_weight == 1.4
    assert settings.search_deep_rerank_score_threshold == 0.21
    assert settings.search_deep_rerank_margin_threshold == 0.03
