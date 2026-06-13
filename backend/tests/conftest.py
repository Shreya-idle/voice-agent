"""
Global test configuration.

Patches HuggingFaceInferenceAPIEmbeddings at the module level so that
test_agent.py (which triggers a module-level RAGEngine() instantiation
via `from agent import Assistant`) does not make real HTTP calls to
the Hugging Face Inference API during collection.
"""
from unittest.mock import MagicMock, patch

_mock_embeddings_instance = MagicMock()
_mock_embeddings_instance.embed_documents.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
_mock_embeddings_instance.embed_query.return_value = [0.1] * 384

# This patch is activated at import time (before pytest collects test modules)
# so that any module-level RAGEngine() call gets a mocked embeddings object.
_patcher = patch(
    'langchain_community.embeddings.HuggingFaceInferenceAPIEmbeddings',
    return_value=_mock_embeddings_instance,
)
_patcher.start()

# Note: We intentionally do NOT call _patcher.stop() because we need
# the mock to remain active for the entire test session.
