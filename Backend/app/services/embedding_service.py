"""Optional semantic-retrieval backend: local sentence-transformers embeddings.

Deliberately thin — the service owns the model handle and nothing else. If the
package is missing, the model cannot be downloaded, or loading fails for any
reason, ``is_available`` stays False and every ``encode()`` returns None. The
knowledge service treats that as "run keyword-only", so a broken embedding
stack can never take retrieval down with it.

The model is loaded eagerly at construction (server startup) rather than on
first query — a multi-second download is acceptable once at boot, not
mid-request.
"""

import logging

import numpy as np

from app.config import Settings

logger = logging.getLogger("lawoud.embedding_service")


class EmbeddingService:
    def __init__(self, settings: Settings):
        self._model = None
        self._available = False
        if not settings.semantic_search_enabled:
            logger.info(
                "Semantic search disabled (SEMANTIC_SEARCH_ENABLED=false); keyword-only retrieval."
            )
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers is not installed; keyword-only retrieval. "
                "Install it or set SEMANTIC_SEARCH_ENABLED=false to silence this."
            )
            return
        try:
            self._model = SentenceTransformer(settings.semantic_model_name)
            self._available = True
            logger.info("Embedding model loaded: %s", settings.semantic_model_name)
        except Exception as e:
            logger.warning(
                "Embedding model %s failed to load (%s); keyword-only retrieval.",
                settings.semantic_model_name,
                e,
            )

    @property
    def is_available(self) -> bool:
        return self._available

    def encode(self, texts: list[str]) -> np.ndarray | None:
        """L2-normalized embeddings, one row per text. None when unavailable."""
        if not self._available or not texts:
            return None
        try:
            vecs = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(vecs, dtype=np.float32)
        except Exception as e:
            # A model that fails mid-run stays failed — flapping between hybrid
            # and keyword scoring would make the sufficiency gate inconsistent
            # about which threshold applies.
            logger.warning("Embedding encode failed (%s); disabling semantic search.", e)
            self._available = False
            return None

    def encode_query(self, text: str) -> np.ndarray | None:
        """Single normalized query embedding. None when unavailable."""
        vecs = self.encode([text])
        return vecs[0] if vecs is not None else None
