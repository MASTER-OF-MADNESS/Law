"""Central configuration. Everything comes from .env — nothing is hardcoded."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- AI providers ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    ai_timeout_seconds: float = 30.0

    # --- Web search ---
    tavily_api_key: str = ""
    web_search_max_results: int = 6

    # --- Local knowledge base ---
    # Corpora searched together: the Constitution (parsed per-Article) and
    # heading-structured guide files for the topics it doesn't cover —
    # general procedures, common offences, and tenancy/consumer matters.
    knowledge_file: str = "data/legal_knowledge.md"
    constitution_file: str = "General provisions all.md"
    offenses_file: str = "data/common_offenses.md"
    tenancy_consumer_file: str = "data/tenancy_and_consumer.md"
    lawyer_directory_file: str = "LAWOUD_Lawyer_Directory final.md"
    # Repealed Articles stay parsed but out of retrieval unless this is turned on.
    include_omitted_articles: bool = False
    # Raised from the single-file defaults: with ~460 Article sections in the
    # corpus a low bar passes on almost any query, and web search would never run.
    knowledge_min_score: float = 8.0
    knowledge_min_coverage: float = 0.5
    knowledge_min_context_chars: int = 400
    knowledge_top_k: int = 4
    knowledge_max_section_chars: int = 4000

    # --- Semantic retrieval (hybrid local search) ---
    # Optional sentence-transformers layer blended with the keyword scorer.
    # Set false to run keyword-only; the same fallback applies automatically
    # when the package is missing or the model fails to load.
    semantic_search_enabled: bool = True
    semantic_model_name: str = "all-MiniLM-L6-v2"
    # Corpus embeddings are cached here, keyed by content hash + model name, so
    # boot-time embedding doesn't re-run unless a corpus file actually changed.
    semantic_cache_dir: str = ".cache/embeddings"
    # While semantic search is active, scored sections carry a combined 0-1
    # score (0.6 keyword-normalized + 0.4 cosine) and the sufficiency gate
    # compares against this instead of knowledge_min_score.
    knowledge_min_combined_score: float = 0.4

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "*"
    log_level: str = "INFO"

    @staticmethod
    def _resolve(file_ref: str) -> Path:
        """Absolute path for a configured file, resolved against the project root."""
        p = Path(file_ref)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def knowledge_path(self) -> Path:
        return self._resolve(self.knowledge_file)

    @property
    def constitution_path(self) -> Path:
        return self._resolve(self.constitution_file)

    @property
    def offenses_path(self) -> Path:
        return self._resolve(self.offenses_file)

    @property
    def tenancy_consumer_path(self) -> Path:
        return self._resolve(self.tenancy_consumer_file)

    @property
    def semantic_cache_path(self) -> Path:
        return self._resolve(self.semantic_cache_dir)

    @property
    def lawyer_directory_path(self) -> Path:
        return self._resolve(self.lawyer_directory_file)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key.strip())

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key.strip())

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
