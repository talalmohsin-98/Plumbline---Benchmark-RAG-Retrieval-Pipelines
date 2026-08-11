"""Application configuration via pydantic-settings: every environment variable."""

from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every runtime setting for the backend.

    Secrets are declared without defaults on purpose: a missing one must fail
    loudly at startup rather than silently fall back to something that
    "works" locally and breaks in the Space.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Secrets. No defaults, deliberately. ---
    database_url: SecretStr
    groq_api_key: SecretStr
    hf_token: SecretStr

    # --- Models ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_base: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_tuned: str = "talalmohsin-98/plumbline-reranker-v1"

    # --- Retrieval parameters ---
    # RRF_K=60 is the value from Cormack et al. It is a parameter this project
    # can measure rather than assume — see ARCHITECTURE §11.
    rrf_k: int = 60
    # Cross-encoder cost is linear in candidates and is the largest CPU item in
    # the system, so only the top RERANK_DEPTH of the fused list is reranked.
    rerank_depth: int = 20
    retrieve_depth: int = 50

    # --- Chunking ---
    # 512 tokens is the max sequence length of bge-small; chunks longer than
    # this would be silently truncated at embed time. 64 tokens of overlap
    # (12.5%) keeps a sentence that straddles a boundary intact in one of the
    # two chunks.
    chunk_size: int = 512
    chunk_overlap: int = 64

    @model_validator(mode="after")
    def _check_invariants(self) -> "Settings":
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if self.rerank_depth > self.retrieve_depth:
            raise ValueError(
                "rerank_depth cannot exceed retrieve_depth: there would be "
                "nothing to rerank beyond what was retrieved"
            )
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructed once.

    Deliberately a function rather than a module-level instance: importing
    `backend.config` must not require a populated environment, so that the
    test suite runs with no keys.
    """
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
