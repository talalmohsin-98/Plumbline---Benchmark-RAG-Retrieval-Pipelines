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

    # --- Secrets ---
    # DATABASE_URL has no default: everything that reads config also reads the
    # store, so failing at construction is failing at the right moment.
    database_url: SecretStr
    # These two are optional here and checked at the point of use instead. An
    # entrypoint must not demand a secret it never reads — `ingest` touches
    # neither Groq nor the Hub, and requiring them made it unrunnable.
    groq_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    # --- Models ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_base: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_tuned: str = "talalmohsin-98/plumbline-reranker-v1"
    # The groundedness judge. Cheap and fast is the right trade for scoring
    # one sentence at a time against supplied context.
    groq_model: str = "llama-3.1-8b-instant"
    # Gold-set drafting is a different job: seven simultaneous constraints,
    # where 8b-instant took the cheapest path through each and produced
    # questions that had to be dropped by hand. Deliberately a separate
    # setting so improving one never silently changes the other.
    goldset_model: str = "openai/gpt-oss-120b"

    # bge-*-en-v1.5 is trained asymmetrically: queries carry this instruction,
    # passages are embedded bare. Not wired into retrieval yet — that is a
    # measured decision for the dense lane, not an assumed one.
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "

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


class MissingSecretError(RuntimeError):
    """A secret was needed at the point of use but is not configured."""

    def __init__(self, variable: str, purpose: str) -> None:
        super().__init__(
            f"{variable} is not set, and it is required for {purpose}. "
            f"Add {variable}=... to your .env file (see .env.example)."
        )
        self.variable = variable


def require_groq_api_key() -> str:
    """Return the Groq key, failing with a message that names the variable.

    Call this at the top of anything that talks to Groq, so the failure lands
    before any work is done rather than partway through a 150-call run.
    """
    key = get_settings().groq_api_key
    if key is None:
        raise MissingSecretError("GROQ_API_KEY", "calling the Groq API")
    return key.get_secret_value()


def require_hf_token() -> str:
    """Return the Hugging Face token, failing with a message that names it."""
    token = get_settings().hf_token
    if token is None:
        raise MissingSecretError("HF_TOKEN", "authenticating with the Hugging Face Hub")
    return token.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructed once.

    Deliberately a function rather than a module-level instance: importing
    `backend.config` must not require a populated environment, so that the
    test suite runs with no keys.
    """
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
