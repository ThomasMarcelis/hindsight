"""Tests for local SentenceTransformers embedding profile configuration."""

from dataclasses import dataclass

import numpy as np
import pytest

LOCAL_EMBEDDING_ENV_VARS = [
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_QUERY_PROMPT_NAME",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_QUERY_PROMPT",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_DOCUMENT_PROMPT_NAME",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_DOCUMENT_PROMPT",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_TRUNCATE_DIM",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_NORMALIZE",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_BATCH_SIZE",
]


@pytest.fixture(autouse=True)
def clean_local_embedding_env(monkeypatch):
    """Keep local embedding env tests independent from developer/runtime settings."""
    from hindsight_api.config import clear_config_cache

    clear_config_cache()
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    for env_var in LOCAL_EMBEDDING_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    yield

    clear_config_cache()


@dataclass
class FakeEncodeCall:
    texts: list[str]
    batch_size: int
    prompt_name: str | None
    prompt: str | None
    truncate_dim: int | None
    normalize_embeddings: bool


class FakeSentenceTransformerModel:
    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.calls: list[FakeEncodeCall] = []

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        show_progress_bar: bool,
        batch_size: int,
        prompt_name: str | None,
        prompt: str | None,
        truncate_dim: int | None,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert convert_to_numpy is True
        assert show_progress_bar is False
        self.calls.append(
            FakeEncodeCall(
                texts=list(texts),
                batch_size=batch_size,
                prompt_name=prompt_name,
                prompt=prompt,
                truncate_dim=truncate_dim,
                normalize_embeddings=normalize_embeddings,
            )
        )
        effective_dim = truncate_dim or self.dimension
        return np.ones((len(texts), effective_dim), dtype=float)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


def test_local_embedding_config_defaults_preserve_legacy_behavior():
    from hindsight_api.config import HindsightConfig

    config = HindsightConfig.from_env()

    assert config.embeddings_local_query_prompt_name is None
    assert config.embeddings_local_query_prompt is None
    assert config.embeddings_local_document_prompt_name is None
    assert config.embeddings_local_document_prompt is None
    assert config.embeddings_local_truncate_dim is None
    assert config.embeddings_local_normalize is False
    assert config.embeddings_local_batch_size == 32


def test_local_embedding_config_reads_prompt_truncation_and_batch_env(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_QUERY_PROMPT_NAME", "query")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_DOCUMENT_PROMPT", "passage: ")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_TRUNCATE_DIM", "2000")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_NORMALIZE", "true")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_BATCH_SIZE", "7")

    config = HindsightConfig.from_env()

    assert config.embeddings_local_query_prompt_name == "query"
    assert config.embeddings_local_document_prompt == "passage: "
    assert config.embeddings_local_truncate_dim == 2000
    assert config.embeddings_local_normalize is True
    assert config.embeddings_local_batch_size == 7


def test_create_embeddings_from_env_wires_local_profile_config(monkeypatch):
    from hindsight_api.engine.embeddings import LocalSTEmbeddings, create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU", "true")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE", "true")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_QUERY_PROMPT_NAME", "query")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_DOCUMENT_PROMPT", "passage: ")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_TRUNCATE_DIM", "2000")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_NORMALIZE", "1")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_BATCH_SIZE", "3")

    embeddings = create_embeddings_from_env()

    assert isinstance(embeddings, LocalSTEmbeddings)
    assert embeddings.model_name == "Qwen/Qwen3-Embedding-4B"
    assert embeddings.force_cpu is True
    assert embeddings.trust_remote_code is True
    assert embeddings.query_prompt_name == "query"
    assert embeddings.document_prompt == "passage: "
    assert embeddings.truncate_dim == 2000
    assert embeddings.normalize_embeddings is True
    assert embeddings.batch_size == 3


def test_local_embeddings_route_query_and_document_encode_options():
    from hindsight_api.engine.embeddings import LocalSTEmbeddings

    embeddings = LocalSTEmbeddings(
        model_name="test-model",
        query_prompt_name="query",
        document_prompt="passage: ",
        truncate_dim=4,
        normalize_embeddings=True,
        batch_size=7,
    )
    fake_model = FakeSentenceTransformerModel(dimension=8)
    embeddings._model = fake_model
    embeddings._dimension = 4

    query_vectors = embeddings.encode_query(["find memory"])
    document_vectors = embeddings.encode_documents(["stored memory"])
    default_vectors = embeddings.encode(["default document side"])

    assert len(query_vectors[0]) == 4
    assert len(document_vectors[0]) == 4
    assert len(default_vectors[0]) == 4

    assert fake_model.calls[0] == FakeEncodeCall(
        texts=["find memory"],
        batch_size=7,
        prompt_name="query",
        prompt=None,
        truncate_dim=4,
        normalize_embeddings=True,
    )
    assert fake_model.calls[1] == FakeEncodeCall(
        texts=["stored memory"],
        batch_size=7,
        prompt_name=None,
        prompt="passage: ",
        truncate_dim=4,
        normalize_embeddings=True,
    )
    assert fake_model.calls[2].texts == ["default document side"]
    assert fake_model.calls[2].prompt == "passage: "


def test_local_embedding_profile_fingerprint_tracks_same_dimension_model_swaps():
    from hindsight_api.engine.embeddings import LocalSTEmbeddings

    base = LocalSTEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    base._dimension = 1024
    prompt_aligned = LocalSTEmbeddings(model_name="BAAI/bge-large-en-v1.5", query_prompt_name="query")
    prompt_aligned._dimension = 1024
    different_model = LocalSTEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l-v2.0")
    different_model._dimension = 1024

    assert len(base.profile_fingerprint) == 64
    assert base.profile_fingerprint != prompt_aligned.profile_fingerprint
    assert base.profile_fingerprint != different_model.profile_fingerprint


def test_local_embedding_profile_fingerprint_ignores_operational_batch_size():
    from hindsight_api.engine.embeddings import LocalSTEmbeddings

    small_batch = LocalSTEmbeddings(model_name="BAAI/bge-large-en-v1.5", batch_size=1)
    small_batch._dimension = 1024
    large_batch = LocalSTEmbeddings(model_name="BAAI/bge-large-en-v1.5", batch_size=32)
    large_batch._dimension = 1024

    assert small_batch.profile_fingerprint == large_batch.profile_fingerprint


def test_local_embedding_constructor_rejects_ambiguous_prompt_config():
    from hindsight_api.engine.embeddings import LocalSTEmbeddings

    with pytest.raises(ValueError, match="query_prompt_name and query_prompt"):
        LocalSTEmbeddings(query_prompt_name="query", query_prompt="query: ")

    with pytest.raises(ValueError, match="document_prompt_name and document_prompt"):
        LocalSTEmbeddings(document_prompt_name="passage", document_prompt="passage: ")


def test_local_embedding_constructor_rejects_invalid_truncation_and_batch_size():
    from hindsight_api.engine.embeddings import LocalSTEmbeddings

    with pytest.raises(ValueError, match="truncate_dim must be >= 1"):
        LocalSTEmbeddings(truncate_dim=0)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        LocalSTEmbeddings(batch_size=0)
