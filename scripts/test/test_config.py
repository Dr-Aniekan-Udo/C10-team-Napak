from pathlib import Path

import pytest

import scripts.utilities.config as config_module
from scripts.utilities.config import AppConfig


def _environment(**overrides: str) -> dict[str, str]:
    values = {
        "AGRO_RAG_API_BASE_URL": "http://localhost:11434/v1",
        "AGRO_RAG_API_KEY": "null",
        "AGRO_RAG_MODEL": "local-model",
        "AGRO_RAG_DATA_DIR": "data",
        "AGRO_RAG_SESSION_DIR": ".local/sessions",
        "AGRO_RAG_RANKER": "cross_encoder",
        "AGRO_RAG_RANK_MODEL": "rank-model",
        "AGRO_RAG_CANDIDATE_MODEL": "candidate-model",
        "AGRO_RAG_TOP_K": "5",
    }
    values.update(overrides)
    return values


def test_from_env_normalizes_nullable_key_and_paths() -> None:
    config = AppConfig.from_env(_environment())

    assert config.api_key is None
    assert config.api_base_url == "http://localhost:11434/v1"
    assert config.data_dir == Path("data")
    assert config.session_dir == Path(".local/sessions")
    assert config.top_k == 5
    assert config.candidate_model == "candidate-model"


def test_from_env_accepts_cloud_key_and_preserves_rank_model() -> None:
    config = AppConfig.from_env(
        _environment(
            AGRO_RAG_API_KEY="secret",
            AGRO_RAG_MODEL="generation-model",
            AGRO_RAG_RANK_MODEL="independent-rank-model",
        )
    )

    assert config.api_key == "secret"
    assert config.model == "generation-model"
    assert config.rank_model == "independent-rank-model"


def test_from_env_keeps_candidate_model_separate_from_rank_model() -> None:
    config = AppConfig.from_env(
        _environment(
            AGRO_RAG_RANK_MODEL="cross-encoder-model",
            AGRO_RAG_CANDIDATE_MODEL="dense-candidate-model",
        )
    )

    assert config.rank_model == "cross-encoder-model"
    assert config.candidate_model == "dense-candidate-model"


@pytest.mark.parametrize("value", ["", "   ", "null", "NULL", "none", "None"])
def test_from_env_treats_empty_and_null_keys_as_none(value: str) -> None:
    assert AppConfig.from_env(_environment(AGRO_RAG_API_KEY=value)).api_key is None


@pytest.mark.parametrize(
    "missing",
    ["AGRO_RAG_API_BASE_URL", "AGRO_RAG_MODEL", "AGRO_RAG_DATA_DIR"],
)
def test_from_env_rejects_missing_required_values(missing: str) -> None:
    environment = _environment()
    del environment[missing]

    with pytest.raises(ValueError, match=missing):
        AppConfig.from_env(environment)


def test_from_env_rejects_invalid_ranker_and_top_k() -> None:
    with pytest.raises(ValueError, match="AGRO_RAG_RANKER"):
        AppConfig.from_env(_environment(AGRO_RAG_RANKER="unknown"))
    with pytest.raises(ValueError, match="AGRO_RAG_TOP_K"):
        AppConfig.from_env(_environment(AGRO_RAG_TOP_K="0"))


@pytest.mark.parametrize(
    "name",
    [
        "AGRO_RAG_API_BASE_URL",
        "AGRO_RAG_API_KEY",
        "AGRO_RAG_MODEL",
        "AGRO_RAG_DATA_DIR",
        "AGRO_RAG_SESSION_DIR",
        "AGRO_RAG_RANKER",
        "AGRO_RAG_RANK_MODEL",
        "AGRO_RAG_CANDIDATE_MODEL",
        "AGRO_RAG_TOP_K",
    ],
)
@pytest.mark.parametrize("source", ["mapping", "process"])
def test_from_env_rejects_non_string_values(
    name: str,
    source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment()
    environment[name] = object()  # type: ignore[assignment]

    if source == "process":
        monkeypatch.setattr(config_module.os, "environ", environment)
        environ = None
    else:
        environ = environment

    with pytest.raises(ValueError, match=name):
        AppConfig.from_env(environ)


def test_explicit_environment_mapping_wins_over_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGRO_RAG_MODEL", "process-model")

    config = AppConfig.from_env(_environment(AGRO_RAG_MODEL="explicit-model"))

    assert config.model == "explicit-model"


def test_dotenv_file_loads_with_process_and_mapping_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "package" / "utilities"
    package_root.mkdir(parents=True)
    (tmp_path / ".env").write_text(
        "AGRO_RAG_API_BASE_URL=https://dotenv.example/v1\n"
        "AGRO_RAG_MODEL=dotenv-model\n"
        "AGRO_RAG_DATA_DIR=~/dotenv-data\n"
        "export AGRO_RAG_RANK_MODEL=dotenv-rank-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "__file__", str(package_root / "config.py"))
    for name in (
        "AGRO_RAG_API_BASE_URL",
        "AGRO_RAG_MODEL",
        "AGRO_RAG_DATA_DIR",
        "AGRO_RAG_RANK_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGRO_RAG_MODEL", "process-model")

    config = AppConfig.from_env({"AGRO_RAG_MODEL": "explicit-model"})

    assert config.api_base_url == "https://dotenv.example/v1"
    assert config.data_dir == Path("~/dotenv-data").expanduser()
    assert config.rank_model == "dotenv-rank-model"
    assert config.model == "explicit-model"
