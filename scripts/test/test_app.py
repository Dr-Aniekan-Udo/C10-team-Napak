from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from scripts.utilities.models import RetrievedDocument
from scripts.utilities.sessions import SessionStore, SessionSummary


@dataclass
class FakePipeline:
    """Factory-only double: app construction must not call pipeline behavior."""

    stream_calls: int = 0


class PublicSessionStore:
    def __init__(self) -> None:
        self.list_calls = 0

    @property
    def root(self) -> Path:
        raise AssertionError("app must not inspect session storage paths")

    def list_summaries(self) -> tuple[SessionSummary, ...]:
        self.list_calls += 1
        return (
            SessionSummary(
                session_id="00000000-0000-4000-8000-000000000001",
                question="How should I scout maize?",
                modified_at=1_700_000_000.0,
            ),
        )


class PrivateProcessorTrap:
    @property
    def _documents(self):
        raise AssertionError("app must use public corpus APIs")

    @property
    def _metadata(self):
        raise AssertionError("app must use public corpus APIs")


class PublicApiPipeline:
    def __init__(self) -> None:
        self.sessions = PublicSessionStore()
        self.processor = PrivateProcessorTrap()
        self.corpus_count = 12
        self.document_metadata = {
            "doc-1": {
                "title": "Maize note",
                "source": "Extension service",
                "crop": "maize",
            }
        }


def _config_text(app: gr.Blocks) -> str:
    return json.dumps(app.get_config_file(), default=str)


def test_import_and_factory_smoke_without_launch_or_pipeline_work(
    monkeypatch,
) -> None:
    import scripts.app as app_module

    def fail_launch(*_args, **_kwargs):
        raise AssertionError("factory must not launch Gradio")

    monkeypatch.setattr(gr.Blocks, "launch", fail_launch)
    pipeline = FakePipeline()
    app = app_module.create_app(pipeline)  # type: ignore[arg-type]

    assert isinstance(app, gr.Blocks)
    assert pipeline.stream_calls == 0
    config_text = _config_text(app)
    assert "Ask the extension library." in config_text
    assert "Educational research tool" in config_text
    assert app.elem_id == "agro-rag-app"
    assert "#agro-rag-app" in app.get_config_file()["css"]


class StreamingPipeline:
    def __init__(self, root: Path, *, error: Exception | None = None) -> None:
        self.sessions = SessionStore(root)
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def stream_chat(self, session_id: str, question: str):
        self.calls.append((session_id, question))
        if self.error is not None:
            raise self.error
        source = RetrievedDocument(
            "doc-1",
            "Maize note",
            "Grounded facts",
            0.97,
            1,
            "Extension service",
            "https://example.test/doc-1",
            "maize",
            "Kenya",
            "synthetic",
            "synthetic (CC0)",
        )
        yield "Grounded", (source,)
        yield " answer [doc:doc-1]", (source,)


def test_stream_handler_shows_retrieval_streaming_and_saved_states(tmp_path: Path) -> None:
    import scripts.app as app_module

    pipeline = StreamingPipeline(tmp_path)
    handler = app_module._make_message_handler(pipeline, {})

    updates = list(handler("What grows?", None, [], []))

    assert [update[3] for update in updates] == [
        app_module.SEARCHING_STATUS,
        app_module.STREAMING_STATUS,
        app_module.STREAMING_STATUS,
        app_module.SAVED_STATUS,
    ]
    assert updates[-1][0][-1]["content"] == "Grounded answer [doc:doc-1]"
    assert "Maize note" in updates[-1][4]
    assert "Kenya" in updates[-1][4]
    assert pipeline.calls == [(updates[-1][7], "What grows?")]


def test_stream_handler_preserves_prior_history_on_failure(tmp_path: Path) -> None:
    import scripts.app as app_module

    pipeline = StreamingPipeline(tmp_path, error=RuntimeError("provider-secret"))
    handler = app_module._make_message_handler(pipeline, {})
    prior_history = [{"role": "assistant", "content": "Prior answer [doc:old]"}]

    updates = list(handler("Try again", None, prior_history, []))

    assert updates[-1][3] == app_module.ERROR_STATUS
    assert updates[-1][0] == prior_history
    assert "provider-secret" not in str(updates[-1])


def test_factory_registers_stable_qa_element_ids() -> None:
    import scripts.app as app_module

    app = app_module.create_app(FakePipeline())  # type: ignore[arg-type]
    element_ids = {
        getattr(block, "elem_id", None)
        for block in app.blocks.values()
        if getattr(block, "elem_id", None)
    }
    assert app.elem_id == "agro-rag-app"

    for element_id in (
        "agro-header",
        "session-rail",
        "new-conversation",
        "session-list",
        "clear-conversation",
        "chat-panel",
        "chatbot",
        "empty-state",
        "message-input",
        "ask-button",
        "query-status",
        "evidence-rail",
        "evidence-cards",
    ):
        assert element_id in element_ids


def test_factory_uses_public_session_and_document_apis() -> None:
    import scripts.app as app_module

    pipeline = PublicApiPipeline()

    app = app_module.create_app(pipeline)  # type: ignore[arg-type]

    assert pipeline.sessions.list_calls == 1
    assert app_module._corpus_count(pipeline) == "12"
    assert app_module._metadata_for_pipeline(pipeline) == pipeline.document_metadata
    config_text = _config_text(app)
    assert "How should I scout maize?" in config_text


def test_source_cards_escape_metadata_and_do_not_expose_ranker_score() -> None:
    import scripts.app as app_module

    source = SimpleNamespace(
        document_id='doc"><script>alert(1)</script>',
        title="<img src=x onerror=alert(1)>",
        source="<b>Publisher</b>",
        crop="maize",
        country="Kenya",
        origin="synthetic",
        source_url="javascript:alert(1)",
        license="<em>CC0</em>",
        rank=1,
        score=0.998,
    )

    rendered = app_module.render_source_cards([source])

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "&lt;b&gt;Publisher&lt;/b&gt;" in rendered
    assert "&lt;em&gt;CC0&lt;/em&gt;" in rendered
    assert "javascript:alert(1)" not in rendered
    assert "0.998" not in rendered
    assert "confidence" not in rendered.casefold()
