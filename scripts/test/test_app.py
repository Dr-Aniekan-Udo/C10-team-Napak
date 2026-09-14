from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest

from scripts.utilities.models import ChatMessage, RetrievedDocument, Session, SessionTurn
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


class LifecycleRanker:
    def __init__(self, *, state: str = "ready", ready: bool = True) -> None:
        self._state = state
        self._ready = ready
        self.start_calls = 0
        self.wait_calls: list[float] = []
        self.shutdown_calls = 0
        self.state_reads = 0

    @property
    def state(self) -> str:
        self.state_reads += 1
        return self._state

    def start_warmup(self) -> bool:
        self.start_calls += 1
        return True

    def wait_until_ready(self, timeout: float = 0.0) -> bool:
        self.wait_calls.append(timeout)
        if self._ready:
            self._state = "ready"
        return self._ready

    def mark_unavailable(self) -> None:
        self._state = "unavailable"
        self._ready = False

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class PlainRanker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rank(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        self.calls.append((query, top_k))
        return []


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
    def __init__(
        self,
        root: Path,
        *,
        error: Exception | None = None,
        ranker: LifecycleRanker | None = None,
    ) -> None:
        self.sessions = SessionStore(root)
        self.error = error
        self.ranker = ranker or LifecycleRanker()
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


class PredictionFailurePipeline(StreamingPipeline):
    def stream_chat(self, session_id: str, question: str):
        self.calls.append((session_id, question))
        self.ranker.mark_unavailable()
        raise RuntimeError("prediction timed out")


def test_stream_handler_shows_retrieval_streaming_and_saved_states(tmp_path: Path) -> None:
    import scripts.app as app_module

    pipeline = StreamingPipeline(tmp_path)
    handler = app_module._make_message_handler(pipeline, {})

    updates = list(handler("What grows?", None, [], []))

    assert [update[3] for update in updates] == [
        app_module.RERANKER_LOADING_STATUS,
        app_module.RERANKER_READY_STATUS,
        app_module.SEARCHING_STATUS,
        app_module.GENERATING_STATUS,
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


def test_normalise_history_accepts_gradio_typed_text_blocks() -> None:
    import scripts.app as app_module

    value = [
        {"role": "user", "content": [{"type": "text", "text": "first question"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first answer"},
                {"type": "text", "text": " continued"},
            ],
        },
    ]

    assert app_module._normalise_history(value) == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer continued"},
    ]


def test_stream_handler_keeps_typed_prior_turn_on_second_send(tmp_path: Path) -> None:
    import scripts.app as app_module

    pipeline = StreamingPipeline(tmp_path)
    handler = app_module._make_message_handler(pipeline, {})

    first_updates = list(handler("First question", None, [], []))
    first_final = first_updates[-1]
    second_updates = list(handler("Second question", first_final[7], first_final[0], first_final[5]))
    second_final = second_updates[-1]

    assert [item["content"] for item in second_final[0]] == [
        "First question",
        "Grounded answer [doc:doc-1]",
        "Second question",
        "Grounded answer [doc:doc-1]",
    ]
    assert pipeline.calls[0][0] == pipeline.calls[1][0] == first_final[7]


def test_load_then_ask_keeps_persisted_turn(tmp_path: Path) -> None:
    import scripts.app as app_module

    pipeline = StreamingPipeline(tmp_path)
    session = pipeline.sessions.create()
    pipeline.sessions.save(
        Session(
            session.session_id,
            (
                SessionTurn(
                    ChatMessage("user", "Loaded question"),
                    ChatMessage("assistant", "Loaded answer"),
                ),
            ),
        )
    )

    loaded = app_module._load_conversation(pipeline, session.session_id, {})
    handler = app_module._make_message_handler(pipeline, {})
    updates = list(handler("New question", loaded[0], loaded[1], loaded[3]))
    final = updates[-1]

    assert [item["content"] for item in final[0]] == [
        "Loaded question",
        "Loaded answer",
        "New question",
        "Grounded answer [doc:doc-1]",
    ]
    assert final[7] == session.session_id
    assert pipeline.calls == [(session.session_id, "New question")]


def test_factory_starts_one_nonblocking_reranker_warmup() -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker(state="loading", ready=False)
    pipeline = SimpleNamespace(ranker=ranker)

    app = app_module.create_app(pipeline)  # type: ignore[arg-type]

    assert ranker.start_calls == 1
    assert ranker.wait_calls == []
    assert app_module.RERANKER_LOADING_STATUS in _config_text(app)


def test_factory_registers_one_second_nonqueued_reranker_status_timer() -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker(state="loading", ready=False)
    app = app_module.create_app(SimpleNamespace(ranker=ranker))  # type: ignore[arg-type]
    config = app.get_config_file()

    timers = [component for component in config["components"] if component["type"] == "timer"]
    poll_dependencies = [dependency for dependency in config["dependencies"] if dependency["queue"] is False]

    assert len(timers) == 1
    assert timers[0]["props"]["value"] == pytest.approx(1.0)
    assert len(poll_dependencies) == 1
    assert poll_dependencies[0]["inputs"]
    assert poll_dependencies[0]["outputs"]


def test_reranker_status_poll_reads_public_state_and_keeps_loading_active() -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker(state="loading", ready=False)
    pipeline = SimpleNamespace(ranker=ranker)

    status_update, timer_update = app_module._poll_reranker_status(
        pipeline,
        app_module.RERANKER_LOADING_STATUS,
    )

    assert status_update == app_module.RERANKER_LOADING_STATUS
    assert timer_update == gr.skip()
    assert ranker.state_reads == 1


@pytest.mark.parametrize(
    ("state", "expected_status"),
    [
        ("ready", "RERANKER_READY_STATUS"),
        ("unavailable", "RERANKER_UNAVAILABLE_STATUS"),
    ],
)
def test_reranker_status_poll_reports_terminal_state_and_stops_timer(
    state: str,
    expected_status: str,
) -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker(state=state, ready=state == "ready")
    pipeline = SimpleNamespace(ranker=ranker)

    status_update, timer_update = app_module._poll_reranker_status(
        pipeline,
        app_module.RERANKER_LOADING_STATUS,
    )

    assert status_update == getattr(app_module, expected_status)
    assert timer_update["active"] is False
    assert ranker.state_reads == 1


@pytest.mark.parametrize(
    "message_status",
    [
        "SEARCHING_STATUS",
        "GENERATING_STATUS",
        "STREAMING_STATUS",
        "SAVED_STATUS",
        "ERROR_STATUS",
    ],
)
def test_reranker_status_poll_does_not_mutate_active_message_status(
    message_status: str,
) -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker(state="ready")
    pipeline = SimpleNamespace(ranker=ranker)

    status_update, timer_update = app_module._poll_reranker_status(
        pipeline,
        getattr(app_module, message_status),
    )

    assert status_update == gr.skip()
    assert timer_update["active"] is False
    assert ranker.state_reads == 0


def test_reranker_status_poll_deactivates_safely_for_plain_ranker() -> None:
    import scripts.app as app_module

    status_update, timer_update = app_module._poll_reranker_status(
        SimpleNamespace(ranker=PlainRanker()),
        app_module.RERANKER_LOADING_STATUS,
    )

    assert status_update == gr.skip()
    assert timer_update["active"] is False


def test_handler_yields_before_waiting_and_reports_ready_generating_and_saved(
    tmp_path: Path,
) -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker(state="loading", ready=True)
    pipeline = StreamingPipeline(tmp_path, ranker=ranker)
    handler = app_module._make_message_handler(pipeline, {})
    events = handler("What grows?", None, [], [])

    first = next(events)

    assert first[3] == app_module.RERANKER_LOADING_STATUS
    assert ranker.wait_calls == []
    assert pipeline.calls == []

    updates = [first, *events]
    statuses = [update[3] for update in updates]

    assert ranker.wait_calls == [app_module.RERANKER_READY_TIMEOUT]
    assert app_module.RERANKER_READY_STATUS in statuses
    assert app_module.GENERATING_STATUS in statuses
    assert statuses[-1] == app_module.SAVED_STATUS
    assert pipeline.calls == [(updates[-1][7], "What grows?")]


def test_app_readiness_uses_ranker_load_timeout_source_of_truth() -> None:
    import scripts.app as app_module

    assert app_module.RERANKER_READY_TIMEOUT == app_module.DEFAULT_MODEL_LOAD_TIMEOUT == 120.0


def test_handler_generates_no_answer_when_reranker_unavailable_and_restores_controls(
    tmp_path: Path,
) -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker(state="unavailable", ready=False)
    pipeline = StreamingPipeline(tmp_path, ranker=ranker)
    handler = app_module._make_message_handler(pipeline, {})

    updates = list(handler("What grows?", None, [], []))

    assert [update[3] for update in updates] == [
        app_module.RERANKER_LOADING_STATUS,
        app_module.RERANKER_UNAVAILABLE_STATUS,
    ]
    assert ranker.wait_calls == [app_module.RERANKER_READY_TIMEOUT]
    assert pipeline.calls == []
    assert updates[-1][0] == []
    assert updates[-1][1]["interactive"] is True
    assert updates[-1][2]["interactive"] is True


def test_handler_keeps_transient_loading_status_after_readiness_timeout(
    tmp_path: Path,
) -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker(state="loading", ready=False)
    pipeline = StreamingPipeline(tmp_path, ranker=ranker)
    handler = app_module._make_message_handler(pipeline, {})

    updates = list(handler("What grows?", None, [], []))

    assert [update[3] for update in updates] == [
        app_module.RERANKER_LOADING_STATUS,
        app_module.RERANKER_LOADING_STATUS,
    ]
    assert pipeline.calls == []


def test_handler_reports_terminal_prediction_timeout_as_unavailable(
    tmp_path: Path,
) -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker(state="ready", ready=True)
    pipeline = PredictionFailurePipeline(tmp_path, ranker=ranker)
    handler = app_module._make_message_handler(pipeline, {})

    updates = list(handler("What grows?", None, [], []))

    assert updates[-1][3] == app_module.RERANKER_UNAVAILABLE_STATUS
    assert updates[-1][3] != app_module.ERROR_STATUS
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0][1] == "What grows?"


def test_handler_preserves_plain_ranker_path_without_lifecycle_gate(
    tmp_path: Path,
) -> None:
    import scripts.app as app_module

    ranker = PlainRanker()
    pipeline = StreamingPipeline(tmp_path, ranker=ranker)  # type: ignore[arg-type]
    handler = app_module._make_message_handler(pipeline, {})

    updates = list(handler("What grows?", None, [], []))
    statuses = [update[3] for update in updates]

    assert pipeline.calls == [(updates[-1][7], "What grows?")]
    assert statuses[-1] == app_module.SAVED_STATUS
    assert app_module.RERANKER_UNAVAILABLE_STATUS not in statuses


def test_app_shutdown_closes_optional_ranker_lifecycle(tmp_path: Path) -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker()
    pipeline = StreamingPipeline(tmp_path, ranker=ranker)

    app_module._shutdown_pipeline(pipeline)

    assert ranker.shutdown_calls == 1


def test_default_launch_retains_pipeline_and_shuts_down_on_launch_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker()
    pipeline = StreamingPipeline(tmp_path, ranker=ranker)

    class FailingApp:
        def launch(self) -> None:
            raise RuntimeError("launch failed")

    monkeypatch.setattr(app_module, "build_default_pipeline", lambda: pipeline)
    monkeypatch.setattr(app_module, "create_app", lambda runtime: FailingApp())

    with pytest.raises(RuntimeError, match="launch failed"):
        app_module._launch_default_app()

    assert ranker.shutdown_calls == 1


def test_default_owned_app_exposes_idempotent_programmatic_teardown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import scripts.app as app_module

    ranker = LifecycleRanker()
    pipeline = StreamingPipeline(tmp_path, ranker=ranker)
    monkeypatch.setattr(app_module, "build_default_pipeline", lambda: pipeline)

    app = app_module.create_app()
    teardown = getattr(app, "teardown", None)

    assert callable(teardown)
    teardown()
    teardown()
    assert ranker.shutdown_calls == 1


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
        "workspace-meta",
        "corpus-count",
        "local-status",
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


def test_factory_exposes_hybrid_viewport_layout_contract() -> None:
    import scripts.app as app_module

    app = app_module.create_app(FakePipeline())  # type: ignore[arg-type]
    css = app.get_config_file()["css"]

    assert "height: 100dvh" in css
    assert "min-height: 0" in css
    assert (
        "grid-template-columns: minmax(220px, 17rem) minmax(0, 1fr) "
        "minmax(260px, 22rem)"
    ) in css
    assert re.search(
        r"#agro-rag-app\s+#session-rail[^{}]*\{[^{}]*overflow-y:\s*auto",
        css,
        re.DOTALL,
    )
    assert re.search(
        r"#agro-rag-app\s+#evidence-rail[^{}]*\{[^{}]*overflow-y:\s*auto",
        css,
        re.DOTALL,
    )
    assert "@media (max-width: 1099px)" in css
    assert "max-height: min(" in css
    assert ".message-content" in css
    assert "overflow-x: auto" in css
    assert "min-width: 42rem" in css
    assert "min-height: 680px" not in css
    assert "min-height: 620px" not in css


def test_factory_places_workspace_metadata_in_session_side_region() -> None:
    import scripts.app as app_module

    app = app_module.create_app(PublicApiPipeline())  # type: ignore[arg-type]
    components = app.get_config_file()["components"]
    component_ids = [component["props"].get("elem_id") for component in components]

    def position(element_id: str) -> int:
        return component_ids.index(element_id)

    assert position("agro-header") < position("session-rail")
    assert position("session-rail") < position("workspace-meta")
    assert position("workspace-meta") < position("corpus-count")
    assert position("corpus-count") < position("local-status")
    assert position("local-status") < position("chat-panel")


def test_source_cards_use_labelled_metadata_rows() -> None:
    import scripts.app as app_module

    source = SimpleNamespace(
        document_id="doc-1",
        title="Maize note",
        source="Extension service",
        crop="maize",
        country="Kenya",
        origin="synthetic",
        rank=1,
    )

    rendered = app_module.render_source_cards([source])

    assert '<dl class="source-metadata">' in rendered
    assert "<dt>Crop</dt><dd>maize</dd>" in rendered
    assert "<dt>Country</dt><dd>Kenya</dd>" in rendered
    assert "Crop: maize" not in rendered


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
