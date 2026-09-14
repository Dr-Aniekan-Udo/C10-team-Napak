from __future__ import annotations

import sys
import threading
import time
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from inspect import signature

import pytest

from scripts.utilities.reranker_worker import (
    DEFAULT_MODEL_LOAD_TIMEOUT,
    DEFAULT_PREDICTION_TIMEOUT,
    RerankerWorkerClient,
    RerankerWorkerError,
    run_reranker_worker,
)


class FakeChildConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, *, alive: bool = True, start_error: BaseException | None = None) -> None:
        self.alive = alive
        self.start_error = start_error
        self.started = False
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls: list[float | None] = []
        self.close_calls = 0
        self.target: object | None = None
        self.args: tuple[object, ...] | None = None

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def close(self) -> None:
        self.close_calls += 1


class FakeConnection:
    def __init__(self, process: FakeProcess, *, responses: list[object] | None = None) -> None:
        self.process = process
        self.responses = deque(responses or [])
        self.sent: list[object] = []
        self.poll_calls: list[float] = []
        self.closed = False
        self.on_send = None
        self.poll_result = False
        self.block_operation: str | None = None
        self.send_entered = threading.Event()
        self.send_release = threading.Event()

    def send(self, command: object) -> None:
        if self.block_operation == command[0]:  # type: ignore[index]
            self.send_entered.set()
            self.send_release.wait(1.0)
        self.sent.append(command)
        if self.on_send is not None:
            self.on_send(self, command)

    def poll(self, timeout: float) -> bool:
        self.poll_calls.append(timeout)
        return bool(self.responses) or self.poll_result

    def recv(self) -> object:
        if not self.responses:
            raise EOFError("pipe closed")
        return self.responses.popleft()

    def close(self) -> None:
        self.closed = True


class WorkerHarness:
    def __init__(self, *, process: FakeProcess | None = None) -> None:
        self.process = process or FakeProcess()
        self.connection = FakeConnection(self.process)
        self.child = FakeChildConnection()
        self.processes_created = 0

        def on_send(connection: FakeConnection, command: object) -> None:
            operation = command[0]  # type: ignore[index]
            if operation == "load":
                connection.responses.append(("ready",))
            elif operation == "predict":
                connection.responses.append(("scores", (0.8, 0.2)))
            elif operation == "shutdown":
                connection.responses.append(("shutdown",))
                self.process.alive = False

        self.connection.on_send = on_send

    def pipe(self, *, duplex: bool) -> tuple[FakeConnection, FakeChildConnection]:
        assert duplex is True
        return self.connection, self.child

    def process_factory(self, *, target: object, args: tuple[object, ...], daemon: bool) -> FakeProcess:
        assert daemon is True
        self.processes_created += 1
        self.process.target = target
        self.process.args = args
        return self.process


def client(harness: WorkerHarness, **kwargs: object) -> RerankerWorkerClient:
    return RerankerWorkerClient(
        "local-cache-model",
        pipe_factory=harness.pipe,
        process_factory=harness.process_factory,
        cleanup_timeout=0.01,
        **kwargs,
    )


def test_client_uses_narrow_protocol_and_reuses_one_process() -> None:
    harness = WorkerHarness()
    worker = client(harness)

    worker.start(0.1)
    worker.start(0.1)
    assert worker.predict([("query", "document")], timeout=0.1) == (0.8, 0.2)
    worker.shutdown()

    assert harness.processes_created == 1
    assert harness.connection.sent == [
        ("load", "local-cache-model"),
        ("predict", [("query", "document")]),
        ("shutdown",),
    ]
    assert harness.connection.closed is True
    assert harness.child.closed is True
    assert harness.process.join_calls
    assert harness.process.close_calls == 1


def test_worker_defaults_use_slow_machine_load_and_bounded_prediction_windows() -> None:
    worker = client(WorkerHarness())

    assert DEFAULT_MODEL_LOAD_TIMEOUT == 120.0
    assert DEFAULT_PREDICTION_TIMEOUT == 60.0
    assert signature(RerankerWorkerClient.start).parameters["timeout"].default == (
        DEFAULT_MODEL_LOAD_TIMEOUT
    )
    assert worker.prediction_timeout == DEFAULT_PREDICTION_TIMEOUT


@pytest.mark.parametrize("operation", ["load", "predict", "shutdown"])
def test_blocked_pipe_sends_fail_boundedly_and_cleanup_ownership_is_retained(
    operation: str,
) -> None:
    harness = WorkerHarness()
    worker = client(harness)
    if operation != "load":
        worker.start(0.1)
    harness.connection.block_operation = operation

    if operation == "load":
        call = lambda: worker.start(0.01)
    elif operation == "predict":
        call = lambda: worker.predict([("query", "document")], timeout=0.01)
    else:
        call = lambda: worker.shutdown(0.01)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(call)
    assert harness.connection.send_entered.wait(1.0)
    try:
        with pytest.raises(RerankerWorkerError, match="send timed out"):
            future.result(timeout=0.2)
    finally:
        harness.connection.send_release.set()
        try:
            future.result(timeout=1.0)
        except BaseException:
            pass
        executor.shutdown(wait=True)

    worker.shutdown()
    assert harness.connection.closed is True
    assert worker._send_thread is None


@pytest.mark.parametrize("operation", ["start", "predict", "shutdown"])
def test_client_operations_fail_boundedly_while_io_lock_is_busy(operation: str) -> None:
    harness = WorkerHarness()
    worker = client(harness)
    worker._io_lock.acquire()
    if operation == "start":
        call = lambda: worker.start(0.01)
    elif operation == "predict":
        call = lambda: worker.predict([("query", "document")], timeout=0.01)
    else:
        call = lambda: worker.shutdown(0.01)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call)
            with pytest.raises(RerankerWorkerError, match="busy"):
                future.result(timeout=0.2)
            worker._io_lock.release()
            try:
                future.result(timeout=1.0)
            except BaseException:
                pass
    finally:
        if worker._io_lock._is_owned():  # type: ignore[attr-defined]
            worker._io_lock.release()


def test_process_start_timeout_keeps_supervisor_ownership_until_cleanup() -> None:
    class BlockingStartProcess(FakeProcess):
        def __init__(self) -> None:
            super().__init__()
            self.start_entered = threading.Event()
            self.release_start = threading.Event()
            self.terminated = threading.Event()

        def start(self) -> None:
            self.start_entered.set()
            assert self.release_start.wait(1.0)
            self.started = True

        def terminate(self) -> None:
            super().terminate()
            self.terminated.set()

    process = BlockingStartProcess()
    harness = WorkerHarness(process=process)
    worker = client(harness)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker.start, 0.01)
        assert process.start_entered.wait(1.0)
        with pytest.raises(RerankerWorkerError, match="start timed out"):
            future.result(timeout=0.2)
        assert worker._process is process
        process.release_start.set()
        assert process.terminated.wait(1.0)

    worker.shutdown()


def test_cleanup_failure_retains_live_process_and_reports_safely() -> None:
    class NeverDiesProcess(FakeProcess):
        def kill(self) -> None:
            self.kill_calls += 1
            raise RuntimeError("kill-secret")

    process = NeverDiesProcess()
    harness = WorkerHarness(process=process)
    worker = client(harness)
    worker.start(0.1)
    harness.connection.responses.clear()
    harness.connection.on_send = lambda connection, command: None

    with pytest.raises(RerankerWorkerError, match="prediction timed out"):
        worker.predict([("query", "document")], timeout=0.05)

    assert worker.cleanup_error is not None
    assert process is worker._process
    process.alive = False
    worker.shutdown()


def test_readiness_timeout_cleans_up_boundedly() -> None:
    harness = WorkerHarness()
    harness.connection.on_send = lambda connection, command: None
    worker = client(harness)

    with pytest.raises(RerankerWorkerError, match="readiness timed out"):
        worker.start(0.05)

    assert harness.process.terminate_calls == 1
    assert harness.process.kill_calls == 1
    assert len(harness.process.join_calls) == 2
    assert harness.connection.closed is True


def test_process_start_failure_is_sanitized_and_cleaned() -> None:
    harness = WorkerHarness(process=FakeProcess(start_error=RuntimeError("secret")))
    worker = client(harness)

    with pytest.raises(RerankerWorkerError, match="could not start") as error:
        worker.start(0.1)

    assert "secret" not in str(error.value)
    assert harness.connection.closed is True
    assert harness.child.closed is True


@pytest.mark.parametrize(
    "response",
    [
        ("unexpected",),
        ("scores",),
        ("scores", "not-a-sequence"),
    ],
)
def test_malformed_worker_responses_are_sanitized(response: object) -> None:
    harness = WorkerHarness()
    harness.connection.responses.append(response)
    worker = client(harness)

    with pytest.raises(RerankerWorkerError, match="malformed"):
        worker.start(0.1)

    assert harness.connection.closed is True


def test_worker_early_exit_and_broken_pipe_are_terminal() -> None:
    exited = WorkerHarness(process=FakeProcess(alive=False))
    exited.connection.on_send = None
    with pytest.raises(RerankerWorkerError, match="exited"):
        client(exited).start(0.1)

    broken = WorkerHarness()

    def fail_send(command: object) -> None:
        raise BrokenPipeError("secret-pipe")

    broken.connection.send = fail_send  # type: ignore[method-assign]
    worker = client(broken)
    with pytest.raises(RerankerWorkerError, match="communication failed") as error:
        worker.start(0.1)
    assert "secret-pipe" not in str(error.value)


def test_prediction_timeout_terminates_and_kills_worker() -> None:
    harness = WorkerHarness()
    worker = client(harness)
    worker.start(0.1)
    harness.connection.responses.clear()
    harness.connection.on_send = lambda connection, command: None

    with pytest.raises(RerankerWorkerError, match="prediction timed out"):
        worker.predict([("query", "document")], timeout=0.05)

    assert harness.process.terminate_calls == 1
    assert harness.process.kill_calls == 1
    assert len(harness.process.join_calls) == 2
    assert harness.connection.closed is True
    assert worker.cleanup_error is None


def test_ready_detects_idle_dead_worker() -> None:
    harness = WorkerHarness()
    worker = client(harness)
    worker.start(0.1)
    harness.process.alive = False

    assert worker.ready is False
    assert harness.connection.closed is True


def test_unstarted_process_cleanup_closes_without_joining_or_retaining_handle() -> None:
    class UnstartedProcess:
        def __init__(self) -> None:
            self.closed = False
            self.join_calls = 0

        def start(self) -> None:
            raise RuntimeError("start-secret")

        def is_alive(self) -> bool:
            raise AssertionError("can only test a started process")

        def join(self, timeout: float | None = None) -> None:
            self.join_calls += 1
            raise AssertionError("join must not be called")

        def close(self) -> None:
            self.closed = True

    process = UnstartedProcess()
    harness = WorkerHarness(process=process)  # type: ignore[arg-type]
    worker = client(harness)

    with pytest.raises(RerankerWorkerError, match="could not start"):
        worker.start(0.1)

    assert process.join_calls == 0
    assert process.closed is True
    assert worker._process is None
    assert worker.cleanup_error is None


def test_concurrent_predictions_are_serialized() -> None:
    harness = WorkerHarness()
    worker = client(harness)
    worker.start(0.1)
    active = 0
    maximum = 0
    guard = threading.Lock()

    def on_send(connection: FakeConnection, command: object) -> None:
        nonlocal active, maximum
        operation = command[0]  # type: ignore[index]
        if operation == "predict":
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            connection.responses.append(("scores", (0.8, 0.2)))
            with guard:
                active -= 1

    harness.connection.on_send = on_send
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(worker.predict, [("query", "document")], timeout=0.2)
            for _ in range(2)
        ]
        assert [future.result(timeout=1.0) for future in futures] == [
            (0.8, 0.2),
            (0.8, 0.2),
        ]

    assert maximum == 1


def test_worker_protocol_loads_local_torch_model_and_handles_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeModel:
        def predict(self, pairs: object) -> tuple[float, float]:
            return 0.8, 0.2

    def load(model_id: str, **kwargs: object) -> FakeModel:
        calls.append((model_id, kwargs))
        return FakeModel()

    class ProtocolConnection:
        def __init__(self) -> None:
            self.commands = deque(
                [
                    ("load", "local-cache-model"),
                    ("predict", (("query", "document"),)),
                    ("shutdown",),
                ]
            )
            self.responses: list[object] = []

        def recv(self) -> object:
            return self.commands.popleft()

        def send(self, response: object) -> None:
            self.responses.append(response)

    connection = ProtocolConnection()
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=load))

    run_reranker_worker(connection)

    assert calls == [
        (
            "local-cache-model",
            {"backend": "torch", "device": "cpu", "local_files_only": True},
        )
    ]
    assert connection.responses == [
        ("ready",),
        ("scores", (0.8, 0.2)),
        ("shutdown",),
    ]


def test_worker_base_exception_is_generic_and_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    class FatalWorker(BaseException):
        pass

    def load(model_id: str, **kwargs: object) -> object:
        raise FatalWorker("prompt-secret")

    class ProtocolConnection:
        def __init__(self) -> None:
            self.responses: list[object] = []
            self.commands = deque([("load", "local-cache-model")])

        def recv(self) -> object:
            return self.commands.popleft()

        def send(self, response: object) -> None:
            self.responses.append(response)

    connection = ProtocolConnection()
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=load))

    run_reranker_worker(connection)

    assert connection.responses == [("error", "Reranker model unavailable locally.")]


def test_reproducibility_helper_seeds_all_backends_and_sets_cudnn_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    from scripts.utilities import reranker_worker as worker_module

    random_seeds: list[int] = []
    numpy_seeds: list[int] = []
    torch_seeds: list[int] = []

    class FakeNumpy:
        class random:
            @staticmethod
            def seed(seed: int) -> None:
                numpy_seeds.append(seed)

    class FakeCudnn:
        deterministic = False
        benchmark = True

    class FakeTorch:
        backends = types.SimpleNamespace(cudnn=FakeCudnn())

        @staticmethod
        def manual_seed(seed: int) -> None:
            torch_seeds.append(seed)

    monkeypatch.setattr(random, "seed", random_seeds.append)
    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    worker_module.configure_reproducibility(123)

    assert random_seeds == [123]
    assert numpy_seeds == [123]
    assert torch_seeds == [123]
    assert FakeTorch.backends.cudnn.deterministic is True
    assert FakeTorch.backends.cudnn.benchmark is False


def test_worker_prediction_base_exception_is_generic_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FatalPrediction(BaseException):
        pass

    class FatalModel:
        def predict(self, pairs: object) -> tuple[float, ...]:
            raise FatalPrediction("prediction-secret")

    def load(model_id: str, **kwargs: object) -> FatalModel:
        return FatalModel()

    class ProtocolConnection:
        def __init__(self) -> None:
            self.responses: list[object] = []
            self.commands = deque(
                [
                    ("load", "local-cache-model"),
                    ("predict", (("query", "document"),)),
                    ("shutdown",),
                ]
            )

        def recv(self) -> object:
            return self.commands.popleft()

        def send(self, response: object) -> None:
            self.responses.append(response)

    connection = ProtocolConnection()
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=load))

    run_reranker_worker(connection)

    assert connection.responses == [
        ("ready",),
        ("error", "Reranker prediction failed."),
    ]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, 10**400])
def test_worker_rejects_non_positive_non_finite_or_huge_timeouts(timeout: object) -> None:
    harness = WorkerHarness()
    worker = client(harness)

    with pytest.raises(ValueError, match="timeout"):
        worker.start(timeout)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout", [threading.TIMEOUT_MAX + 1, 1e308, 10**400])
def test_worker_rejects_timeouts_above_threading_max(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout"):
        client(WorkerHarness(), prediction_timeout=timeout)
    with pytest.raises(ValueError, match="cleanup_timeout"):
        RerankerWorkerClient(
            "local-cache-model",
            pipe_factory=WorkerHarness().pipe,
            process_factory=WorkerHarness().process_factory,
            cleanup_timeout=timeout,  # type: ignore[arg-type]
        )
