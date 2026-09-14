"""Process-isolated local CrossEncoder transport."""

from __future__ import annotations

import math
import multiprocessing as multiprocessing
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any


DEFAULT_MODEL_LOAD_TIMEOUT = 120.0
DEFAULT_PREDICTION_TIMEOUT = 60.0
# Compatibility alias for callers that used the old generic worker timeout.
DEFAULT_WORKER_TIMEOUT = DEFAULT_MODEL_LOAD_TIMEOUT
DEFAULT_CLEANUP_TIMEOUT = 0.5
REPRODUCIBILITY_SEED = 42


def configure_reproducibility(seed: int = REPRODUCIBILITY_SEED) -> None:
    """Set deterministic Python, NumPy, and Torch state inside worker process."""

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RerankerWorkerError(RuntimeError):
    """Sanitized worker transport failure with stable public category."""

    def __init__(self, message: str, *, kind: str = "unavailable") -> None:
        super().__init__(message)
        self.kind = kind


def run_reranker_worker(connection: Any) -> None:
    """Run narrow load/predict/shutdown protocol inside spawned child."""

    configure_reproducibility()
    model: Any | None = None
    while True:
        try:
            command = connection.recv()
        except BaseException:
            return

        operation = _command_operation(command)
        if operation == "load":
            if model is not None or not _valid_load_command(command):
                if not _send_response(connection, ("error", "Reranker request malformed.")):
                    return
                continue
            model_name = command[1]
            try:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(
                    model_name,
                    backend="torch",
                    device="cpu",
                    local_files_only=True,
                )
            except BaseException:
                _send_response(connection, ("error", "Reranker model unavailable locally."))
                return
            if model is None:
                _send_response(connection, ("error", "Reranker model unavailable locally."))
                return
            if not _send_response(connection, ("ready",)):
                return
            continue

        if operation == "predict":
            if model is None or not _valid_predict_command(command):
                if not _send_response(connection, ("error", "Reranker request malformed.")):
                    return
                continue
            try:
                scores = tuple(model.predict(command[1]))
                connection.send(("scores", scores))
            except BaseException:
                _send_response(connection, ("error", "Reranker prediction failed."))
                return
            continue

        if operation == "shutdown":
            if _valid_shutdown_command(command):
                _send_response(connection, ("shutdown",))
                return
            if not _send_response(connection, ("error", "Reranker request malformed.")):
                return
            continue

        if not _send_response(connection, ("error", "Reranker request malformed.")):
            return


class RerankerWorkerClient:
    """Single persistent spawned worker with bounded request/cleanup waits."""

    def __init__(
        self,
        model_name: str,
        *,
        process_factory: Callable[..., Any] | None = None,
        pipe_factory: Callable[..., tuple[Any, Any]] | None = None,
        prediction_timeout: float = DEFAULT_PREDICTION_TIMEOUT,
        cleanup_timeout: float = DEFAULT_CLEANUP_TIMEOUT,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name.strip()
        self.prediction_timeout = _positive_finite_timeout(
            prediction_timeout, "prediction_timeout"
        )
        self.cleanup_timeout = _positive_finite_timeout(cleanup_timeout, "cleanup_timeout")
        context = multiprocessing.get_context("spawn")
        self._process_factory = process_factory or context.Process
        self._pipe_factory = pipe_factory or context.Pipe
        self._connection: Any | None = None
        self._process: Any | None = None
        self._ready = False
        self._io_lock = threading.RLock()
        self._cleanup_error: str | None = None
        self._start_state_lock = threading.Lock()
        self._start_done: threading.Event | None = None
        self._start_outcome: dict[str, object] | None = None
        self._start_process: Any | None = None
        self._start_cancelled = False
        self._start_supervisor: threading.Thread | None = None
        self._send_state_lock = threading.Lock()
        self._send_thread: threading.Thread | None = None
        self._send_done: threading.Event | None = None
        self._send_outcome: dict[str, object] | None = None

    @property
    def cleanup_error(self) -> str | None:
        """Return sanitized cleanup diagnostic, if last cleanup was incomplete."""

        return self._cleanup_error

    @property
    def ready(self) -> bool:
        if not self._io_lock.acquire(blocking=False):
            # A prediction owns the transport lock. Preserve its last known state
            # rather than reporting a false dead-child result to health checks.
            return self._ready and self._process is not None and self._connection is not None
        try:
            return self._ready_locked()
        finally:
            self._io_lock.release()

    def start(self, timeout: float = DEFAULT_MODEL_LOAD_TIMEOUT) -> None:
        """Spawn worker, issue load command, and wait boundedly for ready."""

        timeout = _positive_finite_timeout(timeout, "timeout")
        deadline = time.monotonic() + timeout
        self._acquire_io_lock(deadline)
        child_connection: Any | None = None
        try:
            if self._ready_locked():
                return
            if not self._cleanup_locked():
                cleanup_kind = "cleanup" if self._cleanup_error is not None else "busy"
                cleanup_message = (
                    "Reranker worker cleanup failed."
                    if cleanup_kind == "cleanup"
                    else "Reranker worker cleanup is still pending."
                )
                raise RerankerWorkerError(
                    cleanup_message,
                    kind=cleanup_kind,
                )
            try:
                parent_connection, child_connection = self._pipe_factory(duplex=True)
                self._connection = parent_connection
                self._process = self._process_factory(
                    target=run_reranker_worker,
                    args=(child_connection,),
                    daemon=True,
                )
                process = self._process
                if process is None:
                    raise RuntimeError("missing worker process")
                self._launch_process_start_locked(process, child_connection)
                child_connection = None
                if not self._wait_for_process_start_locked(deadline):
                    if self._cancel_process_start_locked(process):
                        raise RerankerWorkerError(
                            "Reranker worker start timed out.",
                            kind="start_timeout",
                        )

                start_error = self._process_start_error_locked(process)
                self._clear_start_state_locked(process)
                if start_error is not None:
                    self._cleanup_locked()
                    raise RerankerWorkerError(
                        "Reranker worker could not start.",
                        kind="unavailable",
                    ) from None
                self._send_locked(
                    ("load", self.model_name),
                    deadline=deadline,
                    phase="load",
                )
                response = self._receive_locked(deadline, phase="readiness")
                if not _is_response(response, "ready"):
                    self._raise_response_error(response, phase="readiness")
                self._ready = True
                self._cleanup_error = None
            except RerankerWorkerError as error:
                if error.kind != "start_timeout":
                    self._cleanup_locked()
                raise
            except BaseException:
                self._cleanup_locked()
                raise RerankerWorkerError(
                    "Reranker worker could not start.",
                    kind="unavailable",
                ) from None
        finally:
            _close_safely(child_connection)
            self._io_lock.release()

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        timeout: float | None = None,
    ) -> tuple[Any, ...]:
        """Send one prediction request and enforce finite response deadline."""

        request_timeout = self.prediction_timeout if timeout is None else _positive_finite_timeout(
            timeout, "timeout"
        )
        try:
            payload = [(query, document) for query, document in pairs]
        except BaseException:
            raise RerankerWorkerError(
                "Reranker prediction request was invalid.",
                kind="protocol",
            ) from None
        deadline = time.monotonic() + request_timeout
        self._acquire_io_lock(deadline)
        try:
            self._require_live_worker_locked()
            try:
                self._send_locked(
                    ("predict", payload),
                    deadline=deadline,
                    phase="prediction",
                )
                response = self._receive_locked(deadline, phase="prediction")
                if (
                    isinstance(response, tuple)
                    and len(response) == 2
                    and response[0] == "scores"
                    and _valid_scores_payload(response[1])
                ):
                    return tuple(response[1])
                self._raise_response_error(response, phase="prediction")
                raise AssertionError("unreachable")
            except RerankerWorkerError:
                self._cleanup_locked()
                raise
            except BaseException:
                self._cleanup_locked()
                raise RerankerWorkerError(
                    "Reranker worker communication failed.",
                    kind="prediction",
                ) from None
        finally:
            self._io_lock.release()

    def shutdown(self, timeout: float = DEFAULT_CLEANUP_TIMEOUT) -> None:
        """Stop worker and close transport; safe to call repeatedly."""

        timeout = _positive_finite_timeout(timeout, "timeout")
        deadline = time.monotonic() + timeout
        self._acquire_io_lock(deadline)
        try:
            if self._cancel_pending_process_start_locked():
                self._cleanup_locked()
                return
            shutdown_error: RerankerWorkerError | None = None
            if self._connection is not None and self._process is not None and self._ready:
                try:
                    self._send_locked(
                        ("shutdown",),
                        deadline=deadline,
                        phase="shutdown",
                    )
                    self._receive_locked(
                        deadline,
                        phase="shutdown",
                    )
                except RerankerWorkerError as error:
                    if error.kind == "send_timeout":
                        shutdown_error = error
                except BaseException:
                    shutdown_error = RerankerWorkerError(
                        "Reranker worker shutdown failed.",
                        kind="shutdown",
                    )
            self._cleanup_locked()
            if shutdown_error is not None:
                raise shutdown_error
        finally:
            self._io_lock.release()

    def _acquire_io_lock(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RerankerWorkerError(
                "Reranker worker is busy.",
                kind="busy",
            )
        try:
            acquired = self._io_lock.acquire(
                timeout=min(remaining, getattr(threading, "TIMEOUT_MAX", remaining))
            )
        except (OverflowError, ValueError):
            acquired = self._io_lock.acquire(blocking=False)
        if not acquired:
            raise RerankerWorkerError(
                "Reranker worker is busy.",
                kind="busy",
            )

    def _ready_locked(self) -> bool:
        if not self._ready or self._process is None or self._connection is None:
            return False
        alive, inspection_failed, _ = _alive_status(self._process)
        if inspection_failed or not alive:
            self._cleanup_locked()
            return False
        return True

    def _require_live_worker_locked(self) -> None:
        if not self._ready or self._connection is None or self._process is None:
            raise RerankerWorkerError(
                "Reranker worker is unavailable.",
                kind="unavailable",
            )
        alive, inspection_failed, _ = _alive_status(self._process)
        if inspection_failed or not alive:
            self._cleanup_locked()
            raise RerankerWorkerError(
                "Reranker worker exited unexpectedly.",
                kind="unavailable",
            )

    def _launch_process_start_locked(self, process: Any, child_connection: Any) -> None:
        done = threading.Event()
        outcome: dict[str, object] = {}
        with self._start_state_lock:
            self._start_done = done
            self._start_outcome = outcome
            self._start_process = process
            self._start_cancelled = False

        supervisor = threading.Thread(
            target=self._supervise_process_start,
            args=(process, child_connection, done, outcome),
            name="reranker-process-start",
            daemon=True,
        )
        with self._start_state_lock:
            self._start_supervisor = supervisor
        try:
            supervisor.start()
        except BaseException as exc:
            with self._start_state_lock:
                outcome["error"] = exc
                done.set()
                self._start_supervisor = None
            _close_safely(child_connection)

    def _supervise_process_start(
        self,
        process: Any,
        child_connection: Any,
        done: threading.Event,
        outcome: dict[str, object],
    ) -> None:
        try:
            process.start()
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            _close_safely(child_connection)
            with self._start_state_lock:
                done.set()
                cancelled = self._start_cancelled
        if cancelled:
            self._cleanup_after_process_start(process)

    def _wait_for_process_start_locked(self, deadline: float) -> bool:
        with self._start_state_lock:
            done = self._start_done
        if done is None:
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        return done.wait(timeout=min(remaining, getattr(threading, "TIMEOUT_MAX", remaining)))

    def _cancel_process_start_locked(self, process: Any) -> bool:
        with self._start_state_lock:
            if self._start_process is not process or self._start_done is None:
                return False
            if self._start_done.is_set():
                return False
            self._start_cancelled = True
            return True

    def _cancel_pending_process_start_locked(self) -> bool:
        with self._start_state_lock:
            if self._start_process is None or self._start_done is None:
                return False
            if self._start_done.is_set():
                return False
            self._start_cancelled = True
            return True

    def _process_start_error_locked(self, process: Any) -> BaseException | None:
        with self._start_state_lock:
            if self._start_process is not process or self._start_outcome is None:
                return RuntimeError("missing worker start result")
            error = self._start_outcome.get("error")
        return error if isinstance(error, BaseException) else None

    def _clear_start_state_locked(self, process: Any) -> None:
        with self._start_state_lock:
            if self._start_process is not process:
                return
            self._start_done = None
            self._start_outcome = None
            self._start_process = None
            self._start_cancelled = False
            self._start_supervisor = None

    def _cleanup_after_process_start(self, process: Any) -> None:
        with self._io_lock:
            if self._process is process:
                self._cleanup_locked()

    def _send_locked(self, command: object, *, deadline: float, phase: str) -> None:
        if self._connection is None:
            raise RerankerWorkerError("Reranker worker communication failed.", kind="transport")
        self._reap_sender_locked(wait=False)
        with self._send_state_lock:
            if self._send_thread is not None:
                raise RerankerWorkerError(
                    "Reranker worker send is still active.",
                    kind="busy",
                )
            done = threading.Event()
            outcome: dict[str, object] = {}
            sender = threading.Thread(
                target=self._send_command,
                args=(self._connection, command, done, outcome),
                name="reranker-pipe-send",
                daemon=True,
            )
            self._send_thread = sender
            self._send_done = done
            self._send_outcome = outcome
        try:
            sender.start()
        except BaseException:
            self._clear_sender_locked(sender)
            self._cleanup_locked()
            raise RerankerWorkerError(
                "Reranker worker send failed.",
                kind="transport",
            ) from None

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not done.wait(
            timeout=min(remaining, getattr(threading, "TIMEOUT_MAX", remaining))
        ):
            self._cleanup_locked()
            raise RerankerWorkerError(
                "Reranker worker send timed out.",
                kind="send_timeout",
            ) from None

        send_error = outcome.get("error")
        self._reap_sender_locked(wait=True)
        if isinstance(send_error, BaseException):
            self._cleanup_locked()
            raise RerankerWorkerError(
                "Reranker worker communication failed.",
                kind="transport",
            ) from None

    @staticmethod
    def _send_command(
        connection: Any,
        command: object,
        done: threading.Event,
        outcome: dict[str, object],
    ) -> None:
        try:
            connection.send(command)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    def _clear_sender_locked(self, sender: threading.Thread) -> None:
        with self._send_state_lock:
            if self._send_thread is not sender:
                return
            self._send_thread = None
            self._send_done = None
            self._send_outcome = None

    def _reap_sender_locked(self, *, wait: bool) -> bool:
        with self._send_state_lock:
            sender = self._send_thread
            done = self._send_done
        if sender is None:
            return True
        if wait:
            join_error = _join_thread_safely(sender, self.cleanup_timeout)
            if join_error is not None:
                return False
        elif done is None or not done.is_set():
            return False
        try:
            sender_alive = sender.is_alive()
        except BaseException:
            return False
        if sender_alive:
            return False
        self._clear_sender_locked(sender)
        return True

    def _raise_send_deadline(self, phase: str) -> None:
        raise RerankerWorkerError(
            "Reranker worker send timed out.",
            kind="send_timeout",
        ) from None

    def _receive_locked(self, deadline: float, *, phase: str) -> object:
        if self._connection is None or self._process is None:
            raise RerankerWorkerError("Reranker worker is unavailable.", kind="unavailable")
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._raise_deadline(phase)
                alive, inspection_failed, _ = _alive_status(self._process)
                if inspection_failed or not alive:
                    raise RerankerWorkerError(
                        "Reranker worker exited unexpectedly.",
                        kind="unavailable",
                    )
                poll_timeout = min(remaining, getattr(threading, "TIMEOUT_MAX", remaining))
                if self._connection.poll(poll_timeout):
                    break
            response = self._connection.recv()
        except RerankerWorkerError:
            raise
        except BaseException:
            raise RerankerWorkerError(
                "Reranker worker communication failed.",
                kind="transport",
            ) from None
        return response

    @staticmethod
    def _raise_response_error(response: object, *, phase: str) -> None:
        if _is_response(response, "error"):
            if phase == "readiness":
                raise RerankerWorkerError(
                    "Reranker model unavailable locally.",
                    kind="unavailable",
                )
            if phase == "prediction":
                raise RerankerWorkerError(
                    "Reranker prediction failed.",
                    kind="prediction",
                )
            return
        raise RerankerWorkerError(
            "Reranker worker returned malformed response.",
            kind="protocol",
        )

    def _cleanup_locked(self) -> bool:
        self._ready = False
        process = self._process
        if process is not None and self._start_is_pending():
            self._cancel_pending_process_start_locked()
            connection_error = _call_safely(self._connection, "close")
            if connection_error is None:
                self._connection = None
            else:
                self._cleanup_error = "Reranker worker cleanup failed."
            if not self._reap_sender_locked(wait=True):
                self._cleanup_error = "Reranker worker cleanup failed."
            return False

        connection = self._connection
        connection_error = _call_safely(connection, "close")
        if connection_error is None:
            self._connection = None
        else:
            self._cleanup_error = "Reranker worker cleanup failed."
        if process is None:
            sender_ok = self._reap_sender_locked(wait=True)
            if connection_error is None and sender_ok:
                self._cleanup_error = None
            elif not sender_ok:
                self._cleanup_error = "Reranker worker cleanup failed."
            return connection_error is None and sender_ok

        cleanup_failed = connection_error is not None
        alive, inspection_failed, unstarted = _alive_status(process)
        cleanup_failed |= inspection_failed
        if alive:
            cleanup_failed |= _call_safely(process, "terminate") is not None
            cleanup_failed |= _join_safely(process, self.cleanup_timeout) is not None
            alive, inspection_failed, unstarted = _alive_status(process)
            cleanup_failed |= inspection_failed
            if alive:
                kill_method = _method_safely(process, "kill")
                if kill_method is not None:
                    cleanup_failed |= _call_safely(process, "kill") is not None
                else:
                    cleanup_failed |= _call_safely(process, "terminate") is not None
                cleanup_failed |= _join_safely(process, self.cleanup_timeout) is not None
                alive, inspection_failed, unstarted = _alive_status(process)
                cleanup_failed |= inspection_failed
        elif not unstarted:
            cleanup_failed |= _join_safely(process, self.cleanup_timeout) is not None

        sender_ok = self._reap_sender_locked(wait=True)
        cleanup_failed |= not sender_ok
        if alive:
            self._cleanup_error = "Reranker worker cleanup failed."
            return False

        cleanup_failed |= _call_safely(process, "close") is not None
        if cleanup_failed:
            self._cleanup_error = "Reranker worker cleanup failed."
            return False

        self._process = None
        self._cleanup_error = None
        self._clear_start_state_locked(process)
        return True

    def _start_is_pending(self) -> bool:
        with self._start_state_lock:
            return self._start_done is not None and not self._start_done.is_set()

    def _raise_deadline(self, phase: str) -> None:
        if phase == "readiness":
            raise RerankerWorkerError(
                "Reranker worker readiness timed out.",
                kind="readiness_timeout",
            )
        if phase == "prediction":
            raise RerankerWorkerError(
                "Reranker prediction timed out.",
                kind="prediction_timeout",
            )
        raise RerankerWorkerError("Reranker worker shutdown timed out.", kind="shutdown")


def _command_operation(command: object) -> str | None:
    if isinstance(command, tuple) and command and isinstance(command[0], str):
        return command[0]
    return None


def _valid_load_command(command: object) -> bool:
    return (
        isinstance(command, tuple)
        and len(command) == 2
        and isinstance(command[1], str)
        and bool(command[1].strip())
    )


def _valid_predict_command(command: object) -> bool:
    if not isinstance(command, tuple) or len(command) != 2:
        return False
    pairs = command[1]
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        return False
    return all(
        isinstance(pair, (tuple, list))
        and len(pair) == 2
        and all(isinstance(part, str) for part in pair)
        for pair in pairs
    )


def _valid_shutdown_command(command: object) -> bool:
    return isinstance(command, tuple) and len(command) == 1


def _valid_scores_payload(value: object) -> bool:
    return isinstance(value, (tuple, list))


def _is_response(response: object, operation: str) -> bool:
    expected_length = 1 if operation in {"ready", "shutdown"} else 2
    return (
        isinstance(response, tuple)
        and len(response) == expected_length
        and response[0] == operation
    )


def _send_response(connection: Any, response: object) -> bool:
    try:
        connection.send(response)
    except BaseException:
        return False
    return True


def _close_safely(resource: Any | None) -> None:
    _call_safely(resource, "close")


def _call_safely(
    resource: Any | None,
    method_name: str,
    *args: object,
    **kwargs: object,
) -> BaseException | None:
    if resource is None:
        return None
    try:
        method = getattr(resource, method_name, None)
        if callable(method):
            method(*args, **kwargs)
    except BaseException as exc:
        return exc
    return None


def _method_safely(resource: Any | None, method_name: str) -> Any | None:
    if resource is None:
        return None
    try:
        method = getattr(resource, method_name, None)
    except BaseException:
        return None
    return method if callable(method) else None


def _alive_status(process: Any) -> tuple[bool, bool, bool]:
    try:
        return bool(process.is_alive()), False, False
    except AssertionError:
        # multiprocessing.Process uses AssertionError for an unstarted handle;
        # no child can be alive in that state, so it is safe to close.
        return False, False, True
    except BaseException:
        return True, True, False


def _is_alive_safely(process: Any, *, default: bool) -> bool:
    alive, failed, _ = _alive_status(process)
    return default if failed else alive


def _join_safely(process: Any, timeout: float) -> BaseException | None:
    return _call_safely(process, "join", timeout=timeout)


def _join_thread_safely(thread: threading.Thread, timeout: float) -> BaseException | None:
    try:
        thread.join(timeout=timeout)
    except BaseException as exc:
        return exc
    return None


def _positive_finite_timeout(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if (
        not math.isfinite(numeric)
        or numeric <= 0
        or numeric > threading.TIMEOUT_MAX
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return numeric
