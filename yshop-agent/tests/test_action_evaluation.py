"""Fixed action matrix validation and honest, exclusive evaluation reporting."""

import importlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.config import SERVICE_ENV_FILE


def runner():
    assert importlib.util.find_spec("scripts.eval_actions") is not None, "action evaluation runner is missing"
    return importlib.import_module("scripts.eval_actions")


def matrix(path, **updates):
    value = {"schema_version": 1, "scope": "deterministic_no_live_requests",
             "cases": [{"id": "one", "group": "history",
                        "selector": "tests/test_action_history.py::test_history_reopens_and_terminal_replay_does_not_append"}]}
    value.update(updates)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("cases", [
    [], [{"id": "one", "group": "history", "selector": "--disable-safety"}],
    [{"id": "one", "group": "history", "selector": "../test_secret.py::test_case"}],
    [{"id": "one", "group": "history", "selector": "tests/test_missing.py::test_case"}],
    [{"id": "one", "group": "history", "selector": "tests/test_actions.py"}],
    [{"id": "one", "group": "history", "selector": "tests/test_actions.py::test_action_modules_exist"}] * 2,
])
def test_matrix_rejects_empty_duplicate_and_unsafe_selectors(tmp_path, cases):
    module = runner()
    with pytest.raises(ValueError):
        module.load_cases(matrix(tmp_path / "cases.json", cases=cases))


@pytest.mark.parametrize("result,expected", [
    ("", "passed"), ("<failure message='private'/>", "failed"),
    ("<error message='private'/>", "failed"), ("<skipped/>", "skipped"),
])
def test_junit_case_results_do_not_mislabel_failures_or_skips(tmp_path, result, expected):
    module = runner()
    cases = module.load_cases(matrix(tmp_path / "cases.json"))
    xml = tmp_path / "junit.xml"
    xml.write_text("<testsuites><testsuite><testcase classname='tests.test_action_history' "
                   "name='test_history_reopens_and_terminal_replay_does_not_append' time='0.1'>"
                   + result + "</testcase></testsuite></testsuites>", encoding="utf-8")
    records = module.case_results(xml, cases)
    assert records[0]["status"] == expected and records[0]["test_count"] == 1
    assert "private" not in json.dumps(records)


def test_missing_case_never_counts_as_passed(tmp_path):
    module = runner()
    cases = module.load_cases(matrix(tmp_path / "cases.json"))
    xml = tmp_path / "junit.xml"
    xml.write_text("<testsuites/>", encoding="utf-8")
    assert module.case_results(xml, cases)[0]["status"] == "not_run"


def test_existing_report_is_not_overwritten_or_executed(monkeypatch):
    module = runner()
    directory = SERVICE_ENV_FILE.parent / ".tmp" / f"day7-eval-test-{uuid4().hex}"
    directory.mkdir(parents=True)
    report = directory / "existing.json"
    report.write_text("protected", encoding="utf-8")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: calls.append(True))
    assert module.main(["--report", str(report)], stdout=io.StringIO()) == 1
    assert report.read_text(encoding="utf-8") == "protected" and calls == []


def test_subprocess_failure_retains_sanitized_nonpassing_report(monkeypatch):
    module = runner()
    report = SERVICE_ENV_FILE.parent / ".tmp" / f"day7-eval-test-{uuid4().hex}" / "failed.json"
    calls = []

    def fail(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(module.subprocess, "run", fail)
    assert module.main(["--report", str(report)], stdout=io.StringIO()) == 1
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and saved["pytest_exit_code"] == 2
    assert saved["scope"] == "deterministic_no_live_requests"
    assert saved["network"]["verified"] is False
    assert all(case["status"] == "not_run" for case in saved["cases"])
    assert len(calls) == 1 and calls[0][1]["shell"] is False
    assert "scripts.action_eval_worker" in calls[0][0] and "pytest_asyncio.plugin" in calls[0][0]


def test_network_guard_blocks_connection_without_recording_destination(tmp_path):
    module = runner()
    from scripts.eval_policy import _network_guard
    import socket
    attempts = []
    with _network_guard(attempts):
        with socket.socket() as sock, pytest.raises(OSError, match="evaluation_network_disabled"):
            sock.connect(("192.0.2.1", 80))
    assert attempts == [True]


def test_pytest_environment_cannot_filter_parameter_cases(monkeypatch):
    module = runner()
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k not timeout")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted_plugin")
    report = SERVICE_ENV_FILE.parent / ".tmp" / f"day7-eval-test-{uuid4().hex}" / "env.json"
    captured = []
    monkeypatch.setattr(module.subprocess, "run",
                        lambda command, **kwargs: captured.append((command, kwargs)) or SimpleNamespace(returncode=2))
    module.main(["--report", str(report)], stdout=io.StringIO())
    environment = captured[0][1]["env"]
    assert "PYTEST_ADDOPTS" not in environment
    assert "PYTEST_PLUGINS" not in environment
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "pytest_asyncio.plugin" in captured[0][0]


@pytest.mark.parametrize("elapsed", ["1e308", "nan", "-1"])
def test_invalid_junit_time_is_a_controlled_failure(tmp_path, elapsed):
    module = runner()
    cases = module.load_cases(matrix(tmp_path / "cases.json"))
    xml = tmp_path / "junit.xml"
    xml.write_text("<testsuites><testcase classname='tests.test_action_history' "
                   "name='test_history_reopens_and_terminal_replay_does_not_append' "
                   f"time='{elapsed}'/></testsuites>", encoding="utf-8")
    with pytest.raises(ValueError):
        module.case_results(xml, cases)


def worker():
    assert importlib.util.find_spec("scripts.action_eval_worker") is not None, "guarded worker is missing"
    return importlib.import_module("scripts.action_eval_worker")


def test_guard_allows_only_owned_stdlib_socketpairs_not_generic_loopback_connections():
    import asyncio
    import socket
    module = worker()
    attempts, pairs = [], []
    with module.network_guard(attempts, pairs):
        loop = asyncio.new_event_loop()
        loop.close()
        for destination in [("192.0.2.1", 80), ("127.0.0.1", 8008)]:
            with socket.socket() as sock, pytest.raises(OSError, match="evaluation_network_disabled"):
                sock.connect(destination)
    assert attempts == [True, True]


def test_worker_guard_covers_pytest_import_and_all_shutdown_hooks(monkeypatch):
    import builtins
    import socket
    module = worker()
    original = builtins.__import__
    phase_attempts = []

    def blocked():
        with socket.socket() as sock, pytest.raises(OSError, match="evaluation_network_disabled"):
            sock.connect(("192.0.2.1", 80))
        phase_attempts.append(True)

    class PytestBoundary:
        @staticmethod
        def main(args):
            blocked()
            return 0

    def importing(name, *args, **kwargs):
        if name == "pytest":
            blocked()
            return PytestBoundary
        return original(name, *args, **kwargs)

    directory = SERVICE_ENV_FILE.parent / ".tmp" / f"day7-action-eval-{uuid4().hex}"
    directory.mkdir(parents=True)
    monkeypatch.setattr(builtins, "__import__", importing)
    assert module.main(["--network-report", str(directory / "network.json"), "--", "-q"]) == 1
    observed = json.loads((directory / "network.json").read_text(encoding="utf-8"))
    assert observed["connection_attempts"] == 2 and phase_attempts == [True, True]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows IOCP network boundary")
@pytest.mark.parametrize("destination", [("192.0.2.1", 80), ("127.0.0.1", 8008)])
@pytest.mark.parametrize("operation", ["connect_tcp", "connect_udp", "sendto_udp"])
def test_guard_blocks_proactor_network_before_native_io(monkeypatch, destination, operation):
    import asyncio
    from asyncio.windows_events import IocpProactor
    import socket

    module = worker()
    native_calls = []

    def intercepted_native_boundary(proactor, *args):
        # Keep RED safe: an escaped connection stops at this fake native boundary.
        native_calls.append(True)
        result = proactor._loop.create_future()
        result.set_result(0)
        return result

    monkeypatch.setattr(IocpProactor, "connect", intercepted_native_boundary)
    monkeypatch.setattr(IocpProactor, "sendto", intercepted_native_boundary)
    attempts, pairs = [], []
    with module.network_guard(attempts, pairs):
        loop = asyncio.ProactorEventLoop()
        try:
            kind = socket.SOCK_STREAM if operation == "connect_tcp" else socket.SOCK_DGRAM
            with socket.socket(type=kind) as sock:
                sock.setblocking(False)
                request = (loop.sock_sendto(sock, b"probe", destination) if operation == "sendto_udp"
                           else loop.sock_connect(sock, destination))
                with pytest.raises(OSError, match="^evaluation_network_disabled$"):
                    loop.run_until_complete(request)
        finally:
            loop.close()
    assert attempts == [True] and native_calls == []
    assert IocpProactor.connect is intercepted_native_boundary
    assert IocpProactor.sendto is intercepted_native_boundary
