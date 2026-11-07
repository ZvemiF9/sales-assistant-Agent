"""Version-selection contracts for the strict Day 9 evaluator."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from uuid import uuid4

import pytest

from scripts import day9_cases
from scripts import eval_day9


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = "b1ef58e9ee2b483bc1ec0343ab0d8795510102091402592cc2064e6fb1748ff1"


def _report_path(label):
    return ROOT / ".tmp" / ("day9-suite-version-" + label + "-" + uuid4().hex) / "report.json"


def _substitute_execution_boundaries(monkeypatch, observed):
    def deterministic(cases, *, report, update, save, **kwargs):
        observed["deterministic"] = [case["id"] for case in cases]
        report["deterministic"] = {
            "gate_passed": True,
            "pytest_exit_code": 0,
            "network": {"verified": True, "connection_attempts": 0},
        }
        for case in cases:
            update({**case, "status": "passed", "test_count": 1, "elapsed_ms": 0})
        save()

    class Agent:
        def __init__(self, *args, **kwargs):
            observed["agent_constructed"] = True

        def start(self):
            observed["agent_started"] = True

        def stop(self):
            return {"pid": 1, "exit_code": 0, "awaited": True,
                    "termination_requested": True, "exited_before_shutdown": False,
                    "unexpected_exit": False}

        def events(self):
            return []

    class Runner:
        def __init__(self, *args, **kwargs):
            self.secrets = ()
            self.verifications = []

        async def run(self, cases, update):
            observed["live"] = cases
            for case in cases:
                update({**case, "status": "passed",
                        "steps": [{"status": "passed"} for _ in case["steps"]]})

    monkeypatch.setattr(eval_day9, "run_deterministic", deterministic)
    monkeypatch.setattr(eval_day9, "OwnedAgent", Agent)
    monkeypatch.setattr(eval_day9, "LiveRunner", Runner)
    monkeypatch.setattr(eval_day9, "finalize_live", lambda report, agent: None)
    monkeypatch.setattr(eval_day9, "_authorization", lambda *args: "Bearer test")
    monkeypatch.setattr(eval_day9, "_secrets", lambda *args: ())


def test_v2_is_exact_revision_and_v1_remains_frozen():
    original_path = ROOT / "evals/day9_cases.json"
    revised_path = ROOT / "evals/day9_cases_v2.json"
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == ORIGINAL_SHA256

    original = json.loads(original_path.read_text(encoding="utf-8"))
    revised = json.loads(revised_path.read_text(encoding="utf-8"))
    expected = deepcopy(original)
    for case in expected["cases"]:
        if case["id"] not in {"product-baidaiya500", "multi-changing-keyword"}:
            continue
        for step in case["steps"]:
            if step["expect"].get("query", {}).get("keyword") == "百代亚":
                step["message"] = step["message"].replace("百代亚", "柏黛雅")
                step["expect"]["query"]["keyword"] = "柏黛雅"
    assert revised == expected
    assert day9_cases.SUITE_PATHS == {
        "day9-v1": original_path,
        "day9-v2": revised_path,
    }
    assert len(day9_cases.load_cases(revised_path)) == 50


@pytest.mark.parametrize("arguments,suite_id", [([], "day9-v1"), (["--suite", "day9-v2"], "day9-v2")])
def test_cli_selects_suite_executes_its_cases_and_records_identity(monkeypatch, arguments, suite_id):
    observed = {}
    _substitute_execution_boundaries(monkeypatch, observed)
    report_path = _report_path(suite_id)
    stdout = io.StringIO()

    result = eval_day9.main(["--report", str(report_path), *arguments], environ={},
                            env_file=ROOT / ".env", stdout=stdout)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected_path = day9_cases.SUITE_PATHS[suite_id]
    assert result == 0
    assert report["status"] == "passed"
    assert report["suite_id"] == suite_id
    assert report["manifest_path"] == selected_path.relative_to(ROOT).as_posix()
    assert report["manifest_sha256"] == hashlib.sha256(selected_path.read_bytes()).hexdigest()
    assert report["manifest_unchanged_before_run"] is True
    assert report["manifest_unchanged_after_run"] is True
    assert observed["agent_started"] is True
    assert len(observed["deterministic"]) == 20
    assert len(observed["live"]) == 30
    selected_keyword = next(case for case in observed["live"] if case["id"] == "product-baidaiya500")
    expected_keyword = "柏黛雅" if suite_id == "day9-v2" else "百代亚"
    assert selected_keyword["steps"][0]["expect"]["query"]["keyword"] == expected_keyword


@pytest.mark.parametrize("suite_arguments", [
    ["--suite", "unknown"],
    ["--suite", "day9-v1", "--suite", "day9-v2"],
])
def test_invalid_suite_flags_reject_before_worker_or_service_work(monkeypatch, suite_arguments):
    def unexpected(*args, **kwargs):
        pytest.fail("execution boundary reached for invalid suite flags")

    monkeypatch.setattr(eval_day9, "run_deterministic", unexpected)
    monkeypatch.setattr(eval_day9, "OwnedAgent", unexpected)
    report_path = _report_path("invalid")
    result = eval_day9.main(["--report", str(report_path), *suite_arguments], environ={},
                            env_file=ROOT / ".env", stdout=io.StringIO())
    assert result != 0
    assert not report_path.exists()


def test_selected_manifest_mutation_during_execution_fails(monkeypatch):
    source = ROOT / "evals/day9_cases_v2.json"
    selected = ROOT / ".tmp" / ("day9-selected-" + uuid4().hex + ".json")
    selected.write_bytes(source.read_bytes())
    suites = {**day9_cases.SUITE_PATHS, "day9-v2": selected}
    monkeypatch.setattr(eval_day9, "SUITE_PATHS", suites)

    def mutate(cases, *, report, update, save, **kwargs):
        selected.write_bytes(selected.read_bytes() + b"\n")
        report["deterministic"]["gate_passed"] = True
        save()

    monkeypatch.setattr(eval_day9, "run_deterministic", mutate)
    report_path = _report_path("mutation")
    result = eval_day9.main(["--report", str(report_path), "--suite", "day9-v2",
                             "--mode", "deterministic"], environ={}, env_file=ROOT / ".env",
                            stdout=io.StringIO())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 1
    assert report["status"] == "failed"
    assert report["error_code"] == "manifest_changed"
    assert report["manifest_unchanged_after_run"] is False
