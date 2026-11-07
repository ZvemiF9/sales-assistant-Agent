"""Exclusive reports and orchestration for the frozen 50-case development suite."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

from app.config import SERVICE_ENV_FILE
from app.errors import ShopError
from app.rag.config import SERVICE_ROOT
from scripts.action_smoke_test import _secrets
from scripts.chat_smoke_test import _contains_secret
from scripts.day9_cases import CASES_PATH, SUITE_PATHS, digest, load_cases, summarize
from scripts.day9_live import LiveRunner, OwnedAgent
from scripts.day9_observer import counts
from scripts.eval_actions import case_results
from scripts.eval_policy import _arguments, _project_path, _report_path
from scripts.smoke_test import _SafeParser, _authorization, _integer


class OwnedReport:
    """Retain the exclusive descriptor and verify its pathname before every save."""

    def __init__(self, path, *, environ, env_file, case_path=CASES_PATH):
        self.case_path = Path(case_path)
        try:
            self.path = _report_path(path, environ=environ, env_file=env_file, case_path=self.case_path)
        except ShopError:
            raise ValueError("invalid report path") from None
        self.environ, self.env_file, self.stream, self.identity = environ, env_file, None, None
        if self.path.exists() or self.path.with_name(self.path.name + ".tmp").exists():
            raise ValueError("report already exists")

    def reserve(self, report):
        if self.stream is not None:
            raise ValueError("report already reserved")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x+", encoding="utf-8")
        stat = os.fstat(self.stream.fileno())
        self.identity = stat.st_dev, stat.st_ino
        try:
            self.save(report)
        except Exception:
            self.close()
            raise

    def save(self, report):
        if self.stream is None:
            raise ValueError("report not owned")
        try:
            path = _report_path(self.path, environ=self.environ, env_file=self.env_file,
                                case_path=self.case_path)
        except ShopError:
            raise ValueError("report confinement changed") from None
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) != self.identity or stat.st_nlink != 1:
            raise ValueError("report ownership changed")
        content = json.dumps(report, indent=2, allow_nan=False)
        self.stream.seek(0)
        self.stream.write(content + "\n")
        self.stream.truncate()
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def deterministic_results(cases, junit, network, exit_code):
    rows = [{**c, "status": "not_run", "test_count": 0, "elapsed_ms": None} for c in cases]
    observed = {"verified": False, "connection_attempts": None, "local_socketpair_connections": None}
    valid = False
    if Path(junit).is_file():
        rows = case_results(Path(junit), cases)
        for row in rows:
            if row["test_count"] == 0:
                row["elapsed_ms"] = None
    if Path(network).is_file():
        if Path(network).stat().st_size > 4096:
            raise ValueError("oversized worker evidence")
        from app.agent.model import _loads
        value = _loads(Path(network).read_bytes())
        valid = (type(value) is dict and set(value) == set(observed) and value["verified"] is True
                 and type(value["connection_attempts"]) is int and value["connection_attempts"] >= 0
                 and type(value["local_socketpair_connections"]) is int and value["local_socketpair_connections"] >= 0)
        if not valid:
            raise ValueError("invalid worker network evidence")
        observed = value
    return {"rows": rows, "pytest_exit_code": exit_code, "network": observed,
            "gate_passed": exit_code == 0 and valid and observed["connection_attempts"] == 0
            and all(r["status"] == "passed" for r in rows) and bool(rows)}


def run_deterministic(cases, *, run_id, environ, update, report, save):
    runtime = _project_path(SERVICE_ROOT / ".tmp" / ("day7-action-eval-" + run_id))
    runtime.mkdir(parents=True, exist_ok=False)
    junit, network = runtime / "junit.xml", runtime / "network.json"
    native = SERVICE_ROOT.parent / "logs/agent-day9" / ("day9-worker-" + run_id + ".txt")
    if native.exists():
        raise ValueError("worker log exists")
    native.parent.mkdir(parents=True, exist_ok=True)
    report["artifacts"].update(junit=junit.relative_to(SERVICE_ROOT).as_posix(),
                               network=network.relative_to(SERVICE_ROOT).as_posix(),
                               native_output=native.relative_to(SERVICE_ROOT.parent).as_posix())
    save()
    environment = {**environ, "TMP": str(runtime), "TEMP": str(runtime),
                   "PYTHONPYCACHEPREFIX": str(runtime / "p"), "HF_HUB_OFFLINE": "1",
                   "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        environment.pop(key, None)
    executable = sys.executable
    if sys.platform == "win32" and sys.executable != sys._base_executable:
        executable = sys._base_executable
        environment["__PYVENV_LAUNCHER__"] = sys.executable
    command = [executable, "-X", "utf8", "-W", "error", "-m", "scripts.action_eval_worker",
        "--network-report", str(network), "--", "-p", "pytest_asyncio.plugin", "-p", "no:cacheprovider",
        "-c", str(SERVICE_ROOT / "pyproject.toml"), "-o", "addopts=",
        *[c["selector"] for c in cases], "-q", "--tb=no", "--junitxml", str(junit), "--basetemp", str(runtime / "t")]
    exit_code = None
    try:
        with native.open("x", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=SERVICE_ROOT, env=environment, shell=False,
                stdout=output, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                exit_code = process.wait(timeout=600)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=20)
    finally:
        result = deterministic_results(cases, junit, network, exit_code)
        report["deterministic"] = {k: v for k, v in result.items() if k != "rows"}
        for row in result["rows"]:
            update(row)
        save()


def finalize_live(report, agent):
    process = agent.stop()
    report["owned_process"] = process
    report["observations"] = counts(agent.events())
    report["provenance"]["external_model_verified"] = report["observations"]["external_model_verified"]
    if process["unexpected_exit"]:
        report.update(status="failed", error_code="owned_child_exited_unexpectedly")
    elif (report["observations"]["blocked_shop_writes"]
          or report["observations"]["blocked_unexpected_requests"]
          or not report["provenance"]["external_model_verified"]):
        report.update(status="failed", error_code="live_observation_gate_failed")


def new_report(cases, *, manifest_hash, suite_id, manifest_path, run_id, mode, port):
    return {"schema_version": 1, "status": "pending", "run_id": run_id,
            "suite_id": suite_id, "manifest_path": manifest_path,
            "manifest_sha256": manifest_hash, "manifest_unchanged_before_run": False,
            "manifest_unchanged_after_run": False,
            "evaluation_scope": "development_suite_not_independent_generalization",
            "configuration": {"mode": mode, "agent_port": port, "host": "127.0.0.1",
                              "live_case_count": 30, "deterministic_case_count": 20,
                              "llm_timeout_seconds": 60, "llm_enable_thinking": False},
            "declarations": {"no_live_business_writes": True, "response_replacement": False,
                             "model_auto_retry": False, "fresh_sqlite": True, "copied_chroma": True,
                             "original_service_shutdown": False, "embedding_downloads": False},
            "provenance": {"external_model_verified": False, "observation_source": "owned_child_httpx_send",
                           "verification_gets_scope": "independent_runner_not_agent_issued_tools"},
            "live_cart_acceptance": {"measured": False, "claim": None, "prior_day6_evidence": "separate_historical_evidence"},
            "deterministic": {"gate_passed": False, "pytest_exit_code": None,
                              "network": {"verified": False, "connection_attempts": None}},
            "observations": counts([]), "artifacts": {}, "independent_verifications": [],
            "cases": [{**{k: c[k] for k in ("id", "group", "mode")}, "status": "not_run",
                       **({"steps": [{"status": "not_run"} for _ in c["steps"]]} if "steps" in c else
                          {"selector": c["selector"], "test_count": 0, "elapsed_ms": None})} for c in cases]}


def main(argv=None, *, environ=None, env_file=SERVICE_ENV_FILE, stdout=None):
    environment = dict(os.environ if environ is None else environ)
    stdout = sys.stdout if stdout is None else stdout
    report, owner, agent, runner, secrets = None, None, None, None, ()
    cases_path = CASES_PATH
    started = perf_counter()
    exit_code = 1
    try:
        parser = _SafeParser(description="Frozen 50-case read-only development evaluator.", allow_abbrev=False)
        parser.add_argument("--report", required=True)
        parser.add_argument("--port", type=_integer, default=8019)
        parser.add_argument("--mode", choices=("all", "deterministic"), default="all")
        parser.add_argument("--suite", choices=tuple(SUITE_PATHS), default="day9-v1")
        options = _arguments(argv, parser=parser, flags={"--report", "--port", "--mode", "--suite"})
        if not 1024 <= options.port <= 65535:
            raise ValueError("invalid port")
        cases_path = Path(SUITE_PATHS[options.suite])
        manifest_hash, run_id = digest(cases_path), uuid4().hex
        cases = load_cases(cases_path)
        if digest(cases_path) != manifest_hash:
            raise ValueError("manifest changed during validation")
        report = new_report(cases, manifest_hash=manifest_hash, suite_id=options.suite,
                            manifest_path=cases_path.relative_to(SERVICE_ROOT).as_posix(),
                            run_id=run_id, mode=options.mode, port=options.port)
        owner = OwnedReport(options.report, environ=environment, env_file=env_file,
                            case_path=cases_path)
        owner.reserve(report)

        def save():
            if _contains_secret(report, secrets):
                raise ValueError("private report field")
            report["summary"] = summarize(report["cases"])
            owner.save(report)

        def update(row):
            position = next(i for i, existing in enumerate(report["cases"]) if existing["id"] == row["id"])
            report["cases"][position] = row
            save()

        report["manifest_unchanged_before_run"] = digest(cases_path) == manifest_hash
        if not report["manifest_unchanged_before_run"]:
            raise ValueError("manifest changed before run")
        save()
        run_deterministic([c for c in cases if c["mode"] == "deterministic"],
                          run_id=run_id, environ=environment, update=update, report=report, save=save)
        if options.mode == "all":
            authorization = _authorization(environment, Path(env_file))
            secrets = _secrets(environment, env_file, authorization)
            if _contains_secret(str(owner.path), secrets):
                raise ValueError("private report path")
            runtime = SERVICE_ROOT / ".tmp" / ("day9-live-" + run_id)
            report["artifacts"].update(live_runtime=runtime.relative_to(SERVICE_ROOT).as_posix(),
                observations=(runtime / "observations.jsonl").relative_to(SERVICE_ROOT).as_posix(),
                sessions=(runtime / "sessions.sqlite3").relative_to(SERVICE_ROOT).as_posix(),
                copied_index=(runtime / "index").relative_to(SERVICE_ROOT).as_posix())
            save()
            agent = OwnedAgent(runtime, port=options.port, run_id=run_id, environ=environment)
            agent.start()
            runner = LiveRunner(agent, authorization=authorization, secrets=secrets)
            asyncio.run(runner.run([c for c in cases if c["mode"] == "live"], update))
            report["independent_verifications"] = runner.verifications
        report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "incomplete"
    except SystemExit as error:
        return int(error.code or 0)
    except KeyboardInterrupt:
        if report is not None:
            report.update(status="incomplete", error_code="evaluation_interrupted")
    except Exception:
        if report is not None:
            report.update(status="failed", error_code="day9_evaluation_failed")
    finally:
        if agent is not None:
            try:
                finalize_live(report, agent)
            except Exception:
                report.update(status="failed", error_code="owned_child_or_evidence_failed")
        if report is not None and owner is not None and owner.stream is not None:
            try:
                if runner is not None:
                    secrets = runner.secrets
                    report["independent_verifications"] = runner.verifications
                report["manifest_unchanged_after_run"] = digest(cases_path) == report["manifest_sha256"]
                if not report["manifest_unchanged_after_run"]:
                    report.update(status="failed", error_code="manifest_changed")
                if not report["deterministic"]["gate_passed"]:
                    report.update(status="failed", error_code=report.get("error_code", "deterministic_gate_failed"))
                report["summary"] = summarize(report["cases"])
                report["elapsed_ms"] = round((perf_counter() - started) * 1000)
                if _contains_secret(report, secrets):
                    raise ValueError("private final report")
                owner.save(report)
                exit_code = 0 if report["status"] == "passed" else 1
            except Exception:
                report.update(status="failed", error_code="report_save_failed")
            finally:
                owner.close()
    summary = {"status": report["status"] if report is not None else "not_run",
               "error_code": report.get("error_code") if report else "invalid_evaluation_configuration",
               "case_success_rate": report.get("summary", {}).get("case_success_rate") if report else None,
               "deterministic_gate_passed": report["deterministic"]["gate_passed"] if report else False,
               "external_model_verified": report["provenance"]["external_model_verified"] if report else False}
    print(json.dumps(summary, allow_nan=False), file=stdout)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
