"""Fixed deterministic action evaluation with an exclusive, honest JSON report."""

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from uuid import uuid4
import xml.etree.ElementTree as ET

from app.config import SERVICE_ENV_FILE
from app.errors import ShopError
from app.rag.config import SERVICE_ROOT
from scripts.eval_policy import _arguments, _project_path, _report_path
from scripts.smoke_test import _SafeParser
from app.tools.yshop import _unique_json_object


CASES_PATH = SERVICE_ROOT / "evals" / "action_failure_cases.json"
_SELECTOR = re.compile(r"tests/test_[a-z0-9_]+\.py::test_[a-z0-9_]+\Z")
_NAME = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_GROUPS = {"history", "storage", "isolation", "confirmation", "preflight", "failure", "restart", "privacy"}


def load_cases(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    if (type(value) is not dict or set(value) != {"schema_version", "scope", "cases"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["scope"] != "deterministic_no_live_requests"
            or type(value["cases"]) is not list or not 1 <= len(value["cases"]) <= 100):
        raise ValueError("invalid matrix")
    seen_ids, seen_selectors = set(), set()
    for case in value["cases"]:
        if (type(case) is not dict or set(case) != {"id", "group", "selector"}
                or any(type(v) is not str for v in case.values())
                or not _NAME.fullmatch(case["id"]) or case["group"] not in _GROUPS
                or not _SELECTOR.fullmatch(case["selector"])
                or case["id"] in seen_ids or case["selector"] in seen_selectors):
            raise ValueError("invalid case")
        file, name = case["selector"].split("::")
        source = _project_path(file)
        if not source.is_file():
            raise ValueError("missing test")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
                   for node in tree.body):
            raise ValueError("missing test function")
        seen_ids.add(case["id"])
        seen_selectors.add(case["selector"])
    return value["cases"]


def case_results(path: Path, cases: list[dict]) -> list[dict]:
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("oversized test result")
    root = ET.parse(path).getroot()
    tests = list(root.iter("testcase"))
    results = []
    for case in cases:
        file, name = case["selector"].split("::")
        classname = file[:-3].replace("/", ".")
        matches = [node for node in tests if node.get("classname") == classname
                   and (node.get("name") == name or node.get("name", "").startswith(name + "["))]
        elapsed = sum(float(node.get("time", "0")) for node in matches)
        if not math.isfinite(elapsed) or not 0 <= elapsed <= 600:
            raise ValueError("invalid elapsed time")
        status = ("not_run" if not matches
                  else "failed" if any(node.find("failure") is not None or node.find("error") is not None for node in matches)
                  else "skipped" if any(node.find("skipped") is not None for node in matches)
                  else "passed")
        results.append({**case, "status": status, "test_count": len(matches),
                        "elapsed_ms": round(elapsed * 1000)})
    return results


def main(argv=None, *, stdout=None) -> int:
    stdout = sys.stdout if stdout is None else stdout
    report = {"schema_version": 1, "scope": "deterministic_no_live_requests", "status": "failed",
              "pytest_exit_code": None, "network": {"verified": False, "connection_attempts": None,
                                                 "local_socketpair_connections": None},
              "cases": []}
    started = perf_counter()
    destination = None
    try:
        parser = _SafeParser(description="Run the fixed offline action boundary matrix.", allow_abbrev=False)
        parser.add_argument("--report", required=True)
        options = _arguments(argv, parser=parser, flags={"--report"})
        path = _report_path(options.report, environ=os.environ, env_file=SERVICE_ENV_FILE)
        cases = load_cases(CASES_PATH)
        report["matrix_sha256"] = hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()
        report["cases"] = [{**case, "status": "not_run", "test_count": 0, "elapsed_ms": 0} for case in cases]
        path.parent.mkdir(parents=True, exist_ok=True)
        destination = path.open("x", encoding="utf-8")
        destination.write(json.dumps(report))
        destination.flush()
        runtime = _project_path(SERVICE_ROOT / ".tmp" / ("day7-action-eval-" + uuid4().hex))
        runtime.mkdir(parents=True)
        junit, network = runtime / "junit.xml", runtime / "network.json"
        report["artifacts"] = {name: str(value.relative_to(SERVICE_ROOT)).replace("\\", "/")
                               for name, value in {"junit": junit, "network": network,
                                                   "native_output": runtime / "pytest.txt"}.items()}
        environment = dict(os.environ, TMP=str(runtime), TEMP=str(runtime),
                           PYTHONPYCACHEPREFIX=str(runtime / "p"), HF_HUB_OFFLINE="1",
                           PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            environment.pop(name, None)
        command = [sys.executable, "-X", "utf8", "-W", "error", "-m", "scripts.action_eval_worker",
                   "--network-report", str(network), "--", "-p", "pytest_asyncio.plugin",
                   "-c", str(SERVICE_ROOT / "pyproject.toml"), "-o", "addopts=",
                   *[case["selector"] for case in cases],
                   "-q", "--junitxml", str(junit), "--basetemp", str(runtime / "t")]
        with (runtime / "pytest.txt").open("x", encoding="utf-8") as native:
            result = subprocess.run(command, cwd=SERVICE_ROOT, env=environment, shell=False,
                                    stdout=native, stderr=subprocess.STDOUT, timeout=600)
        report["pytest_exit_code"] = result.returncode
        if junit.is_file():
            report["cases"] = case_results(junit, cases)
        if network.is_file():
            observed = json.loads(network.read_text(encoding="utf-8"))
            if (type(observed) is not dict or set(observed) != {"verified", "connection_attempts", "local_socketpair_connections"}
                    or observed["verified"] is not True or type(observed["connection_attempts"]) is not int
                    or observed["connection_attempts"] < 0 or type(observed["local_socketpair_connections"]) is not int
                    or observed["local_socketpair_connections"] < 0):
                raise ValueError("invalid network observation")
            report["network"] = observed
        report["passed_groups"] = sum(case["status"] == "passed" for case in report["cases"])
        report["test_count"] = sum(case["test_count"] for case in report["cases"])
        if (result.returncode == 0 and report["network"]["verified"] is True
                and report["network"]["connection_attempts"] == 0
                and report["passed_groups"] == len(cases)):
            report["status"] = "passed"
    except (OSError, ValueError, ShopError, subprocess.SubprocessError, ET.ParseError, RecursionError, SyntaxError):
        report["status"] = "failed"
        report["error_code"] = "action_evaluation_failed"
    finally:
        report["elapsed_ms"] = round((perf_counter() - started) * 1000)
        if destination is not None:
            try:
                destination.seek(0)
                destination.truncate()
                json.dump(report, destination, indent=2, allow_nan=False)
            except OSError:
                report["status"] = "failed"
            finally:
                destination.close()
    print(json.dumps({"status": report["status"], "scope": report["scope"],
                      "passed_groups": report.get("passed_groups", 0),
                      "group_count": len(report["cases"]), "test_count": report.get("test_count", 0)}), file=stdout)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
