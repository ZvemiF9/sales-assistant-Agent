import importlib
import subprocess
from pathlib import Path

import pytest


DB_CONFIG = {
    "YSHOP_DB_NAME": "shop_test-01",
    "YSHOP_DB_USER": "launcher-private-user",
    "YSHOP_DB_PASSWORD": "launcher-private-password",
}
PRIVATE_TOKEN = "launcher-private-token"


@pytest.fixture
def service(tmp_path):
    directory = tmp_path / "repo" / "yshop-agent"
    script = directory / "scripts" / "start_shop.py"
    script.parent.mkdir(parents=True)
    script.write_text("# staged source location\n", encoding="utf-8")
    java = tmp_path / "jdk" / "java.exe"
    java.parent.mkdir()
    java.write_bytes(b"test executable placeholder")
    jar = directory.parent / "yshop-mall-boot" / "yshop-app" / "target" / "yshop-app-2.0.0.jar"
    jar.parent.mkdir(parents=True)
    jar.write_bytes(b"test JAR placeholder")
    env_file = directory / ".env"
    env_file.write_text("\n".join(f"{key}={value}" for key, value in DB_CONFIG.items())
                        + f"\nYSHOP_AUTHORIZATION={PRIVATE_TOKEN}\n", encoding="utf-8")
    return directory, script, java, jar, env_file


def run_launcher(monkeypatch, service, *, environ=None, argv=None, exit_code=0, failure=None):
    try:
        launcher = importlib.import_module("scripts.start_shop")
    except ModuleNotFoundError:
        pytest.fail("the required local Java startup helper is not implemented")
    directory, script, java, jar, _ = service
    monkeypatch.setattr(launcher, "__file__", str(script))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if failure is not None:
            raise failure
        return subprocess.CompletedProcess(command, exit_code)

    def no_real_process(*args, **kwargs):
        pytest.fail("launcher tests must never start a real subprocess")

    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher.subprocess, "Popen", no_real_process)
    arguments = ["--java", str(java)] if argv is None else argv
    code = launcher.main(arguments, environ={} if environ is None else environ)
    return code, calls


def test_source_relative_env_and_jar_ignore_current_working_directory(monkeypatch, service, tmp_path, capsys):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / ".env").write_text("YSHOP_DB_NAME=wrong_database\n", encoding="utf-8")
    monkeypatch.chdir(unrelated)

    code, calls = run_launcher(monkeypatch, service)

    assert code == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == [str(service[2]), "-jar", str(service[3])]
    assert Path(kwargs["cwd"]) == service[0]
    assert kwargs["env"]["SPRING_DATASOURCE_DRUID_MASTER_URL"] == (
        "jdbc:mysql://127.0.0.1:3306/shop_test-01?useUnicode=true&characterEncoding=utf8"
        "&zeroDateTimeBehavior=convertToNull&useSSL=false&serverTimezone=GMT%2B8"
        "&allowPublicKeyRetrieval=true"
    )
    assert kwargs["env"]["SPRING_DATASOURCE_DRUID_MASTER_USERNAME"] == DB_CONFIG["YSHOP_DB_USER"]
    assert kwargs["env"]["SPRING_DATASOURCE_DRUID_MASTER_PASSWORD"] == DB_CONFIG["YSHOP_DB_PASSWORD"]
    assert kwargs["shell"] is False
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert "--spring.output.ansi.enabled=never" in command
    assert "--mybatis-plus.configuration.log-impl=org.apache.ibatis.logging.nologging.NoLoggingImpl" in command
    for logger in ("co.yixiang", "org.springframework.web", "com.github.binarywang.demo.wx.mp", "me.chanjar.weixin"):
        assert f"--logging.level.{logger}=INFO" in command
    output = capsys.readouterr()
    for secret in (DB_CONFIG["YSHOP_DB_USER"], DB_CONFIG["YSHOP_DB_PASSWORD"], PRIVATE_TOKEN):
        assert secret not in " ".join(command) + output.out + output.err
    assert "YSHOP_AUTHORIZATION" not in kwargs["env"]


def test_token_bearing_repeat_submit_info_logger_is_disabled(monkeypatch, service):
    code, calls = run_launcher(monkeypatch, service)
    assert code == 0
    command = calls[0][0]
    assert "--logging.level.co.yixiang.common.aspect.RepeatSubmitAspect=OFF" in command


def test_environment_overrides_dotenv_without_mutating_parent(monkeypatch, service, capsys):
    environ = {**DB_CONFIG, "YSHOP_DB_NAME": "env_database", "YSHOP_DB_PASSWORD": "override-private-password",
               "PATH": "test-path", "SPRING_DATASOURCE_DRUID_MASTER_PASSWORD": "old-parent-password"}
    before = environ.copy()

    code, calls = run_launcher(monkeypatch, service, environ=environ, exit_code=7)

    assert code == 7
    child = calls[0][1]["env"]
    assert "/env_database?" in child["SPRING_DATASOURCE_DRUID_MASTER_URL"]
    assert child["SPRING_DATASOURCE_DRUID_MASTER_PASSWORD"] == environ["YSHOP_DB_PASSWORD"]
    assert child["PATH"] == "test-path"
    assert environ == before
    output = capsys.readouterr()
    assert environ["YSHOP_DB_PASSWORD"] not in " ".join(calls[0][0]) + output.out + output.err


@pytest.mark.parametrize("name", ["", "shop?password=private", "shop/other", "shop;--debug", "shop\n", "数据库"])
def test_invalid_database_names_prevent_launch(monkeypatch, service, name, capsys):
    code, calls = run_launcher(monkeypatch, service, environ={"YSHOP_DB_NAME": name})
    assert code != 0
    assert calls == []
    output = capsys.readouterr()
    assert "YSHOP_DB_NAME" in output.out + output.err
    assert "shop?password=private" not in output.out + output.err


@pytest.mark.parametrize("field", ["YSHOP_DB_USER", "YSHOP_DB_PASSWORD"])
def test_explicit_empty_environment_credential_does_not_fall_back(monkeypatch, service, field, capsys):
    code, calls = run_launcher(monkeypatch, service, environ={field: ""})
    assert code != 0
    assert calls == []
    output = capsys.readouterr()
    assert field in output.out + output.err
    assert DB_CONFIG["YSHOP_DB_PASSWORD"] not in output.out + output.err


def test_missing_dotenv_can_use_complete_environment(monkeypatch, service):
    service[4].unlink()
    code, calls = run_launcher(monkeypatch, service, environ=DB_CONFIG)
    assert code == 0
    assert len(calls) == 1


def test_dotenv_preserves_literal_password_without_interpolation(monkeypatch, service):
    service[4].write_text("YSHOP_DB_NAME=shop\nYSHOP_DB_USER=user\nYSHOP_DB_PASSWORD='${PRIVATE_VALUE}'\n",
                         encoding="utf-8")
    monkeypatch.setenv("PRIVATE_VALUE", "unexpected-private-expansion")
    code, calls = run_launcher(monkeypatch, service)
    assert code == 0
    assert calls[0][1]["env"]["SPRING_DATASOURCE_DRUID_MASTER_PASSWORD"] == "${PRIVATE_VALUE}"


@pytest.mark.parametrize("missing", ["java", "jar", "credentials"])
def test_missing_prerequisites_prevent_launch(monkeypatch, service, missing, capsys):
    if missing == "java":
        service[2].unlink()
    elif missing == "jar":
        service[3].unlink()
    else:
        service[4].unlink()
    code, calls = run_launcher(monkeypatch, service)
    assert code != 0
    assert calls == []
    output = capsys.readouterr()
    assert output.out + output.err
    assert DB_CONFIG["YSHOP_DB_PASSWORD"] not in output.out + output.err


@pytest.mark.parametrize("failure", [FileNotFoundError("launcher-private-password"),
                                     OSError("launcher-private-token")])
def test_os_errors_return_static_message_without_error_details(monkeypatch, service, failure, capsys):
    code, calls = run_launcher(monkeypatch, service, failure=failure)
    assert code != 0
    assert len(calls) == 1
    output = capsys.readouterr()
    assert output.out + output.err
    assert DB_CONFIG["YSHOP_DB_PASSWORD"] not in output.out + output.err
    assert PRIVATE_TOKEN not in output.out + output.err


def test_jar_override_and_ctrl_c_exit_without_traceback(monkeypatch, service, tmp_path, capsys):
    jar = tmp_path / "custom.jar"
    jar.write_bytes(b"custom JAR placeholder")
    code, calls = run_launcher(monkeypatch, service, argv=["--java", str(service[2]), "--jar", str(jar)],
                               failure=KeyboardInterrupt())
    assert code == 130
    assert calls[0][0][:3] == [str(service[2]), "-jar", str(jar)]
    output = capsys.readouterr()
    assert "Traceback" not in output.out + output.err


@pytest.mark.parametrize("arguments", [["--unexpected", PRIVATE_TOKEN], ["--java"], ["--jar\0bad"],
                                      ["--java", "\0" + PRIVATE_TOKEN], ["--jar", "\0" + PRIVATE_TOKEN]])
def test_invalid_cli_arguments_do_not_echo_values_or_start_java(monkeypatch, service, arguments, capsys):
    code, calls = run_launcher(monkeypatch, service, argv=arguments)
    assert code != 0
    assert calls == []
    output = capsys.readouterr()
    assert PRIVATE_TOKEN not in output.out + output.err
    assert "--unexpected" not in output.out + output.err


def test_unreadable_env_returns_static_message_without_launch(monkeypatch, service, capsys):
    service[4].write_bytes(b"YSHOP_DB_NAME=shop\nYSHOP_DB_PASSWORD=\xff\n")
    code, calls = run_launcher(monkeypatch, service)
    assert code != 0
    assert calls == []
    output = capsys.readouterr()
    assert "UTF-8" in output.out + output.err
    assert "Traceback" not in output.out + output.err
