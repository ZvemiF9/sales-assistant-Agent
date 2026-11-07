"""Start the local shop JAR with database credentials in its environment."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from dotenv import dotenv_values


DEFAULT_JAVA = Path(r"D:\develop\JDK17\bin\java.exe")
DATABASE_FIELDS = ("YSHOP_DB_NAME", "YSHOP_DB_USER", "YSHOP_DB_PASSWORD")
QUIET_LOGGERS = ("co.yixiang", "org.springframework.web", "com.github.binarywang.demo.wx.mp", "me.chanjar.weixin")


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.start_shop", description="Start the existing local shop JAR.",
                                     exit_on_error=False, allow_abbrev=False)
    parser.add_argument("--java", type=Path, default=DEFAULT_JAVA, help="Java executable path")
    parser.add_argument("--jar", type=Path, help="Shop JAR path")
    try:
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            raise ValueError
    except (argparse.ArgumentError, ValueError):
        print("Invalid startup arguments. Use --help for supported options.", file=sys.stderr)
        return 2

    try:
        service_dir = Path(__file__).resolve().parents[1]
        jar = args.jar or service_dir.parent / "yshop-mall-boot" / "yshop-app" / "target" / "yshop-app-2.0.0.jar"
        java = args.java.resolve()
        jar = jar.resolve()
        if not java.is_file():
            print("Java executable is missing. Install the local JDK17 or pass --java with its executable path.", file=sys.stderr)
            return 2
        if not jar.is_file():
            print("Shop JAR is missing. Build yshop-app or pass --jar with the existing JAR path.", file=sys.stderr)
            return 2
    except (OSError, ValueError, RuntimeError):
        print("Cannot access startup files. Check --java, --jar, and file permissions.", file=sys.stderr)
        return 2

    parent_env = dict(os.environ if environ is None else environ)
    try:
        local_values = dotenv_values(service_dir / ".env", encoding="utf-8", interpolate=False)
    except (OSError, UnicodeError):
        print("Cannot read yshop-agent/.env. Check file permissions and UTF-8 encoding.", file=sys.stderr)
        return 2
    credentials = {field: parent_env.get(field, local_values.get(field)) for field in DATABASE_FIELDS}
    for field in DATABASE_FIELDS:
        value = credentials[field]
        if not value or not value.strip() or "\0" in value:
            print(f"Set a nonempty {field} in yshop-agent/.env or the environment.", file=sys.stderr)
            return 2
    if re.fullmatch(r"[A-Za-z0-9_-]+", credentials["YSHOP_DB_NAME"]) is None:
        print("YSHOP_DB_NAME must contain only letters, digits, underscores, or hyphens.", file=sys.stderr)
        return 2

    child_env = parent_env.copy()
    child_env["SPRING_DATASOURCE_DRUID_MASTER_URL"] = (
        f"jdbc:mysql://127.0.0.1:3306/{credentials['YSHOP_DB_NAME']}?useUnicode=true&characterEncoding=utf8"
        "&zeroDateTimeBehavior=convertToNull&useSSL=false&serverTimezone=GMT%2B8&allowPublicKeyRetrieval=true"
    )
    child_env["SPRING_DATASOURCE_DRUID_MASTER_USERNAME"] = credentials["YSHOP_DB_USER"]
    child_env["SPRING_DATASOURCE_DRUID_MASTER_PASSWORD"] = credentials["YSHOP_DB_PASSWORD"]
    command = [
        str(java), "-jar", str(jar),
        "--spring.output.ansi.enabled=never",
        "--mybatis-plus.configuration.log-impl=org.apache.ibatis.logging.nologging.NoLoggingImpl",
        *(f"--logging.level.{logger}=INFO" for logger in QUIET_LOGGERS),
        "--logging.level.co.yixiang.common.aspect.RepeatSubmitAspect=OFF",
    ]
    try:
        result = subprocess.run(command, env=child_env, cwd=service_dir, shell=False, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return result.returncode
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError):
        print("Java could not start. Check executable permissions, --java, and --jar paths.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
