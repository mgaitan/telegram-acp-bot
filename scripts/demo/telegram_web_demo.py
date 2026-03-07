# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "python-dotenv>=1.2.1",
# ]
# ///

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_SERVER_WAIT_SECONDS = 2.5
SERVER_START_FAILED_MESSAGE = "Demo bot server exited before recording started."
INVALID_WAIT_SECONDS_MESSAGE = "--server-wait-seconds must be a positive number"


class DemoCliConfig(argparse.Namespace):
    mode: str
    start_server: bool
    server_wait_seconds: float
    server_command: str
    recorder_script: Path


def parse_args() -> tuple[DemoCliConfig, list[str]]:
    parser = argparse.ArgumentParser(
        prog="telegram-web-demo",
        description="Run Telegram Web demo flow. In record mode it can auto-start the demo bot server.",
    )
    parser.add_argument("--mode", choices=("login", "record"), default="record")
    parser.add_argument(
        "--start-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-start demo bot server in record mode (default: true).",
    )
    parser.add_argument(
        "--server-wait-seconds",
        type=float,
        default=DEFAULT_SERVER_WAIT_SECONDS,
        help="Seconds to wait after starting server before recording begins.",
    )
    parser.add_argument(
        "--server-command",
        default=f"{shlex.quote(sys.executable)} scripts/demo/run_demo_bot.py",
        help="Command used to run demo bot server.",
    )
    parser.add_argument(
        "--recorder-script",
        type=Path,
        default=Path("scripts/demo/record_telegram_web_demo.py"),
        help="Path to recorder script.",
    )
    config, record_args = parser.parse_known_args(namespace=DemoCliConfig())
    return config, record_args


def _start_server(command: str) -> subprocess.Popen[bytes]:
    parts = shlex.split(command)
    return subprocess.Popen(parts)


def _run_recorder(recorder_script: Path, *, mode: str, record_args: list[str]) -> int:
    cmd = [sys.executable, str(recorder_script), "--mode", mode, *record_args]
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def _shutdown_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    load_dotenv(override=False)
    config, record_args = parse_args()

    if config.server_wait_seconds <= 0:
        raise SystemExit(INVALID_WAIT_SECONDS_MESSAGE)

    should_start_server = config.mode == "record" and config.start_server
    server_process: subprocess.Popen[bytes] | None = None

    try:
        if should_start_server:
            server_process = _start_server(config.server_command)
            time.sleep(config.server_wait_seconds)
            if server_process.poll() is not None:
                raise SystemExit(SERVER_START_FAILED_MESSAGE)
        return _run_recorder(config.recorder_script, mode=config.mode, record_args=record_args)
    finally:
        _shutdown_server(server_process)


if __name__ == "__main__":
    raise SystemExit(main())
