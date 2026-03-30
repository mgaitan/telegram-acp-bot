from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(slots=True)
class WorkflowRun:
    database_id: int
    head_sha: str
    status: str
    conclusion: str
    url: str


@dataclass(slots=True)
class WaitRequest:
    repo: str
    workflow: str
    branch: str
    event: str
    commit: str
    timeout_seconds: int
    poll_interval_seconds: int


def _run_gh(*args: str) -> str:
    completed = subprocess.run(
        ["gh", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _load_runs(*, repo: str, workflow: str, branch: str, event: str, limit: int) -> list[WorkflowRun]:
    payload = _run_gh(
        "run",
        "list",
        "--repo",
        repo,
        "--workflow",
        workflow,
        "--branch",
        branch,
        "--event",
        event,
        "--limit",
        str(limit),
        "--json",
        "databaseId,headSha,status,conclusion,url",
    )
    runs = json.loads(payload)
    return [
        WorkflowRun(
            database_id=run["databaseId"],
            head_sha=run["headSha"],
            status=run["status"],
            conclusion=run["conclusion"] or "",
            url=run["url"],
        )
        for run in runs
    ]


def wait_for_green_ci(request: WaitRequest) -> int:
    deadline = time.monotonic() + request.timeout_seconds
    seen_run = False

    while time.monotonic() < deadline:
        for run in _load_runs(
            repo=request.repo,
            workflow=request.workflow,
            branch=request.branch,
            event=request.event,
            limit=20,
        ):
            if run.head_sha != request.commit:
                continue

            seen_run = True
            print(f"Found CI run for {request.commit[:7]}: status={run.status} conclusion={run.conclusion or '-'}")
            print(run.url)

            if run.status != "completed":
                break
            if run.conclusion == "success":
                return 0
            return 1

        if not seen_run:
            print(f"Waiting for CI run for {request.commit[:7]} on {request.branch}...")
        else:
            print(f"Waiting for CI completion for {request.commit[:7]}...")
        time.sleep(request.poll_interval_seconds)

    print(f"Timed out waiting for CI for {request.commit[:7]}.", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for a GitHub Actions workflow run on a specific commit to finish successfully."
    )
    parser.add_argument("--repo", required=True, help="Repository in OWNER/NAME form.")
    parser.add_argument("--workflow", required=True, help="Workflow name as shown in GitHub Actions.")
    parser.add_argument("--branch", default="main", help="Branch name to filter runs.")
    parser.add_argument("--event", default="push", help="GitHub event name to filter runs.")
    parser.add_argument("--commit", required=True, help="Commit SHA to watch.")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="Maximum time to wait.")
    parser.add_argument("--poll-interval-seconds", type=int, default=5, help="Polling interval.")
    args = parser.parse_args()

    return wait_for_green_ci(
        WaitRequest(
            repo=args.repo,
            workflow=args.workflow,
            branch=args.branch,
            event=args.event,
            commit=args.commit,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
