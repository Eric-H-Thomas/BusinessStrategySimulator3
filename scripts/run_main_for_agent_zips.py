"""Run the C++ simulator for every Agent.zip discovered under a folder tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find Agent.zip files, then run the simulator executable once "
            "per sibling config.json file."
        )
    )
    parser.add_argument(
        "search_dir",
        type=Path,
        help="Directory to recursively search for Agent.zip files.",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("cmake-build-debug/BusinessStrategySimulator3"),
        help=(
            "Path to the compiled simulator executable. "
            "Default: cmake-build-debug/BusinessStrategySimulator3"
        ),
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("WorkingFiles"),
        help=(
            "Working directory to launch the simulator from. "
            "Default: WorkingFiles"
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately after the first failed run.",
    )
    return parser.parse_args()


def find_agent_archives(search_dir: Path) -> list[Path]:
    return sorted(path for path in search_dir.rglob("Agent.zip") if path.is_file())


def run_simulation(
    executable: Path,
    config_path: Path,
    working_dir: Path,
) -> subprocess.CompletedProcess[str]:
    command = [str(executable.resolve()), str(config_path.resolve())]
    return subprocess.run(
        command,
        cwd=working_dir.resolve(),
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    args = parse_args()

    search_dir = args.search_dir.resolve()
    executable = args.executable.resolve()
    working_dir = args.working_dir.resolve()

    if not search_dir.exists() or not search_dir.is_dir():
        print(f"Search directory does not exist or is not a directory: {search_dir}", file=sys.stderr)
        return 1
    if not executable.exists() or not executable.is_file():
        print(f"Simulator executable not found: {executable}", file=sys.stderr)
        return 1
    if not working_dir.exists() or not working_dir.is_dir():
        print(f"Working directory not found: {working_dir}", file=sys.stderr)
        return 1

    archives = find_agent_archives(search_dir)
    if not archives:
        print(f"No Agent.zip files found under: {search_dir}")
        return 0

    failures = 0

    for archive_path in archives:
        run_dir = archive_path.parent
        config_path = run_dir / "config.json"

        if not config_path.exists():
            print(f"Skipping {archive_path}: missing {config_path.name}", file=sys.stderr)
            failures += 1
            if args.fail_fast:
                return 1
            continue

        print(f"Running simulator for: {archive_path}")
        result = run_simulation(executable=executable, config_path=config_path, working_dir=working_dir)

        if result.returncode != 0:
            failures += 1
            print(
                f"FAILED ({result.returncode}) for {archive_path}\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}",
                file=sys.stderr,
            )
            if args.fail_fast:
                return 1
            continue

        print(f"OK: output files were written by simulator using {config_path}")

    if failures > 0:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print(f"Completed successfully for {len(archives)} Agent.zip file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
