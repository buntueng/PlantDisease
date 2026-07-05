#!/usr/bin/env python3
"""
run_baselines_sequence.py

Run baseline experiments sequentially, one model + experiment per fresh
Python process.

Examples
--------
Run all experiment families for all models:
    python3 run_baselines_sequence.py

Run only within-CV:
    python3 run_baselines_sequence.py --experiments within_cv

Run pairwise then multisource:
    python3 run_baselines_sequence.py --experiments pairwise multisource

Run selected models only:
    python3 run_baselines_sequence.py \
        --experiments within_cv \
        --models mobilenet_v2 efficientnet_b0 resnet50

Restart and skip jobs already completed successfully:
    python3 run_baselines_sequence.py --resume

Continue even if one job fails:
    python3 run_baselines_sequence.py --continue-on-error
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SUPPORTED_EXPERIMENTS = (
    "within_cv",
    "pairwise",
    "multisource",
    "pooled_cv",
)

DEFAULT_MODELS = (
    "mobilenet_v2",
    "mobilenet_v3_small",
    "shufflenet_v2_x1_0",
    "efficientnet_b0",
    "efficientnet_b4",
    "resnet50",
    "densenet121",
    "swin_t",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run run_baselines.py sequentially. Each model + experiment "
            "combination is launched in a fresh Python process."
        )
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=SUPPORTED_EXPERIMENTS,
        default=list(SUPPORTED_EXPERIMENTS),
        help=(
            "Experiment families to run, in the specified order. "
            f"Default: {' '.join(SUPPORTED_EXPERIMENTS)}"
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Models to run, in the specified order.",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file passed to run_baselines.py.",
    )

    parser.add_argument(
        "--runner",
        default="run_baselines.py",
        help="Path to run_baselines.py.",
    )

    parser.add_argument(
        "--python",
        default="python3",
        help="Python executable. Default: python3",
    )

    parser.add_argument(
        "--order",
        choices=("model-first", "experiment-first"),
        default="model-first",
        help=(
            "model-first: run every experiment for model 1, then model 2. "
            "experiment-first: run every model for experiment 1, then experiment 2."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs with an existing successful .done marker.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next job if a job fails.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print commands without training. Note: this launcher dry-run "
            "does not invoke run_baselines.py."
        ),
    )

    parser.add_argument(
        "--runner-dry-run",
        action="store_true",
        help="Invoke run_baselines.py with its --dry-run option.",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Pass --no-pretrained to run_baselines.py.",
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help=(
            "Pass --rebuild-index only to the first launched job. "
            "Later jobs reuse the rebuilt index."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional dataset IDs for within_cv only, e.g. --datasets A B C. "
            "Ignored for other experiment families."
        ),
    )

    parser.add_argument(
        "--log-dir",
        default="logs/baselines_sequence",
        help="Root directory for per-job log files.",
    )

    parser.add_argument(
        "--status-dir",
        default=".batch_status/baselines_sequence",
        help="Directory for .done completion markers.",
    )

    return parser.parse_args()


def build_jobs(
    models: list[str],
    experiments: list[str],
    order: str,
) -> list[tuple[str, str]]:
    if order == "model-first":
        return [
            (model, experiment)
            for model in models
            for experiment in experiments
        ]

    return [
        (model, experiment)
        for experiment in experiments
        for model in models
    ]


def safe_name(text: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in text
    )


def print_command(cmd: list[str]) -> None:
    print("Command:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))


def main() -> int:
    args = parse_args()

    project_dir = Path(__file__).resolve().parent
    runner = Path(args.runner)
    if not runner.is_absolute():
        runner = project_dir / runner
    runner = runner.resolve()

    config = Path(args.config)
    if not config.is_absolute():
        config = project_dir / config
    config = config.resolve()

    if not runner.exists():
        print(f"ERROR: runner not found: {runner}", file=sys.stderr)
        return 2

    if not config.exists():
        print(f"ERROR: config not found: {config}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_root = Path(args.log_dir)
    if not log_root.is_absolute():
        log_root = project_dir / log_root
    run_log_dir = log_root / timestamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    status_dir = Path(args.status_dir)
    if not status_dir.is_absolute():
        status_dir = project_dir / status_dir
    status_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(
        models=list(args.models),
        experiments=list(args.experiments),
        order=args.order,
    )

    print("=" * 72)
    print("Sequential baseline launcher")
    print("=" * 72)
    print(f"Project directory : {project_dir}")
    print(f"Runner            : {runner}")
    print(f"Config            : {config}")
    print(f"Python            : {args.python}")
    print(f"Order             : {args.order}")
    print(f"Experiments       : {', '.join(args.experiments)}")
    print(f"Models            : {', '.join(args.models)}")
    print(f"Number of jobs    : {len(jobs)}")
    print(f"Log directory     : {run_log_dir}")
    print(f"Status directory  : {status_dir}")
    print("=" * 72)

    completed = 0
    skipped = 0
    failed = 0
    launched = 0

    for job_index, (model, experiment) in enumerate(jobs, start=1):
        job_name = f"{safe_name(model)}__{safe_name(experiment)}"
        done_marker = status_dir / f"{job_name}.done"
        log_file = run_log_dir / f"{job_name}.log"

        print()
        print("=" * 72)
        print(f"[{job_index}/{len(jobs)}] Model={model} | Experiment={experiment}")
        print("=" * 72)

        if args.resume and done_marker.exists():
            print(f"SKIP: completed marker exists: {done_marker}")
            skipped += 1
            continue

        cmd = [
            args.python,
            str(runner),
            "--config",
            str(config),
            "--experiment",
            experiment,
            "--models",
            model,
        ]

        if args.no_pretrained:
            cmd.append("--no-pretrained")

        if args.runner_dry_run:
            cmd.append("--dry-run")

        # Rebuild the training index only once, on the first actually launched job.
        if args.rebuild_index and launched == 0:
            cmd.append("--rebuild-index")

        # --datasets only applies to within_cv in run_baselines.py.
        if args.datasets and experiment == "within_cv":
            cmd.extend(["--datasets", *args.datasets])

        print_command(cmd)
        print(f"Log: {log_file}")

        if args.dry_run:
            print("LAUNCHER DRY-RUN: command not executed.")
            continue

        launched += 1
        start_time = datetime.now()

        # Use a fresh process for every job.
        # stdout/stderr are streamed to terminal and written to a log file.
        with log_file.open("w", encoding="utf-8", buffering=1) as log:
            log.write(f"Start: {start_time.isoformat()}\n")
            log.write(
                "Command: "
                + " ".join(shlex.quote(part) for part in cmd)
                + "\n\n"
            )
            log.flush()

            process = subprocess.Popen(
                cmd,
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )

            assert process.stdout is not None

            try:
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                    log.flush()
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Terminating current job...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                print(f"Stopped. Log preserved at: {log_file}")
                return 130

            return_code = process.wait()

            end_time = datetime.now()
            elapsed = end_time - start_time

            log.write("\n")
            log.write(f"End: {end_time.isoformat()}\n")
            log.write(f"Elapsed: {elapsed}\n")
            log.write(f"Return code: {return_code}\n")

        if return_code == 0:
            done_marker.write_text(
                "\n".join(
                    [
                        f"model={model}",
                        f"experiment={experiment}",
                        f"completed_at={datetime.now().isoformat()}",
                        f"log={log_file}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            completed += 1
            print(f"SUCCESS: {job_name}")
            print(f"Done marker: {done_marker}")
        else:
            failed += 1
            print(
                f"FAILED: {job_name} returned exit code {return_code}",
                file=sys.stderr,
            )
            print(f"See log: {log_file}", file=sys.stderr)

            if not args.continue_on_error:
                print(
                    "Stopping sequence. Re-run with --continue-on-error "
                    "to continue after failures.",
                    file=sys.stderr,
                )
                return return_code if return_code != 0 else 1

    print()
    print("=" * 72)
    print("Sequence finished")
    print("=" * 72)
    print(f"Successful jobs : {completed}")
    print(f"Skipped jobs    : {skipped}")
    print(f"Failed jobs     : {failed}")
    print(f"Logs            : {run_log_dir}")
    print("=" * 72)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
