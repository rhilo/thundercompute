#!/usr/bin/env python3
"""Run FLUX LoRA pipeline stages using pipeline.yaml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from pipeline_config import (
    PipelineConfigError,
    PipelineSettings,
    add_config_argument,
    apply_runtime_env,
    load_pipeline_settings,
)
from rich.console import Console

console = Console(stderr=True)

STAGES: tuple[str, ...] = ("preprocess", "caption", "generate_config", "train")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pipeline stages defined in pipeline.yaml.",
    )
    add_config_argument(parser)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"Comma-separated stages to run (default: all). Choices: {', '.join(STAGES)}",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        type=str,
        default=None,
        choices=STAGES,
        help="Start at this stage and run through the end.",
    )
    return parser.parse_args(argv)


def resolve_stage_list(args: argparse.Namespace) -> list[str]:
    if args.only:
        selected = [part.strip() for part in args.only.split(",") if part.strip()]
        unknown = [stage for stage in selected if stage not in STAGES]
        if unknown:
            raise PipelineConfigError(f"Unknown stages: {', '.join(unknown)}")
        return selected

    if args.from_stage:
        start_index = STAGES.index(args.from_stage)
        return list(STAGES[start_index:])

    return list(STAGES)


def run_python_stage(script_name: str, settings: PipelineSettings) -> None:
    script_path = settings.config_path.parent / script_name
    command = [sys.executable, str(script_path), "--config", str(settings.config_path)]
    console.print(f"[cyan]>>[/cyan] {' '.join(command)}")
    subprocess.run(command, check=True, cwd=settings.config_path.parent)


def run_train_stage(settings: PipelineSettings) -> None:
    train_script = settings.config_path.parent / "train.sh"
    command = ["bash", str(train_script), "--config", str(settings.config_path)]
    console.print(f"[cyan]>>[/cyan] {' '.join(command)}")
    subprocess.run(command, check=True, cwd=settings.config_path.parent)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = load_pipeline_settings(args.config)
        apply_runtime_env(settings)
        stages = resolve_stage_list(args)
    except PipelineConfigError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    console.print(f"[green]Using config:[/green] {settings.config_path}")
    console.print(f"[green]Stages:[/green] {', '.join(stages)}")

    try:
        for stage in stages:
            if stage == "preprocess":
                run_python_stage("preprocess.py", settings)
            elif stage == "caption":
                run_python_stage("caption.py", settings)
            elif stage == "generate_config":
                run_python_stage("generate_config.py", settings)
            elif stage == "train":
                run_train_stage(settings)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Stage failed with exit code {exc.returncode}[/red]")
        return exc.returncode or 1

    console.print("[green]Pipeline finished.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
