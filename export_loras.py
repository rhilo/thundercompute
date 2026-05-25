#!/usr/bin/env python3
"""Copy trained LoRA checkpoints from ai-toolkit output into export/loras for Drive push."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

from pipeline_config import (
    PipelineConfigError,
    PipelineSettings,
    add_config_argument,
    load_pipeline_settings,
)


def parse_args(argv: Sequence[str] | None = None) -> tuple[PipelineSettings, Path]:
    parser = argparse.ArgumentParser(
        description="Export .safetensors LoRA files from training_output_dir to export/loras.",
    )
    add_config_argument(parser)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("/home/ubuntu/export/loras"),
        help="Destination directory for Comfy-ready LoRA files.",
    )
    args = parser.parse_args(argv)
    return load_pipeline_settings(args.config), args.export_dir.expanduser()


def find_lora_files(training_dir: Path) -> list[Path]:
    patterns = ("*.safetensors",)
    files: list[Path] = []
    if not training_dir.is_dir():
        return files
    for pattern in patterns:
        files.extend(training_dir.rglob(pattern))
    # Prefer paths that look like LoRA exports, not full model shards.
    filtered = [
        path
        for path in files
        if "lora" in path.name.lower() or "lora" in str(path.parent).lower()
    ]
    return sorted(filtered or files)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings, export_dir = parse_args(argv)
        export_dir.mkdir(parents=True, exist_ok=True)
        loras = find_lora_files(settings.training_output_dir)
        if not loras:
            print(f"No .safetensors found under {settings.training_output_dir}")
            return 1

        copied = 0
        for source in loras:
            target = export_dir / source.name
            if target.exists() and target.stat().st_size == source.stat().st_size:
                continue
            shutil.copy2(source, target)
            print(f"Exported: {source} -> {target}")
            copied += 1

        print(f"Done. {copied} file(s) in {export_dir} (total candidates: {len(loras)}).")
        return 0
    except PipelineConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
