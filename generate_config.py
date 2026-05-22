#!/usr/bin/env python3
"""Synthesize ai-toolkit YAML training configuration for FLUX.1-dev LoRA runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml
from pipeline_config import (
    PipelineConfigError,
    PipelineSettings,
    add_config_argument,
    load_pipeline_settings,
)
from rich.console import Console

HF_FLUX_DEV_REPO = "black-forest-labs/FLUX.1-dev"

DIFFUSERS_DIR_CANDIDATES: tuple[Path, ...] = (
    Path("/home/ubuntu/FLUX.1-dev"),
    Path("/home/ubuntu/flux1-dev"),
)

HF_SNAPSHOTS_ROOT = (
    Path.home() / ".cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots"
)

COMFY_SINGLE_FILE_PATHS: tuple[Path, ...] = (
    Path("/home/ubuntu/flux1-dev.safetensors"),
    Path.home() / "ComfyUI/models/unet/flux1-dev.safetensors",
    Path.home() / "ComfyUI/models/diffusion_models/flux1-dev.safetensors",
    Path.home() / "ComfyUI/models/checkpoints/flux1-dev.safetensors",
)

PORTRAIT_RESOLUTIONS = [768, 1152]
TARGET_SAMPLE_WIDTH = 768
TARGET_SAMPLE_HEIGHT = 1152

console = Console(stderr=True)


class ConfigError(Exception):
    """Raised when configuration synthesis cannot continue safely."""


def parse_args(argv: Sequence[str] | None = None) -> PipelineSettings:
    parser = argparse.ArgumentParser(
        description="Generate an ai-toolkit YAML config for FLUX.1-dev LoRA training.",
    )
    add_config_argument(parser)
    args = parser.parse_args(argv)
    return load_pipeline_settings(args.config)


def validate_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.is_dir():
        raise ConfigError(f"Dataset directory not found: {dataset_dir}")

    image_suffixes = {".jpg", ".jpeg", ".png"}
    has_images = any(
        path.is_file() and path.suffix.lower() in image_suffixes
        for path in dataset_dir.iterdir()
    )
    if not has_images:
        raise ConfigError(f"No training images found in dataset directory: {dataset_dir}")


def is_diffusers_flux_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "model_index.json").is_file():
        return True
    return (path / "transformer" / "config.json").is_file()


def find_local_diffusers_dir(settings: PipelineSettings | None = None) -> Path | None:
    candidates: list[Path] = []
    if settings is not None and settings.hf_flux_local_dir is not None:
        candidates.append(settings.hf_flux_local_dir)
    candidates.extend(DIFFUSERS_DIR_CANDIDATES)

    for candidate in candidates:
        if is_diffusers_flux_dir(candidate):
            return candidate

    if HF_SNAPSHOTS_ROOT.is_dir():
        for snapshot in sorted(HF_SNAPSHOTS_ROOT.iterdir(), reverse=True):
            if is_diffusers_flux_dir(snapshot):
                return snapshot
    return None


def find_comfy_safetensors() -> Path | None:
    for candidate in COMFY_SINGLE_FILE_PATHS:
        if candidate.is_file():
            return candidate
    return None


def resolve_model_reference(
    model_id_override: str | None,
    settings: PipelineSettings | None = None,
) -> str:
    if model_id_override:
        override = model_id_override.strip()
        if override.endswith(".safetensors"):
            raise ConfigError(
                f"training.model_id cannot be a single safetensors file: {override}\n"
                "Use a diffusers directory (e.g. /home/ubuntu/FLUX.1-dev) or "
                f"{HF_FLUX_DEV_REPO}."
            )
        return override

    local_dir = find_local_diffusers_dir(settings)
    if local_dir is not None:
        console.print(
            f"[green]Using local diffusers FLUX.1-dev folder:[/green] {local_dir}"
        )
        return str(local_dir.resolve())

    comfy_weights = find_comfy_safetensors()
    if comfy_weights is not None:
        console.print(
            "[yellow]Found ComfyUI single-file weights at[/yellow] "
            f"{comfy_weights}\n"
            "[yellow]ai-toolkit cannot train from one .safetensors file.[/yellow] "
            f"Using Hugging Face repo [bold]{HF_FLUX_DEV_REPO}[/bold] for "
            "transformer, VAE, and text encoders.\n"
            "For offline training, download the full diffusers tree:\n"
            "  hf auth login\n"
            "  hf download black-forest-labs/FLUX.1-dev "
            "--local-dir /home/ubuntu/FLUX.1-dev"
        )
    else:
        console.print(
            f"[yellow]No local diffusers folder found.[/yellow] "
            f"Using Hugging Face repo [bold]{HF_FLUX_DEV_REPO}[/bold] "
            "(requires network and Hugging Face access on first run)."
        )

    return HF_FLUX_DEV_REPO


def build_config(settings: PipelineSettings, model_reference: str) -> dict[str, Any]:
    dataset_dir = settings.dataset_dir
    training_folder = settings.training_output_dir
    trigger_word = settings.trigger_word
    job_name = settings.job_name
    steps = settings.training_steps
    dataset_path = str(dataset_dir.resolve())
    training_path = str(training_folder.resolve())
    sample_prompt = "[trigger] portrait photograph, natural lighting, detailed textiles and hair"

    return {
        "job": "extension",
        "config": {
            "name": job_name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": training_path,
                    "device": "cuda:0",
                    "trigger_word": trigger_word,
                    "network": {
                        "type": "lora",
                        "linear": 16,
                        "linear_alpha": 16,
                    },
                    "save": {
                        "dtype": "float16",
                        "save_every": 250,
                        "max_step_saves_to_keep": 4,
                        "push_to_hub": False,
                    },
                    "logging": {
                        "use_wandb": False,
                        "use_ui_logger": False,
                    },
                    "datasets": [
                        {
                            "folder_path": dataset_path,
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": PORTRAIT_RESOLUTIONS,
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": steps,
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": settings.train_gradient_checkpointing,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": 1e-4,
                        "ema_config": {
                            "use_ema": True,
                            "ema_decay": 0.99,
                        },
                        "dtype": "bf16",
                    },
                    "model": {
                        "name_or_path": model_reference,
                        "is_flux": True,
                        "quantize": settings.model_quantize,
                        **(
                            {"low_vram": True}
                            if settings.model_low_vram
                            else {}
                        ),
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": 250,
                        "width": TARGET_SAMPLE_WIDTH,
                        "height": TARGET_SAMPLE_HEIGHT,
                        "prompts": [sample_prompt],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 4,
                        "sample_steps": 20,
                    },
                }
            ],
        },
        "meta": {
            "name": "[name]",
            "version": "1.0",
        },
    }


def write_config(config: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = parse_args(argv)
        validate_dataset(settings.dataset_dir)
        model_reference = resolve_model_reference(settings.model_id, settings)
        config = build_config(settings, model_reference)
        write_config(config, settings.ai_toolkit_config_out)
    except (ConfigError, PipelineConfigError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    console.print(
        f"[green]Done.[/green] Wrote config to {settings.ai_toolkit_config_out}\n"
        f"  model.name_or_path = {model_reference}\n"
        f"  model.quantize = {settings.model_quantize}\n"
        f"  train.gradient_checkpointing = {settings.train_gradient_checkpointing}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
