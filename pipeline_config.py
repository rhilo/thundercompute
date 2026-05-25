#!/usr/bin/env python3
"""Load and apply settings from pipeline.yaml for all pipeline stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAME = "pipeline.yaml"

INSTANCE_PRESETS: dict[str, dict[str, bool]] = {
    "a100_80gb": {
        "quantize": False,
        "gradient_checkpointing": False,
        "low_vram": False,
    },
    "standard_24gb": {
        "quantize": True,
        "gradient_checkpointing": True,
        "low_vram": False,
    },
}


class PipelineConfigError(Exception):
    """Raised when pipeline.yaml is missing or invalid."""


@dataclass(frozen=True)
class PipelineSettings:
    config_path: Path
    hf_token: str | None
    hf_token_env: str
    hf_flux_local_dir: Path | None
    hf_hub_offline: bool
    trigger_word: str
    raw_zip: Path
    dataset_dir: Path
    training_output_dir: Path
    ai_toolkit_config_out: Path
    blur_threshold: float
    hash_threshold: int
    caption_backend: str
    caption_attn: str
    caption_batch_size: int
    caption_num_workers: int
    caption_prefetch_factor: int
    caption_overwrite: bool
    caption_max_new_tokens: int
    drive_rclone_remote: str
    drive_export_loras_dir: Path
    drive_comfy_output_dir: Path
    job_name: str
    training_steps: int
    model_id: str | None
    model_quantize: bool
    model_low_vram: bool
    train_gradient_checkpointing: bool
    disable_otel: bool
    wandb_mode: str


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    section = raw.get(key)
    if not isinstance(section, dict):
        raise PipelineConfigError(f"Missing or invalid '{key}:' section in pipeline config.")
    return section


def _resolve_config_path(config_path: Path | None) -> Path:
    if config_path is None:
        return Path(DEFAULT_CONFIG_NAME).resolve()
    return config_path.expanduser().resolve()


def load_pipeline_settings(config_path: Path | None = None) -> PipelineSettings:
    resolved = _resolve_config_path(config_path)
    if not resolved.is_file():
        raise PipelineConfigError(
            f"Pipeline config not found: {resolved}\n"
            f"Copy pipeline.example.yaml to {DEFAULT_CONFIG_NAME} and edit it."
        )

    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise PipelineConfigError(f"Pipeline config must be a YAML mapping: {resolved}")

    huggingface = _require_mapping(raw, "huggingface")
    project = _require_mapping(raw, "project")
    paths = _require_mapping(raw, "paths")
    preprocess = _require_mapping(raw, "preprocess")
    caption = _require_mapping(raw, "caption")
    training = _require_mapping(raw, "training")
    instance = raw.get("instance", {})
    if instance is not None and not isinstance(instance, dict):
        raise PipelineConfigError("'instance:' must be a mapping when present.")

    profile_name = str((instance or {}).get("profile", "")).strip()
    preset = INSTANCE_PRESETS.get(profile_name, {})

    def instance_bool(key: str, default: bool) -> bool:
        if instance is not None and key in instance:
            return bool(instance[key])
        if key in preset:
            return bool(preset[key])
        return default

    runtime = raw.get("runtime", {})
    if runtime is not None and not isinstance(runtime, dict):
        raise PipelineConfigError("'runtime:' must be a mapping when present.")

    drive = raw.get("drive", {})
    if drive is not None and not isinstance(drive, dict):
        raise PipelineConfigError("'drive:' must be a mapping when present.")

    trigger_word = str(project.get("trigger_word", "")).strip()
    if not trigger_word:
        raise PipelineConfigError("'project.trigger_word' is required.")

    token_raw = huggingface.get("token")
    hf_token = str(token_raw).strip() if token_raw else None
    hf_token_env = str(huggingface.get("token_env", "HF_TOKEN")).strip() or "HF_TOKEN"

    flux_dir_raw = huggingface.get("flux_local_dir")
    hf_flux_local_dir = Path(flux_dir_raw).expanduser() if flux_dir_raw else None

    return PipelineSettings(
        config_path=resolved,
        hf_token=hf_token,
        hf_token_env=hf_token_env,
        hf_flux_local_dir=hf_flux_local_dir,
        hf_hub_offline=bool(huggingface.get("hub_offline_during_train", True)),
        trigger_word=trigger_word,
        raw_zip=Path(str(paths["raw_zip"])).expanduser(),
        dataset_dir=Path(str(paths["dataset_dir"])).expanduser(),
        training_output_dir=Path(str(paths["training_output_dir"])).expanduser(),
        ai_toolkit_config_out=Path(str(paths["ai_toolkit_config_out"])).expanduser(),
        blur_threshold=float(preprocess.get("blur_threshold", 80.0)),
        hash_threshold=int(preprocess.get("hash_threshold", 5)),
        caption_backend=str(caption.get("backend", "batch")).strip().lower(),
        caption_attn=str(caption.get("attn", "auto")).strip().lower(),
        caption_batch_size=int(caption.get("batch_size", 32)),
        caption_num_workers=int(caption.get("num_workers", 6)),
        caption_prefetch_factor=int(caption.get("prefetch_factor", 2)),
        caption_overwrite=bool(caption.get("overwrite", False)),
        caption_max_new_tokens=int(caption.get("max_new_tokens", 300)),
        drive_rclone_remote=str((drive or {}).get("rclone_remote", "gdrive")).strip() or "gdrive",
        drive_export_loras_dir=Path(
            str((drive or {}).get("export_loras_dir", "/home/ubuntu/export/loras"))
        ).expanduser(),
        drive_comfy_output_dir=Path(
            str((drive or {}).get("comfy_output_dir", "/home/ubuntu/ComfyUI/output"))
        ).expanduser(),
        job_name=str(training.get("job_name", "flux_lora_v1")).strip(),
        training_steps=int(training.get("steps", 2000)),
        model_id=(
            str(training["model_id"]).strip()
            if training.get("model_id") not in (None, "")
            else None
        ),
        model_quantize=instance_bool("quantize", True),
        model_low_vram=instance_bool("low_vram", False),
        train_gradient_checkpointing=instance_bool("gradient_checkpointing", True),
        disable_otel=bool((runtime or {}).get("disable_otel", True)),
        wandb_mode=str((runtime or {}).get("wandb_mode", "disabled")),
    )


def apply_runtime_env(settings: PipelineSettings) -> None:
    """Export Hugging Face token and training runtime environment variables."""
    token = settings.hf_token
    if not token:
        token = os.environ.get(settings.hf_token_env)

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    if settings.hf_hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    os.environ["WANDB_MODE"] = settings.wandb_mode

    if settings.disable_otel:
        os.environ["OTEL_SDK_DISABLED"] = "true"
        os.environ["OTEL_TRACES_EXPORTER"] = "none"
        os.environ["OTEL_METRICS_EXPORTER"] = "none"
        os.environ["OTEL_LOGS_EXPORTER"] = "none"


def add_config_argument(parser: Any, default_name: str = DEFAULT_CONFIG_NAME) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(default_name),
        help=f"Pipeline settings file (default: {default_name}).",
    )


def resolve_settings_from_args(args: Any) -> PipelineSettings:
    settings = load_pipeline_settings(args.config)
    apply_runtime_env(settings)
    return settings
