#!/usr/bin/env python3
"""JoyCaption Alpha Two captioning pipeline for FLUX LoRA training datasets."""

from __future__ import annotations

import argparse
import gc
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

import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from transformers import AutoTokenizer, LlavaForConditionalGeneration

MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
JOYCAPTION_IMAGE_SIZE = (384, 384)
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}

SYSTEM_PROMPT = (
    "You are a helpful image captioner. Write pure descriptive prose focusing on "
    "textiles, hair structures, stance, background environment, and lighting. "
    "Do not invent personal identification names."
)
USER_PROMPT = (
    "Write a long descriptive caption for this image in a formal tone. "
    "Describe only what is visible."
)

console = Console(stderr=True)


class CaptionError(Exception):
    """Raised when captioning cannot continue safely."""


def parse_args(argv: Sequence[str] | None = None) -> PipelineSettings:
    parser = argparse.ArgumentParser(
        description="Generate JoyCaption Alpha Two captions for a cleaned dataset.",
    )
    add_config_argument(parser)
    args = parser.parse_args(argv)
    settings = load_pipeline_settings(args.config)
    apply_runtime_env(settings)
    return settings


def validate_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.is_dir():
        raise CaptionError(f"Dataset directory not found: {dataset_dir}")


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise CaptionError("CUDA is required for JoyCaption inference but no GPU was detected.")
    return torch.device("cuda")


def list_images(dataset_dir: Path) -> list[Path]:
    images = [
        path
        for path in sorted(dataset_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not images:
        raise CaptionError(f"No supported images found in dataset: {dataset_dir}")
    return images


def load_model() -> tuple[AutoTokenizer, LlavaForConditionalGeneration]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    model.eval()
    return tokenizer, model


def preprocess_image(image_path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path)
    if image.size != JOYCAPTION_IMAGE_SIZE:
        image = image.resize(JOYCAPTION_IMAGE_SIZE, Image.LANCZOS)
    image = image.convert("RGB")
    pixel_values = TVF.pil_to_tensor(image)
    pixel_values = pixel_values / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
    return pixel_values.to(torch.bfloat16).unsqueeze(0).to(device)


def build_input_ids(
    tokenizer: AutoTokenizer,
    model: LlavaForConditionalGeneration,
) -> tuple[torch.Tensor, torch.Tensor]:
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    conversation_string = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    conversation_tokens = tokenizer.encode(
        conversation_string,
        add_special_tokens=False,
        truncation=False,
    )

    input_tokens: list[int] = []
    for token in conversation_tokens:
        if token == model.config.image_token_index:
            input_tokens.extend(
                [model.config.image_token_index] * model.config.image_seq_length
            )
        else:
            input_tokens.append(token)

    input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def generate_caption(
    model: LlavaForConditionalGeneration,
    tokenizer: AutoTokenizer,
    image_path: Path,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    pixel_values = preprocess_image(image_path, device)
    input_ids, attention_mask = build_input_ids(tokenizer, model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
        )[0]

    generated_ids = generated_ids[input_ids.shape[1] :]
    caption = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return caption.strip()


def format_caption(trigger_word: str, caption_body: str) -> str:
    cleaned = " ".join(caption_body.split())
    return f"{trigger_word}, {cleaned}"


def write_caption(image_path: Path, trigger_word: str, caption_body: str) -> Path:
    caption_path = image_path.with_suffix(".txt")
    caption_path.write_text(format_caption(trigger_word, caption_body) + "\n", encoding="utf-8")
    return caption_path


def caption_dataset(
    images: Sequence[Path],
    trigger_word: str,
    overwrite: bool,
    max_new_tokens: int,
) -> tuple[int, int]:
    device = require_cuda()
    tokenizer, model = load_model()
    written = 0
    skipped = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task("Captioning images", total=len(images))
        for image_path in images:
            progress.advance(task_id)
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists() and not overwrite:
                skipped += 1
                continue

            caption_body = generate_caption(
                model,
                tokenizer,
                image_path,
                device,
                max_new_tokens,
            )
            write_caption(image_path, trigger_word, caption_body)
            written += 1
            torch.cuda.empty_cache()
            gc.collect()

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return written, skipped


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = parse_args(argv)
        validate_dataset(settings.dataset_dir)
        images = list_images(settings.dataset_dir)
        written, skipped = caption_dataset(
            images,
            settings.trigger_word,
            settings.caption_overwrite,
            settings.caption_max_new_tokens,
        )
    except (CaptionError, PipelineConfigError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    console.print(
        f"[green]Done.[/green] Wrote {written} captions. Skipped existing: {skipped}. "
        f"Dataset: {settings.dataset_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
