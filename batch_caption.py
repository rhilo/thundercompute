#!/usr/bin/env python3
"""High-throughput batched JoyCaption Alpha Two captioning for large image directories."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Literal, Sequence

import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlavaForConditionalGeneration

from pipeline_config import (
    PipelineConfigError,
    PipelineSettings,
    add_config_argument,
    apply_runtime_env,
    load_pipeline_settings,
)

MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
JOYCAPTION_IMAGE_SIZE = (384, 384)
SUPPORTED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
}

SYSTEM_PROMPT = (
    "You are a helpful image captioner. Write pure descriptive prose focusing on "
    "textiles, hair structures, stance, background environment, and lighting. "
    "Do not invent personal identification names."
)
USER_PROMPT = (
    "Write a long descriptive caption for this image in a formal tone. "
    "Describe only what is visible."
)

DEFAULT_NUM_WORKERS = 6
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_BATCH_SIZE = 32

AttnChoice = Literal["auto", "sdpa", "flash_attention_2"]


class CaptionError(Exception):
    """Raised when captioning cannot continue safely."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-caption images with JoyCaption Alpha Two (bfloat16, DataLoader prefetch).",
    )
    add_config_argument(parser)
    parser.add_argument("--dataset", type=Path, default=None, help="Override paths.dataset_dir.")
    parser.add_argument("--trigger", type=str, default=None, help="Override project.trigger_word.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=f"Images per GPU forward pass (default from pipeline or {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=f"DataLoader workers (default {DEFAULT_NUM_WORKERS}).",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help=f"Prefetch batches per worker (default {DEFAULT_PREFETCH_FACTOR}).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Override caption.overwrite.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override caption.max_new_tokens.")
    parser.add_argument(
        "--attn",
        choices=("auto", "sdpa", "flash_attention_2"),
        default=None,
        help="Attention backend (default: pipeline caption.attn or auto).",
    )
    return parser.parse_args(argv)


def resolve_attn_implementation(choice: AttnChoice) -> str:
    if choice == "sdpa":
        return "sdpa"
    if choice == "flash_attention_2":
        return "flash_attention_2"
    import transformers

    if hasattr(transformers, "is_flash_attn_2_available") and transformers.is_flash_attn_2_available():
        return "flash_attention_2"
    return "sdpa"


def validate_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.is_dir():
        raise CaptionError(f"Dataset directory not found: {dataset_dir}")


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise CaptionError("CUDA is required for JoyCaption inference but no GPU was detected.")
    return torch.device("cuda")


def list_images(dataset_dir: Path) -> list[Path]:
    images = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not images:
        raise CaptionError(f"No supported images found under: {dataset_dir}")
    return images


def pending_images(images: Sequence[Path], overwrite: bool) -> list[Path]:
    if overwrite:
        return list(images)
    return [path for path in images if not path.with_suffix(".txt").exists()]


def load_model(attn_implementation: str) -> tuple[AutoTokenizer, LlavaForConditionalGeneration]:
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map=0,
        )
    except Exception as exc:
        if attn_implementation == "flash_attention_2":
            raise CaptionError(
                "flash_attention_2 failed to load. Run post-setup.py, use --attn sdpa, "
                "or install a flash-attn wheel from Drive venv/wheels/."
            ) from exc
        raise CaptionError(f"Failed to load model with attn={attn_implementation}: {exc}") from exc
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model.eval()
    return tokenizer, model


def preprocess_image_cpu(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    if image.size != JOYCAPTION_IMAGE_SIZE:
        image = image.resize(JOYCAPTION_IMAGE_SIZE, Image.LANCZOS)
    pixel_values = TVF.pil_to_tensor(image)
    pixel_values = pixel_values / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
    return pixel_values.to(torch.bfloat16)


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


class ImageCaptionDataset(Dataset[tuple[Path, torch.Tensor]]):
    def __init__(self, image_paths: Sequence[Path]) -> None:
        self.image_paths = list(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Path, torch.Tensor]:
        path = self.image_paths[index]
        return path, preprocess_image_cpu(path)


def collate_batch(
    batch: list[tuple[Path, torch.Tensor]],
) -> tuple[list[Path], torch.Tensor]:
    paths, pixel_tensors = zip(*batch)
    return list(paths), torch.stack(pixel_tensors, dim=0)


def generate_captions_batch(
    model: LlavaForConditionalGeneration,
    tokenizer: AutoTokenizer,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    batch_size = pixel_values.shape[0]
    prompt_len = input_ids.shape[1]

    input_ids = input_ids.expand(batch_size, -1).to(device)
    attention_mask = attention_mask.expand(batch_size, -1).to(device)
    pixel_values = pixel_values.to(device, non_blocking=True)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
        )

    captions: list[str] = []
    for row in range(batch_size):
        new_tokens = generated_ids[row, prompt_len:]
        caption = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        captions.append(caption.strip())
    return captions


def format_caption(trigger_word: str, caption_body: str) -> str:
    cleaned = " ".join(caption_body.split())
    return f"{trigger_word}, {cleaned}"


def write_caption(image_path: Path, trigger_word: str, caption_body: str) -> None:
    caption_path = image_path.with_suffix(".txt")
    caption_path.write_text(format_caption(trigger_word, caption_body) + "\n", encoding="utf-8")


def build_dataloader(
    image_paths: Sequence[Path],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader[tuple[list[Path], torch.Tensor]]:
    dataset = ImageCaptionDataset(image_paths)
    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_batch,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def caption_dataset(
    images: Sequence[Path],
    trigger_word: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    max_new_tokens: int,
    attn: AttnChoice,
) -> int:
    if batch_size < 1:
        raise CaptionError("--batch_size must be at least 1.")
    if num_workers < 0:
        raise CaptionError("--num_workers must be non-negative.")

    attn_impl = resolve_attn_implementation(attn)
    print(f"[batch_caption] Using attention implementation: {attn_impl}")

    device = require_cuda()
    tokenizer, model = load_model(attn_impl)
    input_ids, attention_mask = build_input_ids(tokenizer, model)

    dataloader = build_dataloader(images, batch_size, num_workers, prefetch_factor)
    written = 0

    with tqdm(total=len(images), unit="img", desc="Captioning", dynamic_ncols=True) as progress:
        for paths, pixel_values in dataloader:
            captions = generate_captions_batch(
                model,
                tokenizer,
                pixel_values,
                input_ids,
                attention_mask,
                device,
                max_new_tokens,
            )
            for image_path, caption_body in zip(paths, captions, strict=True):
                write_caption(image_path, trigger_word, caption_body)
            batch_count = len(paths)
            written += batch_count
            progress.update(batch_count)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return written


def settings_from_args(args: argparse.Namespace) -> tuple[PipelineSettings, argparse.Namespace]:
    settings = load_pipeline_settings(args.config)
    apply_runtime_env(settings)
    return settings, args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings, args = settings_from_args(args)

        dataset_dir = args.dataset or settings.dataset_dir
        trigger_word = (args.trigger or settings.trigger_word).strip()
        batch_size = args.batch_size if args.batch_size is not None else settings.caption_batch_size
        num_workers = args.num_workers if args.num_workers is not None else settings.caption_num_workers
        prefetch_factor = (
            args.prefetch_factor if args.prefetch_factor is not None else settings.caption_prefetch_factor
        )
        overwrite = args.overwrite or settings.caption_overwrite
        max_new_tokens = (
            args.max_new_tokens if args.max_new_tokens is not None else settings.caption_max_new_tokens
        )
        attn: AttnChoice = (args.attn or settings.caption_attn)  # type: ignore[assignment]

        validate_dataset(dataset_dir)
        if prefetch_factor < 1 and num_workers > 0:
            raise CaptionError("--prefetch_factor must be at least 1 when using workers.")

        all_images = list_images(dataset_dir)
        to_process = pending_images(all_images, overwrite)
        skipped = len(all_images) - len(to_process)

        if not to_process:
            print(
                f"Nothing to do: all {len(all_images)} images already have captions "
                f"(use --overwrite to regenerate)."
            )
            return 0

        written = caption_dataset(
            to_process,
            trigger_word,
            batch_size,
            num_workers,
            prefetch_factor,
            max_new_tokens,
            attn,
        )
    except (CaptionError, PipelineConfigError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Done. Wrote {written} captions. Skipped existing: {skipped}. "
        f"Dataset: {args.dataset or settings.dataset_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
