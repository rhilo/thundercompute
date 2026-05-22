#!/usr/bin/env python3
"""Intelligent dataset sanitizer: crop, blur-filter, and deduplicate training images."""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from pipeline_config import (
    PipelineConfigError,
    PipelineSettings,
    add_config_argument,
    load_pipeline_settings,
)

import cv2
import imagehash
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

TARGET_WIDTH = 768
TARGET_HEIGHT = 1152
TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT
JPEG_QUALITY = 95
FACE_SHIFT_RATIO = 0.15
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

console = Console(stderr=True)


class PreprocessError(Exception):
    """Raised when preprocessing cannot continue safely."""


def parse_args(argv: Sequence[str] | None = None) -> PipelineSettings:
    parser = argparse.ArgumentParser(
        description="Sanitize, crop, blur-filter, and deduplicate portrait training images.",
    )
    add_config_argument(parser)
    args = parser.parse_args(argv)
    return load_pipeline_settings(args.config)


def validate_inputs(zip_path: Path, output_dir: Path) -> None:
    if not zip_path.is_file():
        raise PreprocessError(f"Zip archive not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise PreprocessError(f"Path is not a valid zip archive: {zip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)


def extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)


def iter_image_paths(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.is_file():
        raise PreprocessError(f"OpenCV face cascade not found: {cascade_path}")
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise PreprocessError("Failed to load OpenCV frontal face cascade.")
    return detector


def detect_primary_face(gray: np.ndarray, detector: cv2.CascadeClassifier) -> tuple[int, int, int, int] | None:
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    areas = [int(w * h) for (_, _, w, h) in faces]
    index = int(np.argmax(areas))
    x, y, w, h = faces[index]
    return int(x), int(y), int(w), int(h)


def compute_crop_box(
    image_width: int,
    image_height: int,
    face: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    if image_width / image_height > TARGET_ASPECT:
        crop_height = image_height
        crop_width = int(round(crop_height * TARGET_ASPECT))
    else:
        crop_width = image_width
        crop_height = int(round(crop_width / TARGET_ASPECT))

    crop_width = min(crop_width, image_width)
    crop_height = min(crop_height, image_height)

    if face is not None:
        fx, fy, fw, fh = face
        center_x = fx + (fw / 2.0)
        center_y = fy + (fh / 2.0) + (FACE_SHIFT_RATIO * crop_height)
    else:
        center_x = image_width / 2.0
        center_y = image_height / 2.0

    left = int(round(center_x - (crop_width / 2.0)))
    top = int(round(center_y - (crop_height / 2.0)))

    left = max(0, min(left, image_width - crop_width))
    top = max(0, min(top, image_height - crop_height))
    right = left + crop_width
    bottom = top + crop_height
    return left, top, right, bottom


def laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_and_resize(image_bgr: np.ndarray, detector: cv2.CascadeClassifier) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face = detect_primary_face(gray, detector)
    left, top, right, bottom = compute_crop_box(image_bgr.shape[1], image_bgr.shape[0], face)
    cropped = image_bgr[top:bottom, left:right]
    return cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)


def write_jpeg(image_bgr: np.ndarray, destination: Path) -> None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    pil_image.save(destination, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def hamming_distance(hash_a: imagehash.ImageHash, hash_b: imagehash.ImageHash) -> int:
    return int(hash_a - hash_b)


def is_duplicate(candidate: imagehash.ImageHash, accepted: Iterable[imagehash.ImageHash], threshold: int) -> bool:
    for existing in accepted:
        if hamming_distance(candidate, existing) < threshold:
            return True
    return False


def process_images(
    source_paths: Sequence[Path],
    output_dir: Path,
    blur_threshold: float,
    hash_threshold: int,
) -> tuple[int, int, int]:
    detector = load_face_detector()
    accepted_hashes: list[imagehash.ImageHash] = []
    kept = 0
    rejected_blur = 0
    rejected_duplicate = 0
    index = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task("Processing images", total=len(source_paths))
        for source_path in source_paths:
            progress.advance(task_id)
            image_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue

            if laplacian_variance(image_bgr) < blur_threshold:
                rejected_blur += 1
                del image_bgr
                continue

            processed = crop_and_resize(image_bgr, detector)
            del image_bgr

            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            perceptual_hash = imagehash.phash(pil_image)

            if is_duplicate(perceptual_hash, accepted_hashes, hash_threshold):
                rejected_duplicate += 1
                del processed, pil_image
                continue

            accepted_hashes.append(perceptual_hash)
            index += 1
            output_name = f"img_{index:05d}.jpg"
            write_jpeg(processed, output_dir / output_name)
            kept += 1
            del processed, pil_image

    del detector
    gc.collect()
    return kept, rejected_blur, rejected_duplicate


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = parse_args(argv)
        validate_inputs(settings.raw_zip, settings.dataset_dir)
    except (PreprocessError, PipelineConfigError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    with tempfile.TemporaryDirectory(prefix="flux_preprocess_") as temp_dir:
        extract_root = Path(temp_dir) / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        extract_zip(settings.raw_zip, extract_root)
        source_paths = list(iter_image_paths(extract_root))
        if not source_paths:
            console.print("[red]Error:[/red] No supported images found inside zip archive.")
            return 1

        kept, rejected_blur, rejected_duplicate = process_images(
            source_paths,
            settings.dataset_dir,
            settings.blur_threshold,
            settings.hash_threshold,
        )

    if kept == 0:
        console.print("[red]Error:[/red] All images were rejected by quality or deduplication gates.")
        return 1

    console.print(
        f"[green]Done.[/green] Kept {kept} images. "
        f"Rejected blur: {rejected_blur}. Rejected duplicates: {rejected_duplicate}. "
        f"Output: {settings.dataset_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
