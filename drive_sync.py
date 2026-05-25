#!/usr/bin/env python3
"""Pull/push files under artificial_generation/image/thunder_compute on Google Drive via rclone."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "drive.manifest.yaml"
ARCHIVE_GUARD = "archive"
TRAINING_PULL_IDS = "input,flux,venv"
COMFY_PULL_IDS = "unet,clip,vae,loras"


class DriveSyncError(Exception):
    """Raised when drive sync cannot continue safely."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync thunder_compute folder on Google Drive (rclone). Never touches archive/.",
    )
    parser.add_argument(
        "command",
        choices=("pull", "push", "status", "check", "promote-loras"),
        help="pull/push profile assets; check rclone; promote output/loras to models/loras on Drive",
    )
    parser.add_argument(
        "--profile",
        choices=("training", "comfyui"),
        default="training",
        help="Instance profile (default: training).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated manifest entry ids (e.g. input,flux,loras).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Sync manifest YAML (default: {DEFAULT_MANIFEST.name}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to rclone.")
    parser.add_argument("--force", action="store_true", help="Ignore skip_if_exists on pull.")
    return parser.parse_args(argv)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DriveSyncError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise DriveSyncError(f"Manifest must be a YAML mapping: {path}")
    return raw


def guard_path(relative: str, label: str) -> None:
    normalized = relative.replace("\\", "/").strip("/")
    if ARCHIVE_GUARD in normalized.split("/"):
        raise DriveSyncError(f"{label} must not reference '{ARCHIVE_GUARD}/': {relative}")


def remote_url(manifest: dict[str, Any], relative_drive: str) -> str:
    remote = str(manifest["rclone_remote"]).strip()
    root = str(manifest["drive_root"]).strip().strip("/")
    guard_path(root, "drive_root")
    rel = relative_drive.strip().strip("/")
    guard_path(rel, "drive path")
    return f"{remote}:{root}/{rel}" if rel else f"{remote}:{root}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def local_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    return any(path.iterdir())


def run_rclone(args: list[str], *, dry_run: bool) -> None:
    cmd = ["rclone", *args]
    if dry_run:
        cmd.append("--dry-run")
    print(f"[drive] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def rclone_copy_dir(
    manifest: dict[str, Any],
    drive_rel: str,
    local: Path,
    *,
    dry_run: bool,
    to_remote: bool,
) -> None:
    url = remote_url(manifest, drive_rel)
    local.mkdir(parents=True, exist_ok=True)
    if to_remote:
        run_rclone(["copy", str(local), url, "--create-empty-src-dirs"], dry_run=dry_run)
    else:
        run_rclone(["copy", url, str(local), "--create-empty-src-dirs"], dry_run=dry_run)


def sync_entry(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    *,
    to_remote: bool,
    dry_run: bool,
    force: bool,
) -> None:
    entry_id = str(entry.get("id", "unknown"))
    drive_rel = str(entry["drive"])
    local = Path(str(entry["local"])).expanduser()
    optional = bool(entry.get("optional", False))
    skip_if_exists = bool(entry.get("skip_if_exists", False)) and not force

    if to_remote:
        if not local.exists():
            if optional:
                print(f"[drive] Skip push {entry_id}: local missing {local}")
                return
            raise DriveSyncError(f"Push {entry_id}: local path not found: {local}")
        print(f"[drive] Push {entry_id}: {local} -> {drive_rel}")
        rclone_copy_dir(manifest, drive_rel, local, dry_run=dry_run, to_remote=True)
        return

    if skip_if_exists and local_nonempty(local):
        print(f"[drive] Skip pull {entry_id}: already present at {local}")
        return

    print(f"[drive] Pull {entry_id}: {drive_rel} -> {local}")
    try:
        rclone_copy_dir(manifest, drive_rel, local, dry_run=dry_run, to_remote=False)
    except subprocess.CalledProcessError as exc:
        if optional:
            print(f"[drive] Optional pull {entry_id} failed (exit {exc.returncode}); continuing.")
            return
        raise


def filter_entries(entries: list[dict[str, Any]], only: str | None) -> list[dict[str, Any]]:
    if not only:
        return entries
    allowed = {part.strip() for part in only.split(",") if part.strip()}
    return [entry for entry in entries if str(entry.get("id")) in allowed]


def cmd_check(manifest: dict[str, Any]) -> None:
    remote = str(manifest["rclone_remote"]).strip()
    try:
        subprocess.run(
            ["rclone", "lsd", f"{remote}:"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise DriveSyncError(
            f"rclone cannot access remote '{remote}'. Run: rclone config\n{exc.stderr or exc}"
        ) from exc
    print(f"[drive] rclone remote OK: {remote}")


def cmd_status(manifest: dict[str, Any], profile: str, only: str | None) -> None:
    profile_data = manifest["profiles"][profile]
    for direction in ("pull", "push"):
        entries = filter_entries(profile_data.get(direction, []), only)
        for entry in entries:
            drive_rel = str(entry["drive"])
            url = remote_url(manifest, drive_rel)
            result = subprocess.run(
                ["rclone", "size", url, "--json"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                print(
                    f"[drive] {profile}/{direction}/{entry['id']}: "
                    f"{info.get('count', '?')} objects, {info.get('bytes', '?')} bytes on Drive"
                )
            else:
                print(f"[drive] {profile}/{direction}/{entry['id']}: (not found or empty on Drive)")


def cmd_promote_loras(manifest: dict[str, Any], *, dry_run: bool) -> None:
    src = remote_url(manifest, "output/loras")
    dst = remote_url(manifest, "models/loras")
    run_rclone(["copy", src, dst, "--create-empty-src-dirs"], dry_run=dry_run)
    print("[drive] Promoted Drive output/loras -> models/loras (for ComfyUI pulls).")


def install_wheels_from_staging(venv_staging: Path) -> int:
    wheels_dir = venv_staging / "wheels"
    if not wheels_dir.is_dir():
        return 0
    wheels = sorted(wheels_dir.glob("flash_attn*.whl"))
    if not wheels:
        return 0
    wheel = wheels[-1]
    print(f"[drive] Installing wheel: {wheel.name}")
    subprocess.run([sys.executable, "-m", "pip", "install", str(wheel)], check=True)
    return 1


def link_raw_zip_from_input(staging_input: Path, raw_zip: Path) -> bool:
    if raw_zip.is_file() and raw_zip.stat().st_size > 0:
        return False
    if not staging_input.is_dir():
        return False
    zips = sorted(staging_input.glob("*.zip"))
    if not zips:
        return False
    source = zips[-1]
    ensure_parent(raw_zip)
    if raw_zip.exists():
        raw_zip.unlink()
    shutil.copy2(source, raw_zip)
    print(f"[drive] Staged raw zip: {source.name} -> {raw_zip}")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest.expanduser().resolve())
        if "profiles" not in manifest:
            raise DriveSyncError("Manifest missing 'profiles'.")

        if args.command == "check":
            cmd_check(manifest)
            return 0

        if args.command == "promote-loras":
            cmd_promote_loras(manifest, dry_run=args.dry_run)
            return 0

        profile_data = manifest["profiles"].get(args.profile)
        if not isinstance(profile_data, dict):
            raise DriveSyncError(f"Unknown profile: {args.profile}")

        if args.command == "status":
            cmd_status(manifest, args.profile, args.only)
            return 0

        direction = "pull" if args.command == "pull" else "push"
        entries = profile_data.get(direction, [])
        if not isinstance(entries, list):
            raise DriveSyncError(f"Profile {args.profile} missing '{direction}' list.")

        cmd_check(manifest)
        for entry in filter_entries(entries, args.only):
            if not isinstance(entry, dict):
                continue
            sync_entry(manifest, entry, to_remote=(direction == "push"), dry_run=args.dry_run, force=args.force)

        if args.command == "pull" and args.profile == "training":
            staging = Path("/home/ubuntu/drive_sync")
            install_wheels_from_staging(staging / "venv")
            try:
                from pipeline_config import load_pipeline_settings

                settings = load_pipeline_settings()
                link_raw_zip_from_input(staging / "input", settings.raw_zip)
            except Exception:
                link_raw_zip_from_input(staging / "input", Path("/home/ubuntu/raw_assets.zip"))

        print(f"[drive] {args.command} complete ({args.profile}).")
        return 0
    except DriveSyncError as exc:
        print(f"[drive] Error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"[drive] rclone failed (exit {exc.returncode}).", file=sys.stderr)
        return exc.returncode or 1


if __name__ == "__main__":
    sys.exit(main())
