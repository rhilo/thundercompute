#!/usr/bin/env python3
"""Install GPU-specific dependencies after setup.sh (Flash Attention 2 for batch captioning)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"

FLASH_ATTN_PACKAGE = "flash-attn"
BUILD_PREREQUISITES = ("packaging", "ninja", "wheel")
LOW_MEMORY_MAX_JOBS = 1
BALANCED_MAX_JOBS = 3
RAM_GB_FOR_BALANCED = 32


class PostSetupError(Exception):
    """Raised when post-setup cannot continue safely."""


def mem_available_gb() -> float | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            kb = int(line.split()[1])
            return kb / (1024 * 1024)
    return None


def default_max_jobs() -> int:
    available = mem_available_gb()
    if available is not None and available >= RAM_GB_FOR_BALANCED:
        return BALANCED_MAX_JOBS
    return LOW_MEMORY_MAX_JOBS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    suggested = default_max_jobs()
    parser = argparse.ArgumentParser(
        description="Install Flash Attention 2 into the project venv. Run after bash setup.sh.",
    )
    parser.add_argument("--force", action="store_true", help="Reinstall flash-attn even if importable.")
    parser.add_argument("--skip-flash-attn", action="store_true", help="Verify only; do not install.")
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help=f"Parallel compile jobs (default: {suggested} when MemAvailable >= {RAM_GB_FOR_BALANCED}GB else 1).",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help=f"Force max-jobs={LOW_MEMORY_MAX_JOBS}.",
    )
    return parser.parse_args(argv)


def reexec_in_venv_if_needed() -> None:
    if not VENV_PYTHON.is_file():
        raise PostSetupError(
            f"Virtual environment not found at {VENV_PYTHON.parent.parent}. Run: bash setup.sh"
        )
    if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        os.execv(VENV_PYTHON, [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


def run_pip(args: list[str], *, env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print(f"[post-setup] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require_torch_cuda() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise PostSetupError(
            "CUDA is not available in the active venv. "
            "Run setup.sh on a GPU instance with NVIDIA drivers loaded."
        )
    return torch.cuda.get_device_name(0), torch.version.cuda or "unknown"


def install_build_prerequisites() -> None:
    missing = [pkg for pkg in BUILD_PREREQUISITES if not module_available(pkg.replace("-", "_"))]
    if missing:
        print(f"[post-setup] Installing build prerequisites: {', '.join(missing)}")
        run_pip(["install", *missing])


def require_flash_attn_python() -> None:
    if sys.version_info >= (3, 12):
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise PostSetupError(
            f"flash-attn is not available for Python {version} in this setup. "
            "This is optional: leave caption.attn as 'auto' or set it to 'sdpa' and run batch_caption.py. "
            "Use a Python 3.10/3.11 venv only if you specifically need Flash Attention 2."
        )


def build_env(max_jobs: int) -> dict[str, str]:
    env = os.environ.copy()
    env["MAX_JOBS"] = str(max_jobs)
    env["NVCC_THREADS"] = "1"
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(max_jobs)
    return env


def try_install_flash_attn_wheel(force: bool) -> bool:
    pip_args = ["install", FLASH_ATTN_PACKAGE, "--only-binary", ":all:"]
    if force:
        pip_args.append("--force-reinstall")
    print(f"[post-setup] Trying prebuilt {FLASH_ATTN_PACKAGE} wheel...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", *pip_args],
        env=os.environ.copy(),
        check=False,
    )
    return result.returncode == 0 and module_available("flash_attn")


def install_flash_attn(force: bool, max_jobs: int) -> None:
    if module_available("flash_attn") and not force:
        import flash_attn

        print(f"[post-setup] flash-attn already installed ({flash_attn.__version__}); skipping.")
        return

    require_flash_attn_python()
    install_build_prerequisites()

    if try_install_flash_attn_wheel(force):
        import flash_attn

        print(f"[post-setup] Installed prebuilt flash-attn ({flash_attn.__version__}).")
        return

    env = build_env(max_jobs)
    pip_args = ["install", FLASH_ATTN_PACKAGE, "--no-build-isolation", "--no-cache-dir"]
    if force:
        pip_args.insert(1, "--force-reinstall")

    print(
        f"[post-setup] Compiling {FLASH_ATTN_PACKAGE} (MAX_JOBS={max_jobs}). "
        "On 64GB hosts try --max-jobs 3; use --low-memory for 1."
    )
    run_pip(pip_args, env=env)


def verify_flash_attn() -> None:
    if not module_available("flash_attn"):
        raise PostSetupError("flash-attn is not installed.")
    import flash_attn

    print(f"[post-setup] flash-attn {flash_attn.__version__} import OK")


def verify_transformers_flash_attention() -> None:
    import transformers

    if not hasattr(transformers, "is_flash_attn_2_available"):
        return
    if transformers.is_flash_attn_2_available():
        print("[post-setup] transformers reports Flash Attention 2 is available.")
    else:
        raise PostSetupError(
            "flash-attn is installed but transformers does not detect FA2. "
            "Upgrade transformers: pip install -U 'transformers>=4.44.0'"
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        reexec_in_venv_if_needed()
        args = parse_args(argv)

        max_jobs = LOW_MEMORY_MAX_JOBS if args.low_memory else (args.max_jobs or default_max_jobs())
        if max_jobs < 1:
            raise PostSetupError("--max-jobs must be at least 1.")

        available = mem_available_gb()
        if available is not None:
            print(f"[post-setup] MemAvailable: {available:.1f} GB — using MAX_JOBS={max_jobs}")

        device_name, cuda_version = require_torch_cuda()
        print(f"[post-setup] GPU: {device_name} (torch CUDA {cuda_version})")

        if not args.skip_flash_attn:
            install_flash_attn(args.force, max_jobs)

        verify_flash_attn()
        verify_transformers_flash_attention()

        print(
            "[post-setup] Done. Batch caption: python3 batch_caption.py --config pipeline.yaml "
            "(or --attn sdpa to skip FA2)"
        )
        return 0
    except PostSetupError as exc:
        print(f"[post-setup] Error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"[post-setup] pip failed (exit {exc.returncode}). Try --max-jobs 3 or --low-memory.", file=sys.stderr)
        return exc.returncode or 1


if __name__ == "__main__":
    sys.exit(main())
