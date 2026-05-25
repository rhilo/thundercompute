#!/usr/bin/env python3
"""Textual entry point for LoRA training and ComfyUI generation workflows."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO

import yaml
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, RichLog, Select, Static

from drive_sync import COMFY_PULL_IDS, TRAINING_PULL_IDS
from pipeline_config import THUNDER_HARDWARE_PRESETS, PipelineConfigError, write_hardware_preset

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_CONFIG = SCRIPT_DIR / "pipeline.yaml"
PIPELINE_EXAMPLE = SCRIPT_DIR / "pipeline.example.yaml"
VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
LOG_DIR = SCRIPT_DIR / "logs"


def python_command() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def training_env_ready() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def hardware_options() -> list[tuple[str, str]]:
    suggested = [
        (f"{preset.label} (suggested)", key)
        for key, preset in THUNDER_HARDWARE_PRESETS.items()
        if preset.suggested
    ]
    others = [
        (preset.label, key)
        for key, preset in THUNDER_HARDWARE_PRESETS.items()
        if not preset.suggested
    ]
    return suggested + others


class PipelineTUI(App[None]):
    """One terminal UI for training and ComfyUI instance setup."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #title {
        padding: 1 2;
        text-style: bold;
    }

    #status {
        padding: 0 2 1 2;
        color: $text-muted;
    }

    .row {
        padding: 0 1;
        height: auto;
    }

    Button {
        margin: 0 1 1 0;
        min-width: 22;
    }

    #log {
        height: 1fr;
        border: round $accent;
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("escape", "home", "Home"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.mode = "home"
        self.running = False
        self._log_file: TextIO | None = None
        self._log_path: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Thunder Compute Pipeline", id="title")
        yield Label("", id="status")
        with Horizontal(classes="row"):
            yield Button("Train LoRA", id="mode_train", variant="primary")
            yield Button("Generate Images (ComfyUI)", id="mode_comfy", variant="primary")
            yield Button("Home", id="mode_home")
        with Vertical(id="train_actions", classes="row"):
            yield Label("Training workflow: setup, Drive pull, captioning, training, LoRA push")
            with Horizontal():
                yield Label("Hardware preset:")
                yield Select(hardware_options(), id="hardware_preset", value="best")
                yield Button("Save Hardware Preset", id="save_hardware")
            with Horizontal():
                yield Button("Run Full Training Workflow", id="train_full", variant="success")
                yield Button("Pull Drive Assets", id="train_pull")
                yield Button("Run Setup", id="train_setup")
                yield Button("Post-Setup FA2", id="train_post_setup")
            with Horizontal():
                yield Button("Train without Drive", id="train_local", variant="success")
                yield Button("Preprocess + Caption + Train", id="train_pipeline")
                yield Button("Export LoRAs", id="train_export")
                yield Button("Push LoRAs", id="train_push")
                yield Button("Promote LoRAs", id="train_promote")
        with Vertical(id="comfy_actions", classes="row"):
            yield Label("ComfyUI workflow: pull generation models and push rendered images")
            with Horizontal():
                yield Button("Pull Comfy Models", id="comfy_pull", variant="success")
                yield Button("Pull FLUX Diffusers", id="comfy_pull_flux")
                yield Button("Verify Folders", id="comfy_verify")
                yield Button("Push Renders", id="comfy_push")
        yield RichLog(id="log", highlight=False, markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#hardware_preset", Select).value = self.load_configured_hardware_preset()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.write_log("[bold]Thunder Compute pipeline menu[/bold]")
        self.write_log(
            "Command output appears in this panel. A plain-text copy is also written under "
            f"[cyan]{LOG_DIR}/[/cyan] (one file per workflow you start)."
        )
        self.write_log(
            "[dim]Tip: in a second SSH window, run "
            f"tail -f {LOG_DIR}/tui-*.log[/dim]"
        )
        self.write_log("")
        self.show_home()

    def action_home(self) -> None:
        self.show_home()

    def show_home(self) -> None:
        self.mode = "home"
        self.query_one("#train_actions").display = False
        self.query_one("#comfy_actions").display = False
        self.set_status(
            "Choose a workflow. Training pulls input/flux/venv; Comfy pulls unet/clip/vae/loras only."
        )

    def show_train(self) -> None:
        self.mode = "training"
        self.query_one("#train_actions").display = True
        self.query_one("#comfy_actions").display = False
        self.set_status("Training mode selected. Full workflow includes setup, captioning, training, and LoRA push.")

    def show_comfy(self) -> None:
        self.mode = "comfyui"
        self.query_one("#train_actions").display = False
        self.query_one("#comfy_actions").display = True
        self.set_status("ComfyUI mode selected. This never pulls venv/ and never runs training setup.")

    def set_status(self, text: str) -> None:
        config_status = "pipeline.yaml OK" if PIPELINE_CONFIG.exists() else "pipeline.yaml will be created"
        self.query_one("#status", Label).update(f"{text}  [{config_status}]")

    def write_log(self, message: str) -> None:
        """Append styled text to the RichLog panel. Do not name this 'log' — it shadows Textual's App.log."""
        panel = self.query_one("#log", RichLog)
        panel.write(message)
        panel.scroll_end(animate=False)
        self._write_log_file(self._strip_markup(message))

    @staticmethod
    def _strip_markup(message: str) -> str:
        """Best-effort plain text for the log file (Rich tags removed)."""
        out: list[str] = []
        i = 0
        while i < len(message):
            if message[i] == "[":
                end = message.find("]", i)
                if end == -1:
                    out.append(message[i:])
                    break
                i = end + 1
                continue
            out.append(message[i])
            i += 1
        return "".join(out)

    def _write_log_file(self, plain_line: str) -> None:
        if self._log_file is None:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self._log_file.write(f"{timestamp}  {plain_line}\n")
        self._log_file.flush()

    def write_log_raw(self, message: str) -> None:
        """Append subprocess output without interpreting [brackets] as Rich markup."""
        if not message:
            return
        escaped = message.replace("\\", "\\\\").replace("[", "\\[")
        panel = self.query_one("#log", RichLog)
        panel.write(escaped)
        panel.scroll_end(animate=False)
        self._write_log_file(message)

    def open_run_log(self, workflow_name: str) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = LOG_DIR / f"tui-{stamp}-{workflow_name}.log"
        if self._log_file is not None:
            self._log_file.close()
        self._log_path = path
        self._log_file = path.open("a", encoding="utf-8")
        self._write_log_file(f"=== {workflow_name} started ===")
        self._write_log_file(f"cwd: {SCRIPT_DIR}")
        return path

    def close_run_log(self, *, success: bool) -> None:
        if self._log_file is not None:
            status = "complete" if success else "failed"
            self._write_log_file(f"=== workflow {status} ===")
            self._log_file.close()
            self._log_file = None
        if self._log_path is not None:
            self.write_log(f"[dim]Log file:[/dim] {self._log_path}")
            self._log_path = None

    def ensure_pipeline_config(self) -> None:
        if PIPELINE_CONFIG.exists():
            return
        if not PIPELINE_EXAMPLE.exists():
            raise FileNotFoundError(f"Missing {PIPELINE_EXAMPLE}")
        shutil.copy2(PIPELINE_EXAMPLE, PIPELINE_CONFIG)
        self.write_log(f"[green]Created[/green] {PIPELINE_CONFIG.name} from {PIPELINE_EXAMPLE.name}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if self.running:
            self.write_log("[yellow]A command is already running.[/yellow]")
            return

        if button_id == "mode_home":
            self.show_home()
            return
        if button_id == "mode_train":
            self.show_train()
            return
        if button_id == "mode_comfy":
            self.show_comfy()
            return

        actions = {
            "train_full": self.training_full_workflow,
            "train_local": self.training_local_workflow,
            "train_pull": self.training_pull,
            "train_setup": self.training_setup,
            "train_post_setup": self.training_post_setup,
            "train_pipeline": self.training_pipeline,
            "train_export": self.training_export,
            "train_push": self.training_push,
            "train_promote": self.training_promote,
            "save_hardware": self.save_hardware_action,
            "comfy_pull": self.comfy_pull,
            "comfy_pull_flux": self.comfy_pull_flux,
            "comfy_verify": self.comfy_verify,
            "comfy_push": self.comfy_push,
        }
        action = actions.get(button_id)
        if action is not None:
            self.write_log(f"[bold yellow]Starting:[/bold yellow] {button_id.replace('_', ' ')}")
            self.run_worker(self._run_action(action, button_id), exclusive=True, thread=False)

    async def _run_action(self, action, button_id: str) -> None:
        try:
            await action()
        except Exception as exc:
            self.write_log(f"[bold red]Unhandled error:[/bold red] {exc}")
            raise

    def require_training_env(self) -> None:
        if training_env_ready():
            return
        raise RuntimeError(
            "Training packages are not installed yet. Use Run Setup first "
            "(bash setup.sh --no-sync), then try again."
        )

    async def run_steps(self, steps: Iterable[tuple[str, list[str]]], *, workflow_name: str = "workflow") -> None:
        self.running = True
        self.set_buttons_disabled(True)
        log_path = self.open_run_log(workflow_name)
        self.write_log(f"[cyan]Writing log file:[/cyan] {log_path}")
        success = False
        try:
            self.ensure_pipeline_config()
            if self.mode == "training":
                self.save_selected_hardware_preset()
            step_list = list(steps)
            self.write_log(f"[dim]Steps in this run: {len(step_list)}[/dim]")
            for index, (label, command) in enumerate(step_list, start=1):
                self.write_log(f"\n[bold cyan]Step {index}/{len(step_list)}: {label}[/bold cyan]")
                await self.run_command(command)
            self.write_log("\n[bold green]Workflow complete.[/bold green]")
            success = True
        except Exception as exc:
            self.write_log(f"\n[bold red]Stopped:[/bold red] {exc}")
        finally:
            self.close_run_log(success=success)
            self.running = False
            self.set_buttons_disabled(False)

    def set_buttons_disabled(self, disabled: bool) -> None:
        for button in self.query(Button):
            button.disabled = disabled

    def selected_hardware_preset(self) -> str:
        value = self.query_one("#hardware_preset", Select).value
        if not isinstance(value, str):
            return "best"
        return value

    def save_selected_hardware_preset(self) -> None:
        preset_key = self.selected_hardware_preset()
        write_hardware_preset(PIPELINE_CONFIG, preset_key)
        label = THUNDER_HARDWARE_PRESETS[preset_key].label
        self.write_log(f"[green]Hardware preset saved:[/green] {label} ({preset_key})")

    def load_configured_hardware_preset(self) -> str:
        if not PIPELINE_CONFIG.is_file():
            return "best"
        try:
            with PIPELINE_CONFIG.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
            profile = str(((raw or {}).get("instance") or {}).get("profile", "best"))
        except Exception:
            return "best"
        return profile if profile in THUNDER_HARDWARE_PRESETS else "best"

    def command_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        return env

    def prepare_command(self, command: list[str]) -> list[str]:
        if not command:
            return command
        executable = Path(command[0]).name
        if (executable in {"python", "python3"} or command[0] == python_command()) and "-u" not in command:
            return [command[0], "-u", *command[1:]]
        if executable == "bash" and shutil.which("stdbuf"):
            return ["stdbuf", "-oL", "-eL", *command]
        return command

    async def run_command(self, command: list[str]) -> None:
        command = self.prepare_command(command)
        started = datetime.now(timezone.utc)
        self.write_log(f"[dim]$ {' '.join(command)}[/dim]")
        self.write_log("[dim]Running… (output streams below as the command prints it)[/dim]")

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=SCRIPT_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self.command_env(),
        )
        assert process.stdout is not None

        lines_seen = 0
        last_output = time.monotonic()

        async def heartbeat() -> None:
            nonlocal last_output
            while process.returncode is None:
                await asyncio.sleep(15)
                if process.returncode is not None:
                    break
                if time.monotonic() - last_output >= 15:
                    elapsed = int((datetime.now(timezone.utc) - started).total_seconds())
                    self.write_log(
                        f"[yellow]Still running[/yellow] ({elapsed}s elapsed, no new output yet)…"
                    )

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip("\n\r")
                if text:
                    lines_seen += 1
                    last_output = time.monotonic()
                    self.write_log_raw(text)
                await asyncio.sleep(0)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        returncode = await process.wait()
        elapsed = int((datetime.now(timezone.utc) - started).total_seconds())
        if lines_seen == 0:
            self.write_log(
                f"[yellow]No stdout from this step[/yellow] (exit {returncode}, {elapsed}s). "
                "If you expected output, check the log file or run the same command in a normal shell."
            )
        else:
            self.write_log(f"[dim]Finished in {elapsed}s ({lines_seen} lines, exit {returncode})[/dim]")
        if returncode != 0:
            raise RuntimeError(f"{' '.join(command)} exited with {returncode}")

    def python_steps(self, *parts: str) -> list[str]:
        return [python_command(), "-u", *parts]

    async def training_full_workflow(self) -> None:
        await self.run_steps(
            [
                ("Run training setup", ["bash", "setup.sh", "--no-sync"]),
                ("Check Google Drive", self.python_steps("drive_sync.py", "check")),
                (
                    "Pull training assets (input, flux, venv)",
                    self.python_steps(
                        "drive_sync.py",
                        "pull",
                        "--profile",
                        "training",
                        "--only",
                        TRAINING_PULL_IDS,
                    ),
                ),
                ("Run full pipeline (preprocess, caption, config, train)", self.python_steps("run_pipeline.py")),
                ("Export LoRAs", self.python_steps("export_loras.py")),
                (
                    "Push LoRAs to Drive",
                    self.python_steps("drive_sync.py", "push", "--profile", "training", "--only", "loras"),
                ),
            ],
            workflow_name="train-full",
        )

    async def training_local_workflow(self) -> None:
        await self.run_steps(
            [
                ("Run training setup", ["bash", "setup.sh", "--no-sync"]),
                (
                    "Run full pipeline (preprocess, caption, config, train)",
                    self.python_steps("run_pipeline.py", "--from", "preprocess"),
                ),
                ("Export LoRAs", self.python_steps("export_loras.py")),
            ],
            workflow_name="train-local",
        )

    async def save_hardware_action(self) -> None:
        self.running = True
        self.set_buttons_disabled(True)
        try:
            self.ensure_pipeline_config()
            self.save_selected_hardware_preset()
        except PipelineConfigError as exc:
            self.write_log(f"[bold red]Hardware preset not saved:[/bold red] {exc}")
        finally:
            self.running = False
            self.set_buttons_disabled(False)

    async def training_pull(self) -> None:
        await self.run_steps(
            [
                ("Check Google Drive", self.python_steps("drive_sync.py", "check")),
                (
                    "Pull training assets",
                    self.python_steps(
                        "drive_sync.py",
                        "pull",
                        "--profile",
                        "training",
                        "--only",
                        TRAINING_PULL_IDS,
                    ),
                ),
            ],
            workflow_name="train-pull",
        )

    async def training_setup(self) -> None:
        await self.run_steps(
            [("Run training setup", ["bash", "setup.sh", "--no-sync"])],
            workflow_name="train-setup",
        )

    async def training_post_setup(self) -> None:
        self.require_training_env()
        await self.run_steps(
            [("Install/verify flash-attn", self.python_steps("post-setup.py", "--max-jobs", "3"))],
            workflow_name="train-post-setup",
        )

    async def training_pipeline(self) -> None:
        self.require_training_env()
        await self.run_steps(
            [("Run full pipeline", self.python_steps("run_pipeline.py"))],
            workflow_name="train-pipeline",
        )

    async def training_export(self) -> None:
        self.require_training_env()
        await self.run_steps(
            [("Export LoRAs", self.python_steps("export_loras.py"))],
            workflow_name="train-export",
        )

    async def training_push(self) -> None:
        await self.run_steps(
            [
                (
                    "Push LoRAs",
                    self.python_steps("drive_sync.py", "push", "--profile", "training", "--only", "loras"),
                )
            ],
            workflow_name="train-push",
        )

    async def training_promote(self) -> None:
        await self.run_steps(
            [("Promote LoRAs for Comfy", self.python_steps("drive_sync.py", "promote-loras"))],
            workflow_name="train-promote",
        )

    async def comfy_pull(self) -> None:
        await self.run_steps(
            [
                ("Check Google Drive", self.python_steps("drive_sync.py", "check")),
                (
                    "Pull Comfy models",
                    self.python_steps("drive_sync.py", "pull", "--profile", "comfyui", "--only", COMFY_PULL_IDS),
                ),
            ],
            workflow_name="comfy-pull",
        )

    async def comfy_pull_flux(self) -> None:
        await self.run_steps(
            [
                ("Check Google Drive", self.python_steps("drive_sync.py", "check")),
                (
                    "Pull FLUX diffusers",
                    self.python_steps("drive_sync.py", "pull", "--profile", "comfyui", "--only", "flux_diffusers"),
                ),
            ],
            workflow_name="comfy-pull-flux",
        )

    async def comfy_verify(self) -> None:
        checks = [
            Path("/home/ubuntu/ComfyUI/models/unet"),
            Path("/home/ubuntu/ComfyUI/models/clip"),
            Path("/home/ubuntu/ComfyUI/models/vae"),
            Path("/home/ubuntu/ComfyUI/models/loras"),
        ]
        self.running = True
        self.set_buttons_disabled(True)
        self.open_run_log("comfy-verify")
        try:
            for path in checks:
                if path.is_dir() and any(path.iterdir()):
                    self.write_log(f"[green]OK[/green] {path}")
                else:
                    self.write_log(f"[yellow]Missing or empty[/yellow] {path}")
        finally:
            self.close_run_log(success=True)
            self.running = False
            self.set_buttons_disabled(False)

    async def comfy_push(self) -> None:
        await self.run_steps(
            [
                (
                    "Push Comfy renders",
                    self.python_steps("drive_sync.py", "push", "--profile", "comfyui", "--only", "images"),
                )
            ],
            workflow_name="comfy-push",
        )


if __name__ == "__main__":
    PipelineTUI().run()
