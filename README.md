# Thunder Compute FLUX LoRA Pipeline

Train FLUX.1-dev LoRAs on Thunder Compute GPU instances, with optional Google Drive handoff for ComfyUI.

**Full operator guide:** open [readme.html](readme.html) in a browser.

## Quick start

```bash
git clone https://github.com/rhilo/thundercompute
cd thundercompute
cp pipeline.example.yaml pipeline.yaml
# Edit pipeline.yaml (paths, trigger word, Hugging Face token)
bash tui.sh
```

`bash tui.sh` opens the **project menu** (TUI). It is not the same as `rclone config` (Google Drive setup).

## Train without Google Drive

```bash
bash setup.sh --no-sync
source .venv/bin/activate
which python   # should be .../thundercompute/.venv/bin/python
python3 run_pipeline.py --from preprocess
python3 export_loras.py
```

In the TUI, use **Train without Drive** for the same path with buttons.

## After training (optional Drive)

```bash
python3 drive_sync.py push --profile training --only loras
python3 drive_sync.py promote-loras
```

## License

See [LICENSE](LICENSE).
