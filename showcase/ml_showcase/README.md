# RentAFit ML Showcase

Standalone work-in-progress demo for the RentAFit ML stack.

It exposes a lightweight browser UI for:

- Model A: rental price range prediction
- Model B: moderation and lifecycle decisioning
- Model C: recommendation output

This showcase uses the same `code/`, `models/`, `data/`, and `reports/` assets already stored in this repository, but keeps the demo UI separate from the main platform frontend. It is meant to demonstrate the current ML progress, not to represent the final integrated RentAFit product surface.

## Preview

![RentAFit ML Showcase preview](../assets/screenshots/ml_showcase.png)

## Run

From the repository root:

```bash
python3 showcase/ml_showcase/server.py --port 8090
```

Or with the helper script:

```bash
./showcase/ml_showcase/run_showcase.sh
```

Then open the local URL printed in the terminal.

The helper script automatically uses the repository `.venv` when it exists.

## Notes

- Public static companion demo: `https://anmol-228.github.io/RentAFit-ML-Showcase/`
- If port `8090` is free, the showcase starts there.
- If another RentAFit showcase is already running there, the script exits cleanly.
- If `8090` is occupied by something else, the server automatically moves to the next available port.
