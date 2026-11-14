"""Lightweight structure, syntax, import, and tensor-shape checks.

Run from the repository root with `python scripts/verify_project.py`.
This does not download the Shanghai data or run the long experiments.
"""
from __future__ import annotations

import importlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    required = [
        "genai_dt/__init__.py",
        "genai_dt/data.py",
        "genai_dt/simulator.py",
        "genai_dt/models.py",
        "genai_dt/federated.py",
        "genai_dt/conformal.py",
        "genai_dt/control.py",
        "genai_dt/intervention.py",
        "genai_dt/experiments.py",
        "genai_dt/support_analysis.py",
        "genai_dt/robustness.py",
        "analysis/analyse.py",
        "analysis/figures.py",
        "analysis/real_figure.py",
        "realdata/shanghai.py",
        "realdata/real_experiments.py",
        "data/README.md",
        "requirements.txt",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing expected project files:\n  " + "\n  ".join(missing))

    python_files = sorted(
        path for path in ROOT.rglob("*.py")
        if ".venv" not in path.parts and "build" not in path.parts
    )
    for path in python_files:
        py_compile.compile(str(path), doraise=True)
    print(f"Syntax check passed ({len(python_files)} Python files).")

    modules = [
        "genai_dt.data",
        "genai_dt.simulator",
        "genai_dt.models",
        "genai_dt.federated",
        "genai_dt.conformal",
        "genai_dt.control",
        "genai_dt.intervention",
        "genai_dt.experiments",
        "genai_dt.support_analysis",
        "genai_dt.robustness",
        "analysis.analyse",
        "analysis.figures",
        "realdata.shanghai",
        "realdata.real_experiments",
    ]
    for name in modules:
        importlib.import_module(name)
    print(f"Import check passed ({len(modules)} modules).")

    import numpy as np
    import torch
    from genai_dt import data as D
    from genai_dt.models import ActionResidualTwin, CVAETwin, PointMLP

    context = D.encode_context(
        np.full(D.CONTEXT, D.G_MEAN, dtype=np.float32),
        np.zeros(D.CONTEXT, dtype=np.float32),
        np.zeros(D.CONTEXT, dtype=np.float32),
        np.zeros(D.HORIZON, dtype=np.float32),
        np.zeros(D.HORIZON, dtype=np.float32),
    ).astype(np.float32)
    x = torch.as_tensor(context[None, :])
    y = torch.zeros((1, D.HORIZON), dtype=torch.float32)
    assert x.shape == (1, D.CTX_DIM)
    for model in (CVAETwin(), ActionResidualTwin()):
        loss, _, _ = model(x, y)
        assert torch.isfinite(loss).item()
        sample = model.sample(x, n_samples=2)
        assert sample.shape == (2, 1, D.HORIZON)
    assert PointMLP()(x).shape == (1, D.HORIZON)
    print("Model shape/forward smoke checks passed.")
    print("Project verification passed. Full experiments and real-data checks were not run.")


if __name__ == "__main__":
    main()
