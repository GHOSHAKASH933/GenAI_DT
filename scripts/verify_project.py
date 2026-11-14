"""Repository structure and lightweight runtime checks for GenAI_DT."""
from __future__ import annotations

import importlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_FILES = [
    "pyproject.toml", "requirements.txt", "setup.sh", "Makefile",
    "genai_dt/__init__.py", "genai_dt/data.py", "genai_dt/simulator.py",
    "genai_dt/models.py", "genai_dt/federated.py", "genai_dt/conformal.py",
    "genai_dt/control.py", "genai_dt/intervention.py", "genai_dt/experiments.py",
    "genai_dt/support_analysis.py", "genai_dt/robustness.py",
    "analysis/analyse.py", "analysis/figures.py", "analysis/real_figure.py",
    "realdata/__init__.py", "realdata/shanghai.py", "realdata/real_experiments.py",
    "data/README.md", "results/figures/README.md",
]


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing expected files:\n  " + "\n  ".join(missing))

    py_files = sorted(
        path for path in ROOT.rglob("*.py")
        if not ({".venv", "build", "dist", "__pycache__"} & set(path.parts))
    )
    for path in py_files:
        py_compile.compile(str(path), doraise=True)
    print(f"Syntax check passed ({len(py_files)} Python files).")

    modules = [
        "genai_dt.data", "genai_dt.simulator", "genai_dt.models", "genai_dt.federated",
        "genai_dt.conformal", "genai_dt.control", "genai_dt.intervention",
        "genai_dt.experiments", "genai_dt.support_analysis", "genai_dt.robustness",
        "analysis.analyse", "analysis.figures", "analysis.real_figure",
        "realdata.shanghai", "realdata.real_experiments",
    ]
    for name in modules:
        importlib.import_module(name)
    from genai_dt import experiments
    for symbol in ("main", "run_seed", "e1_treatment_effect", "e2_prediction", "e3_e4", "e5_edge"):
        if not callable(getattr(experiments, symbol, None)):
            raise SystemExit(f"Experiment pipeline is missing callable: {symbol}")
    print(f"Import/pipeline check passed ({len(modules)} modules and E1–E5 entry points).")

    # Restore simulation dimensions in case a real-data module was imported above.
    from genai_dt import data as D
    D.CONTEXT, D.HORIZON = 24, 24
    D.CTX_DIM = D.CONTEXT * 3 + D.HORIZON * 2
    D.I_SCALE, D.C_SCALE = 2.0, 40.0

    import numpy as np
    import torch
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
    assert tuple(x.shape) == (1, D.CTX_DIM)
    for model in (CVAETwin(), ActionResidualTwin()):
        loss, _, _ = model(x, y)
        assert torch.isfinite(loss).item()
        assert tuple(model.sample(x, n_samples=2).shape) == (2, 1, D.HORIZON)
    assert tuple(PointMLP()(x).shape) == (1, D.HORIZON)
    print("Model forward/sample shape checks passed.")
    print("Verification passed. Full simulation and real-data runs were not executed.")


if __name__ == "__main__":
    main()
