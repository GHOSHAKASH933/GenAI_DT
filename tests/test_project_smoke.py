import numpy as np
import torch

from genai_dt import data as D
from genai_dt.models import ActionResidualTwin, CVAETwin, PointMLP


def test_model_forward_and_samples_have_expected_shapes():
    x_np = D.encode_context(
        np.full(D.CONTEXT, D.G_MEAN, dtype=np.float32),
        np.zeros(D.CONTEXT, dtype=np.float32),
        np.zeros(D.CONTEXT, dtype=np.float32),
        np.zeros(D.HORIZON, dtype=np.float32),
        np.zeros(D.HORIZON, dtype=np.float32),
    ).astype(np.float32)
    x = torch.as_tensor(x_np[None])
    y = torch.zeros((1, D.HORIZON), dtype=torch.float32)
    assert x.shape == (1, D.CTX_DIM)
    for model in (CVAETwin(), ActionResidualTwin()):
        loss, _, _ = model(x, y)
        assert torch.isfinite(loss)
        assert model.sample(x, n_samples=2).shape == (2, 1, D.HORIZON)
    assert PointMLP()(x).shape == (1, D.HORIZON)


def test_experiment_driver_has_all_paper_stages():
    from genai_dt import experiments

    for name in ("e1_treatment_effect", "e2_prediction", "e3_e4", "e5_edge", "run_seed", "main"):
        assert callable(getattr(experiments, name))
    assert experiments.CFG_FULL["alpha"] == 0.10
    assert experiments.CFG_QUICK["patients_per_site"] < experiments.CFG_FULL["patients_per_site"]
