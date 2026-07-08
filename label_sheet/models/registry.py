"""
Model runner protocol, the prediction value object, and the runner registry.

A ``ModelRunner`` turns one image into a ``Prediction``. Runners are constructed
from a ``ModelConfig`` (see ``label_sheet.config``) and are heavy to build (they
load torch models), so they are instantiated once per predict run and reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Protocol

from ..config import ModelConfig


@dataclass
class Prediction:
    """Model output for one image, prefix-agnostic.

    The predict stage writes these into the ``m1_``/``m2_`` schema columns by
    prepending the model id. Empty string means "not produced".
    """

    days_to_molt: str = ""
    days_to_molt_std: str = ""
    phase: str = ""
    confidence: str = ""
    sex: str = ""
    view: str = ""
    molt_indicators: str = ""
    estimate_input: str = "not_run"
    in_training_set: str = ""

    def as_prefixed(self, model_id: str) -> Dict[str, str]:
        """Return a dict of ``{model_id}_{field}`` -> value for schema keys."""
        fields = [
            "days_to_molt", "days_to_molt_std", "phase", "confidence", "sex",
            "view", "molt_indicators", "estimate_input", "in_training_set",
        ]
        out: Dict[str, str] = {}
        for f in fields:
            key = f"{model_id}_{f}"
            out[key] = getattr(self, f)
        return out


class ModelRunner(Protocol):
    """Loads a model and predicts on single images."""

    model_id: str

    def predict_path(self, abs_path: str, in_training_set: bool) -> Prediction:
        ...


#: runner-name -> factory(ModelConfig) -> ModelRunner
_FACTORIES: Dict[str, Callable[[ModelConfig], "ModelRunner"]] = {}


def register_runner(name: str, factory: Callable[[ModelConfig], "ModelRunner"]) -> None:
    _FACTORIES[name] = factory


def _ensure_builtin_runners() -> None:
    """Import runner modules so their register_runner() calls execute.

    Done lazily (not at package import) so the extract stage never pulls torch.
    """
    if "estimator" not in _FACTORIES:
        from . import estimator  # noqa: F401  (registers "estimator")


def get_runner(cfg: ModelConfig) -> "ModelRunner":
    _ensure_builtin_runners()
    if cfg.runner not in _FACTORIES:
        raise KeyError(f"Unknown runner {cfg.runner!r}; known: {sorted(_FACTORIES)}")
    return _FACTORIES[cfg.runner](cfg)
