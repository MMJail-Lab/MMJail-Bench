from __future__ import annotations

import os
import random
from typing import Any


PAPER_SEED = 42


def configure_determinism(seed: int = PAPER_SEED, strict: bool = True) -> dict[str, Any]:
    """Seed available runtimes and fail on nondeterministic Torch kernels in strict mode."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    report: dict[str, Any] = {
        "seed": seed,
        "strict": strict,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }

    try:
        import numpy as np

        np.random.seed(seed)
        report["numpy"] = np.__version__
    except ImportError:
        report["numpy"] = None

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(strict, warn_only=not strict)
        report.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
    except ImportError:
        report["torch"] = None
    return report
