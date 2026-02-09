"""
force_coral -- Custom robotics experiments built on LIBERO + FoundationPose.

Importing this package:
1. Ensures LIBERO and the CoRAL workspace root are on sys.path.
2. Registers all custom LIBERO extensions (benchmarks, problems, objects,
   robots, predicates, regions) into LIBERO's plugin registries so that
   environments can be constructed without modifying LIBERO source files.
"""

import os as _os
import sys as _sys

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_CORAL_ROOT = _os.path.dirname(_THIS_DIR)

# Ensure CoRAL root is importable (so ``import force_coral`` works from
# anywhere, and scripts inside force_coral can do cross-module imports).
if _CORAL_ROOT not in _sys.path:
    _sys.path.insert(0, _CORAL_ROOT)

# Ensure the LIBERO sub-repo is importable (``from libero.libero import ...``)
_LIBERO_DIR = _os.path.join(_CORAL_ROOT, "LIBERO")
if _os.path.isdir(_LIBERO_DIR) and _LIBERO_DIR not in _sys.path:
    _sys.path.insert(0, _LIBERO_DIR)

# ---------------------------------------------------------------------------
# Data path helper
# ---------------------------------------------------------------------------
_DATA_DIR = _os.path.join(_THIS_DIR, "data")


def get_data_path(key: str) -> str:
    """Return the absolute path to one of force_coral's custom data dirs.

    Supported keys: ``bddl_files``, ``init_states``, ``assets``.
    """
    _paths = {
        "bddl_files": _os.path.join(_DATA_DIR, "bddl_files"),
        "init_states": _os.path.join(_DATA_DIR, "init_files"),
        "assets": _os.path.join(_DATA_DIR, "assets"),
    }
    if key not in _paths:
        raise KeyError(
            f"Unknown data key {key!r}. Available: {list(_paths.keys())}"
        )
    return _paths[key]


# ---------------------------------------------------------------------------
# Register LIBERO extensions (objects, robots, predicates, regions,
# problems, benchmarks).  Importing the sub-package triggers the
# @register_* decorators and manual dict updates.
# ---------------------------------------------------------------------------
import force_coral.libero_ext  # noqa: E402,F401

# ---------------------------------------------------------------------------
# FORTE modules (dynamics estimation, perception/VLM interface)
# ---------------------------------------------------------------------------
import force_coral.dynamics     # noqa: E402,F401
import force_coral.perception   # noqa: E402,F401
