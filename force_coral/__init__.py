"""
force_coral -- Custom robotics experiments built on LIBERO + FoundationPose.

Importing this package:
1. Ensures LIBERO and the workspace root are on sys.path.
2. Exposes helper utilities and light-weight FORTE modules.
3. Optionally bootstraps the heavy LIBERO / robosuite extension registry when
   the robotics stack is available.
"""

import os as _os
import sys as _sys

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_CORAL_ROOT = _os.path.dirname(_THIS_DIR)
_CACHE_ROOT = _os.path.join("/tmp", "force_coral_runtime")


def _ensure_runtime_cache_dirs() -> None:
    """Point common runtime caches to writable locations.

    This keeps matplotlib, numba, and related robotics dependencies importable
    on machines where the default home-cache locations are not writable.
    """
    mpl_dir = _os.path.join(_CACHE_ROOT, "matplotlib")
    numba_dir = _os.path.join(_CACHE_ROOT, "numba")
    xdg_dir = _os.path.join(_CACHE_ROOT, "xdg_cache")
    for path in (mpl_dir, numba_dir, xdg_dir):
        _os.makedirs(path, exist_ok=True)
    _os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
    _os.environ.setdefault("NUMBA_CACHE_DIR", numba_dir)
    _os.environ.setdefault("XDG_CACHE_HOME", xdg_dir)


_ensure_runtime_cache_dirs()

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


_LIBERO_BOOTSTRAPPED = False
_LIBERO_BOOTSTRAP_ERROR = None


def bootstrap_libero_extensions(require: bool = False) -> bool:
    """Import and register the LIBERO / robosuite extension stack."""
    global _LIBERO_BOOTSTRAPPED, _LIBERO_BOOTSTRAP_ERROR
    if _LIBERO_BOOTSTRAPPED:
        return True
    try:
        import force_coral.libero_ext  # noqa: F401
    except Exception as exc:  # pragma: no cover - robotics stack dependent
        _LIBERO_BOOTSTRAP_ERROR = exc
        if require:
            raise
        return False
    _LIBERO_BOOTSTRAPPED = True
    _LIBERO_BOOTSTRAP_ERROR = None
    return True


def has_libero_extensions() -> bool:
    """Return whether the heavy robotics stack is currently available."""
    return bootstrap_libero_extensions(require=False)


# Try to preserve the previous "import force_coral bootstraps the robotics
# registry" behavior without making light-weight unit tests fail when the heavy
# stack is absent.
bootstrap_libero_extensions(require=False)
