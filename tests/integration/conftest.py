"""Shared pytest configuration for all integration tests.

This module centralizes protection against PyArrow extension type
duplicate registrations which can occur when multiple tests import
Pandas/PyArrow in the same process. It applies a defensive wrapper
around pyarrow.lib.register_extension_type to ignore duplicate
registration errors and provides a lightweight reset helper used by
child conftests.
"""

from __future__ import annotations

import warnings
from typing import Any


def _safe_register_extension_type(extension_type: Any):
    """Call pyarrow.lib.register_extension_type but ignore duplicate registration.

    We intentionally swallow the specific ArrowKeyError raised when an extension
    with the same name is already registered. Other exceptions are re-raised.
    """
    try:
        import pyarrow.lib as _lib

        _lib.register_extension_type(extension_type)
    except Exception as e:  # pragma: no cover - defensive
        # Avoid importing pyarrow.error directly to keep this file import-light
        msg = str(e)
        if "already defined" in msg or "already registered" in msg:
            # Duplicate registration: benign when running many tests in one process
            return
        raise


def reset_pyarrow_state():
    """Attempt to clear pyarrow extension registries used in tests.

    This function mirrors per-directory conftests but is intentionally
    defensive: any error during reset is ignored. It helps keep test runs
    stable across different pyarrow versions.
    """
    try:
        import pyarrow as pa

        # Try to clear known internal registries if present
        for name in ("_extension_types_registry", "_extension_type_registry"):
            if hasattr(pa, name):
                try:
                    getattr(pa, name).clear()
                except Exception:
                    pass

        # pyarrow.lib sometimes keeps registries under the lib module
        try:
            import pyarrow.lib as _lib

            if hasattr(_lib, "_extension_type_registry"):
                try:
                    _lib._extension_type_registry.clear()
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        # Best-effort only; tests should still run if pyarrow is not present
        pass

    def pytest_runtest_setup(item):
        """Ensure a clean global state before each integration test.

        This clears the FinlabGuard global instance and removes any
        installed monkey-patch markers on `finlab.data` to prevent tests
        from leaking state into subsequent tests.
        """
        # Reset FinlabGuard global instance if present
        try:
            from finlab_guard.core import guard as _guard_mod

            if hasattr(_guard_mod, "_global_guard_instance"):
                try:
                    _guard_mod._global_guard_instance = None
                except Exception:
                    pass
        except Exception:
            pass

        # Remove any residual finlab.data._original_get to force tests to
        # install their own patched state explicitly
        try:
            import sys

            if "finlab.data" in sys.modules:
                fd = sys.modules["finlab.data"]
                if hasattr(fd, "_original_get"):
                    try:
                        delattr(fd, "_original_get")
                    except Exception:
                        pass
        except Exception:
            pass

        # Diagnostic: print module entries for finlab_guard.utils.exceptions to
        # help debug exception identity issues when running full integration tests.
        try:
            import inspect
            import sys

            matches = [k for k in sys.modules.keys() if "finlab_guard" in k]
            if matches:
                print("\n[diagnostic] sys.modules finlab_guard entries:")
                for k in matches:
                    m = sys.modules.get(k)
                    try:
                        print(
                            f"[diagnostic] {k} -> id={id(m)} file={getattr(m, '__file__', None)}"
                        )
                    except Exception:
                        pass
        except Exception:
            pass


def pytest_configure(config):
    # Reset state early so child conftests see a clean starting point
    reset_pyarrow_state()

    # Suppress noisy warnings about duplicate registrations which tests
    # intentionally guard against.
    warnings.filterwarnings("ignore", message=".*pandas.period already defined.*")
    warnings.filterwarnings("ignore", message=".*extension.*already.*registered.*")

    # Defensive wrapper: replace pyarrow.lib.register_extension_type with a
    # safe wrapper that ignores duplicate registration ArrowKeyError. This
    # prevents pandas' import-time registration (which calls
    # pyarrow.register_extension_type) from crashing the entire test run
    # when multiple modules attempt to register the same extension name.
    try:
        import pyarrow as _pa

        try:
            import pyarrow.lib as _lib

            # Keep original if available
            _orig = getattr(_lib, "register_extension_type", None)

            def _safe_register(ext_type):
                try:
                    if _orig is not None:
                        _orig(ext_type)
                except Exception as _e:  # pragma: no cover - defensive
                    msg = str(_e)
                    if "already defined" in msg or "already registered" in msg:
                        return
                    raise

            if hasattr(_lib, "register_extension_type"):
                _lib.register_extension_type = _safe_register

        except Exception:
            # If pyarrow.lib import or monkeypatch fails, ignore; tests can still
            # proceed (they'll hit the normal error which existing directory
            # conftests try to reset).
            pass
    except Exception:
        pass


# Apply the safe wrapper eagerly at import time so that child conftests which
# import pandas/pyarrow at module import time see the guarded behavior.
try:  # pragma: no cover - best-effort at import
    import pyarrow as _pa

    # Patch top-level pyarrow.register_extension_type if present
    if hasattr(_pa, "register_extension_type"):
        _orig_top = _pa.register_extension_type

        def _safe_top(ext_type):
            try:
                _orig_top(ext_type)
            except Exception as _e:
                if "already defined" in str(_e) or "already registered" in str(_e):
                    return
                raise

        _pa.register_extension_type = _safe_top

    # Also patch pyarrow.lib.register_extension_type if available
    try:
        import pyarrow.lib as _lib

        if hasattr(_lib, "register_extension_type"):
            _orig_lib = _lib.register_extension_type

            def _safe_lib(ext_type):
                try:
                    _orig_lib(ext_type)
                except Exception as _e:
                    if "already defined" in str(_e) or "already registered" in str(_e):
                        return
                    raise

            _lib.register_extension_type = _safe_lib
    except Exception:
        pass
except Exception:
    # pyarrow not present or patch failed; ignore (other conftests still try reset)
    pass
