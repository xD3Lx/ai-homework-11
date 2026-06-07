"""Observability shim — uses LangSmith when installed, no-ops otherwise.

Lets the code run in environments without langsmith while keeping the real
@traceable instrumentation active in production (pip install -r requirements.txt).
"""
from __future__ import annotations

from functools import wraps

try:
    from langsmith import traceable as _ls_traceable  # type: ignore

    HAVE_LANGSMITH = True

    def traceable(*d_args, **d_kwargs):
        return _ls_traceable(*d_args, **d_kwargs)

except Exception:  # langsmith not installed
    HAVE_LANGSMITH = False

    def traceable(*d_args, **d_kwargs):
        # Support both @traceable and @traceable(name=..., tags=[...])
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]

        def deco(fn):
            @wraps(fn)
            def wrapper(*a, **k):
                return fn(*a, **k)

            return wrapper

        return deco
