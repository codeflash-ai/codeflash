"""Cache libcst visitor dispatch table construction.

libcst's ``MatcherDecoratableTransformer`` and
``MatcherDecoratableVisitor`` rebuild visitor dispatch tables on
every instantiation by iterating ``dir(self)`` (~600 attributes)
and calling ``getattr`` + ``inspect.ismethod`` on each.  The
results depend only on the class, not the instance, so caching
by ``type(obj)`` is safe.

Import this module before any libcst visitors are instantiated
to install the cache.
"""

from __future__ import annotations

from typing import Any

import libcst.matchers._visitors as _mv

_visit_cache: dict[type, Any] = {}
_leave_cache: dict[type, Any] = {}
_matchers_cache: dict[type, Any] = {}

_original_visit = _mv._gather_constructed_visit_funcs  # noqa: SLF001
_original_leave = _mv._gather_constructed_leave_funcs  # noqa: SLF001
_original_matchers = _mv._gather_matchers  # noqa: SLF001


def _cached_visit(obj: object) -> Any:
    """Return cached visit-function dispatch table for the object's class."""
    cls = type(obj)
    try:
        return _visit_cache[cls]
    except KeyError:
        result = _original_visit(obj)
        _visit_cache[cls] = result
        return result


def _cached_leave(obj: object) -> Any:
    """Return cached leave-function dispatch table for the object's class."""
    cls = type(obj)
    try:
        return _leave_cache[cls]
    except KeyError:
        result = _original_leave(obj)
        _leave_cache[cls] = result
        return result


def _cached_matchers(obj: object) -> Any:
    """Return cached matcher dispatch table for the object's class."""
    cls = type(obj)
    try:
        return dict(_matchers_cache[cls])
    except KeyError:
        result = _original_matchers(obj)
        _matchers_cache[cls] = result
        return dict(result)


_mv._gather_constructed_visit_funcs = _cached_visit  # noqa: SLF001
_mv._gather_constructed_leave_funcs = _cached_leave  # noqa: SLF001
_mv._gather_matchers = _cached_matchers  # noqa: SLF001
