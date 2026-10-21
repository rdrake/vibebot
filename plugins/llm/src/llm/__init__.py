"""LLM: AI-powered commands."""

from __future__ import annotations

import sys

import __main__

__version__ = "0.1.0"

# Make Limnoria's ``reload`` a true deep reload. ``loadPluginModule`` only
# re-execs this package's ``__init__``; the ``from . import`` below would then
# rebind to already-cached submodules (config, plugin, service, verse.*) — so
# edits to those files (e.g. prompt constants in service.py) would silently run
# stale. Dropping any cached submodules first forces the imports below to
# re-execute the whole tree. On first import at startup nothing is cached, so
# this is a no-op then; it only bites on ``@reload``.
for _name in [m for m in list(sys.modules) if m.startswith(__name__ + ".")]:
    del sys.modules[_name]

if not hasattr(__main__, "__file__"):  # pragma: no cover - offline CLI (python -m) only
    # supybot i18n.getPluginDir() falls back to sys.modules['__main__'].__file__
    # when the plugin isn't bot-loaded; under `python -m llm.…` the bootstrap
    # __main__ has no __file__, raising AttributeError at import. A placeholder
    # lets getPluginDir degrade to the (caught) PluginNotFound path so the
    # package imports cleanly outside the bot (e.g. the taste_mine CLI).
    __main__.__file__ = "<llm-standalone>"

from . import config, plugin  # noqa: E402  (must follow the submodule purge)

Class = plugin.LLM
configure = config.configure

__all__ = ["Class", "configure", "__version__"]
