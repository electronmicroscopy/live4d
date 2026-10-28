"""docs/CLI.md is generated from the programs' parsers: it must match them, and every parameter
must carry a description (the reference is only as good as the help strings)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = (Path(__file__).resolve().parents[2] / "python/live4d/parallax")


def _generator():
    spec = importlib.util.spec_from_file_location("gen_cli_reference",
                                                  REPO / "scripts" / "gen_cli_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_reference_is_current():
    gen = _generator()
    text = (Path(__file__).resolve().parents[3] / "docs/parallax/CLI.md").read_text()
    assert text == gen.render(), "docs/CLI.md is stale: run python scripts/gen_cli_reference.py"
    assert sys.executable not in text, "docs/CLI.md names this machine's Python"   # same on every machine


def test_every_parameter_is_described():
    gen = _generator()

    def actions(parser):
        for a in parser._actions:
            if isinstance(a, argparse._SubParsersAction):
                for sub in a.choices.values():
                    yield from actions(sub)
            elif not isinstance(a, argparse._HelpAction):
                yield a

    missing = [f"{key}: {a.option_strings or a.dest}"
               for key, parser in gen._parsers() for a in actions(parser) if not a.help]
    assert not missing, missing
