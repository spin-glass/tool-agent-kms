"""Internal helper: assemble a Jupyter notebook from (type, source) cells.

Not part of the published artefact — exists only so we can regenerate
notebook skeletons reproducibly during initial authoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def make_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for ctype, src in cells:
        cell = {
            "cell_type": ctype,
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if ctype == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: str, cells: list[tuple[str, str]]) -> None:
    nb = make_notebook(cells)
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    print("module, not a CLI", file=sys.stderr)
    sys.exit(1)
