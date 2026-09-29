"""Config loader. Every leaf in config/*.yaml is {value, unit, source[, tier]}.

Values whose source starts with "PLACEHOLDER" are flagged; any Quantity computed
from them carries `placeholder=True` in its provenance.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(os.environ.get("TWIN_CONFIG_DIR", Path(__file__).resolve().parents[2] / "config"))
SECTIONS = ("field", "economics", "rods", "controller", "optimizer")


def _is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node and "source" in node


class Config:
    """Nested config with dotted access: cfg.v("field.reservoir.h")."""

    def __init__(self, tree: dict[str, Any]):
        self.tree = tree

    def node(self, path: str) -> Any:
        cur: Any = self.tree
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"config path not found: {path}")
            cur = cur[part]
        return cur

    def v(self, path: str) -> Any:
        n = self.node(path)
        if not _is_leaf(n):
            raise KeyError(f"config path is not a leaf: {path}")
        return n["value"]

    def meta(self, path: str) -> dict[str, Any]:
        n = self.node(path)
        return {k: val for k, val in n.items() if k != "value"}

    def is_placeholder(self, path: str) -> bool:
        return str(self.node(path).get("source", "")).upper().startswith("PLACEHOLDER")

    def leaves(self) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []

        def walk(prefix: str, node: Any) -> None:
            if _is_leaf(node):
                out.append((prefix, node))
            elif isinstance(node, dict):
                for k, child in node.items():
                    walk(f"{prefix}.{k}" if prefix else k, child)

        walk("", self.tree)
        return out

    def placeholders(self) -> list[str]:
        return [p for p, n in self.leaves() if str(n["source"]).upper().startswith("PLACEHOLDER")]

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        tree = copy.deepcopy(self.tree)
        for path, value in overrides.items():
            cur = tree
            parts = path.split(".")
            for part in parts[:-1]:
                cur = cur[part]
            leaf = cur[parts[-1]]
            if _is_leaf(leaf):
                leaf["value"] = value
            else:
                cur[parts[-1]] = {"value": value, "unit": "-", "source": "override"}
        return Config(tree)

    def as_dict(self) -> dict[str, Any]:
        tree = copy.deepcopy(self.tree)
        for path, _ in self.leaves():
            parts = path.split(".")
            cur = tree
            for part in parts[:-1]:
                cur = cur[part]
            cur[parts[-1]]["placeholder"] = self.is_placeholder(path)
        return tree


def load_config(config_dir: Path | str | None = None) -> Config:
    d = Path(config_dir) if config_dir else CONFIG_DIR
    tree: dict[str, Any] = {}
    for name in SECTIONS:
        with open(d / f"{name}.yaml", encoding="utf-8") as fh:
            tree[name] = yaml.safe_load(fh) or {}
    return Config(tree)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
