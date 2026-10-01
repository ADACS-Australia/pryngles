#!/usr/bin/env python3
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _norm_req(req: str) -> str:
    return req.strip()


def _iter_requirement_files(root: Path) -> list[Path]:
    requirement_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).resolve().relative_to(root)
        parts = set(rel.parts)

        # Always avoid contrib, and skip common virtualenv/build folders.
        if "contrib" in parts or ".venv" in parts or ".git" in parts or "build" in parts or "dist" in parts:
            dirnames[:] = []
            continue

        # Also avoid walking into those directories if present as direct children.
        dirnames[:] = [d for d in dirnames if d not in {"contrib", ".venv", ".git", "build", "dist", "__pycache__"}]

        for fn in filenames:
            if fn.startswith("requirements") and fn.endswith(".txt"):
                requirement_files.append(Path(dirpath) / fn)

    requirement_files.sort(key=lambda p: str(p))
    return requirement_files


def _parse_requirements_txt(path: Path) -> list[str]:
    reqs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Keep pip directives as-is (e.g., -r, -e, --find-links).
        reqs.append(line)
    return reqs


def _extract_str_list_from_ast(node: ast.AST) -> list[str] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        items: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                items.append(elt.value)
            else:
                return None
        return items
    return None


def _parse_setup_py_install_requires(path: Path) -> list[str]:
    """
    Best-effort extraction of `install_requires=[...]` from setup.py without executing it.
    Only supports literal list/tuple of strings.
    """
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    install_requires: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Look for setuptools.setup(...) or setup(...)
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name != "setup":
            continue

        for kw in node.keywords or []:
            if kw.arg != "install_requires":
                continue
            extracted = _extract_str_list_from_ast(kw.value)
            if extracted:
                install_requires.extend(extracted)

    return install_requires


def _parse_pyproject_build_requires(path: Path) -> list[str]:
    """
    Extracts build-system.requires from pyproject.toml.
    """
    try:
        import tomllib  # py3.11+
    except Exception:  # pragma: no cover
        return []

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    reqs = data.get("build-system", {}).get("requires", [])
    if isinstance(reqs, list) and all(isinstance(x, str) for x in reqs):
        return list(reqs)
    return []


_REQ_SPLIT_RE = re.compile(r"\s+")


def _clean_req_line(req: str) -> str:
    # Collapse weird whitespace; keep original operators/specifiers.
    return _REQ_SPLIT_RE.sub(" ", req).strip()


def generate_requirements_txt(root: Path) -> str:
    reqs: list[str] = []

    setup_py = root / "setup.py"
    if setup_py.exists():
        reqs.extend(_parse_setup_py_install_requires(setup_py))

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        reqs.extend(_parse_pyproject_build_requires(pyproject))

    for req_file in _iter_requirement_files(root):
        # Don't treat the generated file as an input if it already exists.
        if req_file.resolve() == (root / "requirements.txt").resolve():
            continue
        reqs.extend(_parse_requirements_txt(req_file))

    # Normalize, dedupe (preserve first-seen order).
    out: list[str] = []
    seen: set[str] = set()
    for r in reqs:
        r = _clean_req_line(_norm_req(r))
        if not r:
            continue
        key = r.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    return "\n".join(out) + ("\n" if out else "")


def main(argv: list[str]) -> int:
    out_path = ROOT / "requirements.txt"
    contents = generate_requirements_txt(ROOT)
    out_path.write_text(contents, encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)} ({len(contents.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

