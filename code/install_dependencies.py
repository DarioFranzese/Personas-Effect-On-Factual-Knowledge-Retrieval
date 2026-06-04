#!/usr/bin/env python3
"""Scan the code directory for imports and install missing third-party packages."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE_ORDER = [
    "torch",
    "vllm",
    "datasets",
    "pydantic",
    "tqdm",
    "openai",
    "tiktoken",
    "nltk",
]
IMPORT_LINE = re.compile(r"^(?:from\s+|import\s+)")

if hasattr(sys, "stdlib_module_names"):
    STDLIB_MODULES = set(sys.stdlib_module_names)
else:
    STDLIB_MODULES = {
        "abc",
        "argparse",
        "ast",
        "collections",
        "dataclasses",
        "enum",
        "importlib",
        "json",
        "logging",
        "os",
        "pathlib",
        "random",
        "re",
        "subprocess",
        "sys",
        "time",
        "typing",
    }

LOCAL_MODULES = {path.stem for path in ROOT.glob("*.py")}


def iter_source_texts() -> list[tuple[Path, str]]:
    sources: list[tuple[Path, str]] = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix == ".py":
            sources.append((path, path.read_text(encoding="utf-8")))
            continue
        if path.suffix == ".ipynb":
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[warn] skip notebook non leggibile {path.name}: {exc}")
                continue

            cell_sources: list[str] = []
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                cell_sources.append(source)

            sources.append((path, "\n".join(cell_sources)))
    return sources


def extract_imports(source_text: str) -> set[str]:
    imports: set[str] = set()
    for raw_line in source_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or not IMPORT_LINE.match(line):
            continue

        if line.startswith("import "):
            remainder = line[len("import ") :]
            for segment in remainder.split(","):
                module = segment.strip().split(" as ", 1)[0].strip()
                if not module:
                    continue
                imports.add(module.split(".", 1)[0])
            continue

        from_match = re.match(r"from\s+([.\w]+)\s+import\s+", line)
        if not from_match:
            continue
        module = from_match.group(1)
        if module.startswith("."):
            continue
        imports.add(module.split(".", 1)[0])

    return imports


def discover_third_party_modules() -> set[str]:
    discovered: set[str] = set()
    for _, source_text in iter_source_texts():
        for module in extract_imports(source_text):
            if module in STDLIB_MODULES:
                continue
            if module in LOCAL_MODULES:
                continue
            discovered.add(module)
    return discovered


def ordered_packages(packages: set[str]) -> list[str]:
    priority = {name: index for index, name in enumerate(PACKAGE_ORDER)}
    return sorted(packages, key=lambda name: (priority.get(name, len(PACKAGE_ORDER)), name))


def is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def install_package(module_name: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", module_name], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scansiona code/ e installa le dipendenze esterne mancanti."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra solo cosa verrebbe installato senza eseguire pip.",
    )
    args = parser.parse_args()

    discovered = discover_third_party_modules()
    missing = [module for module in ordered_packages(discovered) if not is_installed(module)]

    if not missing:
        print("Nessuna dipendenza esterna mancante rilevata.")
        return 0

    print("Dipendenze mancanti rilevate in ordine di installazione:")
    for module in missing:
        print(f"- {module}")

    if args.dry_run:
        return 0

    for module in missing:
        print(f"Installo {module}...")
        install_package(module)

    print("Installazione completata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())