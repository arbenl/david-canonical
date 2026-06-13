#!/usr/bin/env python3
"""Validate the DAVID documentation publication rule.

The rule is deliberately mechanical: active formal math sources are TeX/PDF
pairs, while active Markdown files under docs/ must be registered as summaries,
indices, ledgers, prompts, or operational documents.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ALLOWED_MARKDOWN_ROLES = {
    "architecture_blueprint",
    "audit_summary",
    "escalation_ledger",
    "gate_index",
    "human_process",
    "implementation_index",
    "implementation_prompt",
    "implementation_report",
    "operational_contract",
    "planning_artifact",
    "pedagogical_deep_dive",
    "review_gate_summary",
    "summary_of_latex_source",
    "theorem_index",
    "tracker",
}

ALLOWED_LATEX_ROLES = {"formal_math_source"}

FILE_URL_RE = re.compile(r"file://[^\s)\]}]*?/(?P<path>docs/[^\s)\]}]+)")
ACTIVE_DOC_SUFFIXES = {".md", ".pdf", ".tex"}
EXCLUDED_ACTIVE_PREFIXES = ("docs/archive/",)


def _repo_path(repo: Path, value: str, field: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} must be a repository-relative path: {value}")
    if not candidate.parts or candidate.parts[0] != "docs":
        raise ValueError(f"{field} must live under docs/: {value}")
    return repo / candidate


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema_version") != 1:
        raise ValueError("docs publication manifest schema_version must be 1")
    return manifest


def _is_active_doc(path: str) -> bool:
    return (
        path.startswith("docs/")
        and Path(path).suffix in ACTIVE_DOC_SUFFIXES
        and not any(path.startswith(prefix) for prefix in EXCLUDED_ACTIVE_PREFIXES)
    )


def _active_doc_paths(repo: Path) -> set[str]:
    git_result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "--", "docs"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if git_result.returncode == 0:
        return {
            path
            for path in git_result.stdout.splitlines()
            if _is_active_doc(path) and (repo / path).is_file()
        }

    return {
        path.relative_to(repo).as_posix()
        for path in (repo / "docs").rglob("*")
        if path.is_file() and _is_active_doc(path.relative_to(repo).as_posix())
    }


def validate_publication_rule(repo: Path) -> list[str]:
    repo = repo.resolve()
    manifest_path = repo / "docs" / "publication_manifest.json"
    errors: list[str] = []

    try:
        manifest = _load_manifest(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"{manifest_path.relative_to(repo)}: {exc}"]

    latex_docs = manifest.get("latex_documents")
    markdown_docs = manifest.get("markdown_documents")
    if not isinstance(latex_docs, list):
        errors.append("latex_documents must be a list")
        latex_docs = []
    if not isinstance(markdown_docs, list):
        errors.append("markdown_documents must be a list")
        markdown_docs = []

    active_docs = _active_doc_paths(repo)
    active_tex = {path for path in active_docs if path.endswith(".tex")}
    active_pdf = {path for path in active_docs if path.endswith(".pdf")}
    active_md = {path for path in active_docs if path.endswith(".md")}

    manifest_tex: set[str] = set()
    manifest_pdf: set[str] = set()
    for index, entry in enumerate(latex_docs):
        if not isinstance(entry, dict):
            errors.append(f"latex_documents[{index}] must be an object")
            continue
        role = entry.get("role")
        if role not in ALLOWED_LATEX_ROLES:
            errors.append(f"latex_documents[{index}].role is invalid: {role}")
        tex = entry.get("tex")
        pdf = entry.get("pdf")
        if not isinstance(tex, str) or not isinstance(pdf, str):
            errors.append(f"latex_documents[{index}] must declare tex and pdf")
            continue
        manifest_tex.add(tex)
        manifest_pdf.add(pdf)
        expected_pdf = Path(tex).with_suffix(".pdf").as_posix()
        if pdf != expected_pdf:
            errors.append(
                f"latex_documents[{index}] pairs {tex} with {pdf}; expected corresponding PDF {expected_pdf}"
            )
        for field, value in (("tex", tex), ("pdf", pdf)):
            try:
                path = _repo_path(repo, value, f"latex_documents[{index}].{field}")
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not path.exists():
                errors.append(f"{value} is registered but missing")

    for tex in sorted(active_tex - manifest_tex):
        errors.append(f"{tex} is an active TeX source but is missing from docs/publication_manifest.json")
    for pdf in sorted(active_pdf - manifest_pdf):
        errors.append(f"{pdf} is an active PDF but is missing from docs/publication_manifest.json")
    for tex in sorted(manifest_tex - active_tex):
        errors.append(f"{tex} is registered but is not an active docs/*.tex file")
    for pdf in sorted(manifest_pdf - active_pdf):
        errors.append(f"{pdf} is registered but is not an active docs/*.pdf file")

    manifest_md: set[str] = set()
    for index, entry in enumerate(markdown_docs):
        if not isinstance(entry, dict):
            errors.append(f"markdown_documents[{index}] must be an object")
            continue
        role = entry.get("role")
        if role not in ALLOWED_MARKDOWN_ROLES:
            errors.append(f"markdown_documents[{index}].role is invalid: {role}")
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            errors.append(f"markdown_documents[{index}].path must be a string")
            continue
        manifest_md.add(path_value)
        try:
            path = _repo_path(repo, path_value, f"markdown_documents[{index}].path")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.exists():
            errors.append(f"{path_value} is registered but missing")

        for source_field in ("source_tex", "source_pdf"):
            source_value = entry.get(source_field)
            if source_value is None:
                continue
            if not isinstance(source_value, str):
                errors.append(f"markdown_documents[{index}].{source_field} must be a string")
                continue
            try:
                source_path = _repo_path(repo, source_value, f"markdown_documents[{index}].{source_field}")
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not source_path.exists():
                errors.append(f"{path_value} references missing {source_field}: {source_value}")

    for md in sorted(active_md - manifest_md):
        errors.append(f"{md} is an active Markdown doc but is missing from docs/publication_manifest.json")
    for md in sorted(manifest_md - active_md):
        errors.append(f"{md} is registered but is not an active docs/*.md file")

    active_text_docs = active_md | active_tex
    for text_doc in sorted(active_text_docs):
        path = repo / text_doc
        text = path.read_text(encoding="utf-8")
        for match in FILE_URL_RE.finditer(text):
            link_value = match.group("path")
            if not (repo / link_value).exists():
                errors.append(f"{text_doc} contains file:// link to missing active artifact: {link_value}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    errors = validate_publication_rule(args.repo)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("docs publication rule: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
