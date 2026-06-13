from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_docs_publication_rule.py"
SPEC = importlib.util.spec_from_file_location("check_docs_publication_rule", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _copy_minimal_docs_tree(tmp: Path) -> Path:
    repo = tmp / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    for name in (
        "publication_manifest.json",
        "thesis_mathematical_core.tex",
        "thesis_mathematical_core.pdf",
        "unified_tensor_hsmm_framework.tex",
        "unified_tensor_hsmm_framework.pdf",
    ):
        shutil.copy2(REPO / "docs" / name, docs / name)
    manifest = json.loads((docs / "publication_manifest.json").read_text(encoding="utf-8"))
    manifest["markdown_documents"] = [
        {"path": "docs/INDEX.md", "role": "theorem_index", "source_tex": "docs/thesis_mathematical_core.tex"}
    ]
    (docs / "publication_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (docs / "INDEX.md").write_text("# Index\n", encoding="utf-8")
    return repo


def test_current_docs_publication_rule_passes() -> None:
    assert MODULE.validate_publication_rule(REPO) == []


def test_unregistered_markdown_doc_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        repo = _copy_minimal_docs_tree(Path(tmp_name))
        (repo / "docs" / "new_theorem.md").write_text("# New theorem\n", encoding="utf-8")

        errors = MODULE.validate_publication_rule(repo)

    assert any("new_theorem.md is an active Markdown doc but is missing" in error for error in errors)


def test_unregistered_nested_markdown_doc_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        repo = _copy_minimal_docs_tree(Path(tmp_name))
        nested = repo / "docs" / "math" / "new_theorem.md"
        nested.parent.mkdir()
        nested.write_text("# New theorem\n", encoding="utf-8")

        errors = MODULE.validate_publication_rule(repo)

    assert any("docs/math/new_theorem.md is an active Markdown doc but is missing" in error for error in errors)


def test_missing_file_url_target_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        repo = _copy_minimal_docs_tree(Path(tmp_name))
        (repo / "docs" / "INDEX.md").write_text(
            "[missing](file:///tmp/repo/docs/missing.pdf)\n",
            encoding="utf-8",
        )

        errors = MODULE.validate_publication_rule(repo)

    assert any("missing active artifact: docs/missing.pdf" in error for error in errors)


def test_missing_file_url_target_in_tex_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        repo = _copy_minimal_docs_tree(Path(tmp_name))
        (repo / "docs" / "thesis_mathematical_core.tex").write_text(
            "\\href{file:///tmp/repo/docs/missing.pdf}{missing}\n",
            encoding="utf-8",
        )

        errors = MODULE.validate_publication_rule(repo)

    assert any("docs/thesis_mathematical_core.tex contains file:// link to missing" in error for error in errors)


def test_latex_source_must_pair_with_corresponding_pdf() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        repo = _copy_minimal_docs_tree(Path(tmp_name))
        docs = repo / "docs"
        (docs / "new_formal.tex").write_text("\\section{New}\n", encoding="utf-8")
        manifest_path = docs / "publication_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["latex_documents"].append(
            {
                "tex": "docs/new_formal.tex",
                "pdf": "docs/thesis_mathematical_core.pdf",
                "role": "formal_math_source",
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        errors = MODULE.validate_publication_rule(repo)

    assert any("expected corresponding PDF docs/new_formal.pdf" in error for error in errors)


def test_active_tex_requires_manifest_entry_and_pdf_pair() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        repo = _copy_minimal_docs_tree(Path(tmp_name))
        (repo / "docs" / "new_formal.tex").write_text("\\section{New}\n", encoding="utf-8")

        errors = MODULE.validate_publication_rule(repo)

    assert any("new_formal.tex is an active TeX source but is missing" in error for error in errors)


if __name__ == "__main__":
    test_current_docs_publication_rule_passes()
    test_unregistered_markdown_doc_fails()
    test_unregistered_nested_markdown_doc_fails()
    test_missing_file_url_target_fails()
    test_missing_file_url_target_in_tex_fails()
    test_latex_source_must_pair_with_corresponding_pdf()
    test_active_tex_requires_manifest_entry_and_pdf_pair()
