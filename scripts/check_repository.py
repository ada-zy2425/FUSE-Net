#!/usr/bin/env python3
"""Check packaging integrity without ML imports, downloads, or dataset loading."""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    files = []
    for directory in (root / "FUSENet", root / "fusenetpp", root / "scripts"):
        files.extend(directory.rglob("*.py"))
    files.extend(root.glob("*.py"))
    for path in sorted(files):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            compile(tree, str(path), "exec")
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors.append("{}: {}".format(path.relative_to(root), exc))

    try:
        manifest = json.loads((root / "docs/source_manifest.json").read_text(encoding="utf-8"))
        entries = manifest["files"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("Source manifest has no file entries")
        seen = set()
        for item in entries:
            relative = item["path"]
            path = (root / relative).resolve()
            if root not in path.parents or relative in seen:
                raise ValueError("Invalid or duplicate manifest path: {}".format(relative))
            seen.add(relative)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                errors.append("Source integrity mismatch: {}".format(relative))
            if len(data) != item["size_bytes"]:
                errors.append("Source size mismatch: {}".format(relative))
        expected = {str(p.relative_to(root)) for p in (root / "FUSENet").rglob("*.py")}
        expected.add("docs/paper/FUSE-Net.pdf")
        if seen != expected:
            errors.append("Manifest does not exactly cover canonical source and PDF")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        errors.append("Manifest: {}".format(exc))
        entries = []

    documents = list(root.glob("*.md")) + list((root / "docs").rglob("*.md"))
    link_count = 0
    for document in documents:
        content = document.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", content):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            link_count += 1
            if not (document.parent / unquote(parsed.path)).exists():
                errors.append("Broken local link in {}: {}".format(document.relative_to(root), target))

    if errors:
        for error in errors:
            print("FAIL:", error, file=sys.stderr)
        return 1
    print("PASS: {} Python files parse and compile".format(len(files)))
    print("PASS: {} original source/PDF files match their manifest".format(len(entries)))
    print("PASS: {} local documentation links resolve".format(link_count))
    print("Scope: packaging checks only; training and paper results are not validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
