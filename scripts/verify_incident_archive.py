#!/usr/bin/env python3
"""Verify the immutable historical ZIP, not a live tree that has since evolved."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def verify(root):
    folder = root / "docs/incidents"
    archive = folder / "problem-package.zip"
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member")
        manifest_bytes = handle.read("ProbeGRPO/docs/incidents/bundle-manifest.json")
        if manifest_bytes != (folder / "bundle-manifest.json").read_bytes():
            raise ValueError("external manifest differs from archived manifest")
        manifest = json.loads(manifest_bytes)
        for row in manifest["files"]:
            relative = PurePosixPath(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe archived source path")
            raw = handle.read("ProbeGRPO/" + relative.as_posix())
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"archive hash mismatch: {relative}")
        ledger = json.loads(handle.read("ProbeGRPO/docs/incidents/incidents.json"))
        ids = [item["id"] for item in ledger["incidents"]]
        if len(ids) != len(set(ids)) or len(ids) != manifest["incident_count"]:
            raise ValueError("incident ledger count mismatch")
    return {"status": "passed", "files": len(manifest["files"]), "incidents": len(ids),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "scope": "immutable historical archive; does not require current source to equal the old snapshot"}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
