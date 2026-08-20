#!/usr/bin/env python3
"""Prepare writable runtime files, then run the configured AIC command."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

runtime = Path(os.environ.get("AIC_CACHE_DIR", "/runtime"))
supplied_indexes = Path("/supplied-indexes")
target_indexes = runtime / "indexes"

# Keep index refreshes in the writable runtime mount.
target_indexes.mkdir(parents=True, exist_ok=True)
(runtime / "extracted").mkdir(parents=True, exist_ok=True)
for name in ("siglip_5fps_flat_ip.faiss", "siglip_5fps_manifest.json"):
    source = supplied_indexes / name
    target = target_indexes / name
    if source.is_file() and not target.exists():
        shutil.copy2(source, target)

os.execvp(sys.argv[1], sys.argv[1:])
