"""
fix_meta.py
===========

One-off repair utility for MLflow's filesystem tracking backend (``./mlruns``).

Why this exists
---------------
MLflow has put the file store into maintenance mode. It still runs if you set
``MLFLOW_ALLOW_FILE_STORE=true``, but it no longer gets compatibility fixes, so
older ``meta.yaml`` files can drift out of sync with the current ``RunInfo``
entity schema.

Two specific breakages show up in this project:

1. Missing ``run_uuid``.
   ``FileStore._get_run_info_from_dir`` reads ``meta.yaml``, filters it down to
   the properties ``RunInfo`` declares, and passes the result to the
   constructor. ``RunInfo.__init__`` still requires the legacy ``run_uuid``
   positional argument, but runs written by older MLflow versions only record
   ``run_id``. The filtered dict comes up one argument short and any call that
   touches the run (``search_runs``, ``get_run``, the UI, and the
   ``mlflow migrate-filestore`` tool itself) dies with::

       TypeError: RunInfo.__init__() missing 1 required positional argument: 'run_uuid'

   This script backfills ``run_uuid`` from the existing ``run_id``. They are the
   same value; only the key name changed.

2. Stale absolute artifact paths.
   ``artifact_uri`` and ``artifact_location`` are stored as absolute paths, so
   renaming or copying the project directory leaves every run pointing at the
   old location. Artifacts logged after the move land somewhere unexpected or
   fail silently. The path rewrite corrects those in place.

Is this required?
-----------------
No. It is optional and situational:

* Not needed for a fresh ``mlruns`` tree created by the current MLflow version.
* Not needed at all if the project is on a database backend
  (``sqlite:///mlflow.db``), which is the direction MLflow is pushing everyone.
* Needed only when reading legacy runs through the file store, or as a
  prerequisite to migrating them, since ``mlflow migrate-filestore`` reads
  through the same broken code path and will hit the identical error.

Nothing in the training or inference pipeline imports this module. It is a
manual maintenance step, not part of the DAG.

Safety
------
Edits ``meta.yaml`` files in place. Back up first::

    cp -r mlruns mlruns.bak

The run_uuid backfill is idempotent; files that already have the key are
skipped. Re-running is harmless.

Usage
-----
    python fix_meta.py

Then, if artifact paths also moved::

    grep -rl 'OLD_PROJECT_NAME' mlruns/ \\
        | xargs sed -i 's|/OLD_PROJECT_NAME/|/NEW_PROJECT_NAME/|g'
"""

from pathlib import Path
import re

root = Path("mlruns")
patched = 0
for meta in root.rglob("meta.yaml"):
    text = meta.read_text()
    if "run_uuid:" in text or "run_id:" not in text:
        continue
    m = re.search(r"^run_id:\s*(\S+)\s*$", text, re.M)
    if not m:
        continue
    meta.write_text(text.rstrip("\n") + f"\nrun_uuid: {m.group(1)}\n")
    patched += 1
print(f"patched {patched}")
