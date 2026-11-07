"""Check the bundled core's manifest and, optionally, the R source checkout."""

import argparse
import hashlib
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--r-source", type=Path,
                    help="R checkout whose authoritative core must match")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
directory = root / "src" / "ibist" / "_native"
core = directory / "wang_core.h"
manifest = json.loads((directory / "wang_core.json").read_text())
content = core.read_bytes()
if hashlib.md5(content).hexdigest() != manifest["md5"]:
    raise SystemExit("Bundled Wang core does not match its provenance hash.")
if args.r_source is not None:
    authoritative = args.r_source / "src" / "wang_core.h"
    if not authoritative.is_file() or authoritative.read_bytes() != content:
        raise SystemExit(
            "Bundled Wang core differs from the R source checkout.")
print("Wang core synchronization check passed.")
