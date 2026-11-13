"""Package the assignment materials as Assignment03_<Surname>_Materials.zip.

The archive contains exactly the files tracked by git at HEAD (sources,
scripts, build files, raw data, tables, figures, vectorisation evidence,
README and report) under a top-level folder of the same name. Untracked and
ignored files (build/, binaries, data/*.bin, .claude/) are never included.

The working tree must be clean, so that the archive matches a commit; pass
--allow-dirty to package the committed versions anyway (uncommitted changes
are then NOT in the archive, because files are read from HEAD).

Usage:
    python scripts/make_zip.py --surname <Surname> [--out-dir DIR] [--allow-dirty]
"""

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          check=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--surname", required=True, help="surname used in the archive name")
    ap.add_argument("--out-dir", type=Path, default=ROOT, help="where to write the zip")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="package HEAD even if the working tree has changes")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z][A-Za-z-]*", args.surname):
        sys.exit("error: --surname must contain only ASCII letters and hyphens")
    name = f"Assignment03_{args.surname}_Materials"

    dirty = git("status", "--porcelain").decode().strip()
    if dirty and not args.allow_dirty:
        sys.exit("error: uncommitted changes; commit first or pass --allow-dirty\n" + dirty)

    files = [f for f in git("ls-tree", "-r", "--name-only", "-z", "HEAD").decode().split("\0") if f]
    commit = git("rev-parse", "--short", "HEAD").decode().strip()
    out = args.out_dir / f"{name}.zip"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            # Content from HEAD (not the working tree), so the archive matches the commit.
            data = git("show", f"HEAD:{rel}")
            info = zipfile.ZipInfo(f"{name}/{rel}")
            info.external_attr = 0o644 << 16  # rw-r--r-- when extracted on POSIX
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)

    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
        count = len(zf.namelist())
    if bad is not None or count != len(files):
        sys.exit(f"error: archive check failed (bad member: {bad}, {count}/{len(files)} files)")
    print(f"{out} : {count} files from commit {commit}, {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
