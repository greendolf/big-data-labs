"""Install the original Kaggle IDS CSV from a browser-downloaded archive.

Kaggle authentication is intentionally handled by the user in the browser. This
script never requests credentials or calls the Kaggle API; it extracts only the
expected CSV member and records a local provenance sidecar under ``data/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


DATASET_ID = "azalhowaide/iot-dataset-for-intrusion-detection-systems-ids"
SOURCE_FILENAME = "BoTNeTIoT-L01-v2.csv"
DEFAULT_OUTPUT = Path("data/raw/ids") / SOURCE_FILENAME


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> bool:
    """Return false for absolute paths and traversal entries in ZIP archives."""
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts and not name.startswith("/")


def find_source_member(archive: Path) -> str:
    """Find the unique expected CSV in a Kaggle ZIP without extracting the ZIP."""
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a readable ZIP archive: {archive}")
    with zipfile.ZipFile(archive) as zipped:
        matches = [
            info.filename
            for info in zipped.infolist()
            if PurePosixPath(info.filename.replace("\\", "/")).name == SOURCE_FILENAME
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {SOURCE_FILENAME} in {archive.name}; found {len(matches)}"
            )
        member = matches[0]
        if not _safe_member_name(member):
            raise ValueError(f"Unsafe archive member path: {member!r}")
        if zipped.getinfo(member).is_dir():
            raise ValueError(f"Archive member is a directory, not a CSV: {member}")
        return member


def install_archive(archive: Path, output: Path, *, force: bool = False) -> dict[str, object]:
    """Extract only the source CSV atomically and write exact-file provenance."""
    archive = archive.resolve()
    output = output.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Archive does not exist: {archive}")
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")

    member = find_source_member(archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(archive) as zipped, zipped.open(member, "r") as source:
            with temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
        if temporary.stat().st_size == 0:
            raise ValueError(f"The source CSV in {archive.name} is empty")
        source_sha = sha256_file(temporary)
        # os.replace is atomic on the same volume and safely replaces output
        # when --force was requested.
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    provenance: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_url": f"https://www.kaggle.com/datasets/{DATASET_ID}",
        "source_filename": SOURCE_FILENAME,
        "archive_filename": archive.name,
        "archive_member": member,
        "source_bytes": output.stat().st_size,
        "source_sha256": source_sha,
        "download_method": "Kaggle browser download; local ZIP extraction",
    }
    provenance_path = output.with_suffix(output.suffix + ".source.json")
    provenance_tmp = provenance_path.with_suffix(provenance_path.suffix + ".partial")
    try:
        provenance_tmp.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        provenance_tmp.replace(provenance_path)
    finally:
        provenance_tmp.unlink(missing_ok=True)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Path to the ZIP downloaded from the Kaggle dataset page",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination for the original {SOURCE_FILENAME} (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing output files")
    args = parser.parse_args()
    if args.archive is None:
        parser.error(
            "--archive is required. Download the dataset ZIP in the signed-in Kaggle browser, "
            f"then run: python lab1/download_dataset.py --archive <downloaded-archive.zip>"
        )
    provenance = install_archive(args.archive, args.output, force=args.force)
    print(
        f"Installed {provenance['source_filename']}: {provenance['source_bytes']:,} bytes, "
        f"SHA-256 {provenance['source_sha256']}"
    )
    print(f"Dataset: {DATASET_ID}; member: {provenance['archive_member']}")


if __name__ == "__main__":
    main()
