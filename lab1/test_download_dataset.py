from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from download_dataset import SOURCE_FILENAME, find_source_member, install_archive


class KaggleArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.archive = self.root / "kaggle-download.zip"
        self.payload = (
            b"pkSeqID,proto,sport,dport,DeviceName,Attack,AttackSubType,label\n"
            b"1,tcp,12345,80,Device0,Normal,Normal,0\r\n"
        )
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr(f"nested/{SOURCE_FILENAME}", self.payload)
            zipped.writestr("README.txt", b"Kaggle dataset archive")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_extracts_exact_source_bytes_and_writes_provenance(self) -> None:
        output = self.root / "data" / SOURCE_FILENAME

        provenance = install_archive(self.archive, output)

        self.assertEqual(output.read_bytes(), self.payload)
        self.assertEqual(provenance["source_bytes"], len(self.payload))
        self.assertEqual(provenance["archive_member"], f"nested/{SOURCE_FILENAME}")
        self.assertEqual(
            json.loads(output.with_suffix(output.suffix + ".source.json").read_text("utf-8")),
            provenance,
        )

    def test_refuses_existing_output_unless_force_is_set(self) -> None:
        output = self.root / SOURCE_FILENAME
        output.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            install_archive(self.archive, output)
        install_archive(self.archive, output, force=True)
        self.assertEqual(output.read_bytes(), self.payload)

    def test_rejects_archives_missing_expected_member(self) -> None:
        wrong = self.root / "wrong.zip"
        with zipfile.ZipFile(wrong, "w") as zipped:
            zipped.writestr("different.csv", b"not the assigned data")
        with self.assertRaisesRegex(ValueError, "found 0"):
            find_source_member(wrong)

    def test_rejects_ambiguous_duplicate_source_members(self) -> None:
        duplicate = self.root / "duplicate.zip"
        with zipfile.ZipFile(duplicate, "w") as zipped:
            zipped.writestr(SOURCE_FILENAME, b"first")
            zipped.writestr(f"nested/{SOURCE_FILENAME}", b"second")
        with self.assertRaisesRegex(ValueError, "found 2"):
            find_source_member(duplicate)

    def test_rejects_unsafe_member_path(self) -> None:
        malicious = self.root / "malicious.zip"
        with zipfile.ZipFile(malicious, "w") as zipped:
            zipped.writestr(f"../{SOURCE_FILENAME}", self.payload)
        with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
            find_source_member(malicious)


if __name__ == "__main__":
    unittest.main()
