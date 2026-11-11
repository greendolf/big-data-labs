"""Lab 2 Part 1: preparation of the IoT IDS dataset assigned in Lab 1.

The module builds the cleaned benchmark input from the Kaggle CSV that Lab 1
downloaded, runs a Great Expectations quality suite, and creates the byte-size
slices required by the assignment (100 MB for local testing, 1 GB for the
Pandas/Dask comparison, and the full dataset for Spark).

Every artifact is cached under the ignored ``data/lab2/`` directory. The source
file is verified against the SHA-256 digest recorded by Lab 1, so a corrupted
or replaced download fails fast instead of silently changing benchmark inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import duckdb

SOURCE_BYTES = 1_678_174_220
TARGET_BYTES_PER_MB = 1_000_000

# Assignment task 1: one aggregation over a complex key with several aggregates.
KEY_COLUMNS = ["Attack", "Attack_subType", "Device_Name"]
MEAN_COLUMNS = ["MI_dir_L0.1_mean", "HH_L0.1_magnitude"]
MAX_COLUMN = "H_L0.1_variance"
LABEL_COLUMN = "label"

# Domains enumerated from the complete Lab 1 audit (category_counts.csv): the
# three Attack values and nine Attack_subType values cover all 7,062,606 rows.
ATTACK_DOMAIN = ["Normal", "gafgyt", "mirai"]
ATTACK_SUBTYPE_DOMAIN = [
    "Normal", "ack", "combo", "junk", "scan", "syn", "tcp", "udp", "udpplain",
]

TEXT_COLUMNS = ["Device_Name", "Attack", "Attack_subType"]
VARIANCE_COLUMNS = ["MI_dir_L0.1_variance", "H_L0.1_variance", "HH_jit_L0.1_variance"]
STD_COLUMNS = ["HH_L0.1_std", "HpHp_L0.1_std"]
PCC_COLUMNS = ["HH_L0.1_pcc", "HpHp_L0.1_pcc"]


def sha256_of(path: Path, chunk_bytes: int = 8 * 1_048_576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(root: Path) -> Path:
    override = os.environ.get("LAB2_SOURCE_CSV")
    source = Path(override).resolve() if override else root / "data" / "raw" / "ids" / "BoTNeTIoT-L01-v2.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"Kaggle CSV not found: {source}. Download the dataset while signed in to Kaggle, "
            "run lab1/download_dataset.py first, or set LAB2_SOURCE_CSV."
        )
    return source


def expected_sha256(source: Path) -> str | None:
    provenance = source.with_name(source.name + ".source.json")
    if provenance.is_file():
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        digest = payload.get("source_sha256")
        if isinstance(digest, str) and len(digest) == 64:
            return digest
    return None


def verify_source(source: Path) -> dict[str, Any]:
    digest = sha256_of(source)
    expected = expected_sha256(source)
    if expected and digest != expected:
        raise ValueError(
            f"Source CSV digest {digest} does not match the Lab 1 provenance record {expected}"
        )
    return {"path": str(source), "bytes": source.stat().st_size, "sha256": digest}


def _columns_of(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()]


def build_cleaned(source: Path, output_dir: Path, force: bool = False) -> dict[str, Any]:
    """Deduplicate the full CSV, keep every surviving row, and write CSV + Parquet copies."""
    cleaned_csv = output_dir / "ids_cleaned.csv"
    cleaned_parquet = output_dir / "ids_cleaned.parquet"
    stats_path = output_dir / "cleaned_stats.json"
    if not force and cleaned_csv.is_file() and cleaned_parquet.is_file() and stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["reused"] = True
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    try:
        con.execute(
            f"""
            CREATE TEMP TABLE raw AS
            SELECT * FROM read_csv(
                '{str(source).replace("'", "''")}',
                header=true, union_by_name=false, ignore_errors=false
            )
            """
        )
        columns = _columns_of(con, "raw")
        raw_rows = int(con.execute("SELECT count(*) FROM raw").fetchone()[0])
        trimmed = ", ".join(
            f'trim("{column}") AS "{column}"' if column in TEXT_COLUMNS else f'"{column}"'
            for column in columns
        )
        con.execute(f'CREATE TEMP TABLE cleaned AS SELECT DISTINCT {trimmed} FROM raw')
        cleaned_rows = int(con.execute("SELECT count(*) FROM cleaned").fetchone()[0])
        null_counts = {
            column: int(con.execute(f'SELECT count(*) FROM cleaned WHERE "{column}" IS NULL').fetchone()[0])
            for column in columns
        }
        attack_share = con.execute(
            'SELECT "Attack", count(*) AS rows FROM cleaned GROUP BY 1 ORDER BY rows DESC'
        ).fetchall()
        label_share = con.execute(
            f'SELECT "{LABEL_COLUMN}", count(*) AS rows FROM cleaned GROUP BY 1 ORDER BY rows DESC'
        ).fetchall()
        mismatch = int(con.execute(
            f'SELECT count(*) FROM cleaned WHERE ("Attack" = \'Normal\') <> ("{LABEL_COLUMN}" = 1)'
        ).fetchone()[0])

        con.execute(f"COPY cleaned TO '{str(cleaned_csv).replace(chr(39), chr(39)*2)}' (HEADER, FORMAT CSV)")
        con.execute(
            f"COPY cleaned TO '{str(cleaned_parquet).replace(chr(39), chr(39)*2)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        stats = {
            "columns": columns,
            "raw_rows": raw_rows,
            "cleaned_rows": cleaned_rows,
            "duplicate_excess_rows": raw_rows - cleaned_rows,
            "null_counts_after_cleaning": null_counts,
            "rows_with_any_null": sum(null_counts.values()),
            "attack_share_after_cleaning": [
                {"attack": attack, "rows": int(rows)} for attack, rows in attack_share
            ],
            "label_share_after_cleaning": [
                {"label": int(label), "rows": int(rows)} for label, rows in label_share
            ],
            "label_attack_mismatch_rows": mismatch,
            "cleaned_csv_bytes": cleaned_csv.stat().st_size,
            "cleaned_parquet_bytes": cleaned_parquet.stat().st_size,
        }
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        stats["reused"] = False
        return stats
    finally:
        con.close()


def create_prefix_slices(source_csv: Path, output_dir: Path, sizes_mb: list[int]) -> dict[int, Path]:
    """Cut byte-exact prefix slices on newline boundaries from the cleaned CSV.

    The Lab 1 dataset has more than 6.4 million real rows, so slices are real
    data prefixes; unlike the pre-rework draft no synthetic row replay is used.
    """
    if not sizes_mb or any(not isinstance(size, int) or size <= 0 for size in sizes_mb):
        raise ValueError("sizes_mb must contain positive integer decimal megabyte targets")
    if len(set(sizes_mb)) != len(sizes_mb):
        raise ValueError("sizes_mb must not contain duplicates")
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    source_bytes = source_csv.stat().st_size
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {size: size * TARGET_BYTES_PER_MB for size in sizes_mb}
    if max(targets.values()) >= source_bytes:
        raise ValueError("every slice target must be smaller than the source CSV")

    paths = {size: output_dir / f"ids_slice_{size}MB.csv" for size in sizes_mb}
    window = 1 << 20
    with source_csv.open("rb") as reader:
        for size in sizes_mb:
            target = targets[size]
            reader.seek(target - min(window, target))
            block = reader.read(min(window, target))
            boundary = block.rfind(b"\n")
            if boundary < 0:
                raise ValueError(f"no CSV row boundary within {window} bytes before the {size} MB target")
            cut = target - min(window, target) + boundary + 1
            reader.seek(0)
            remaining = cut
            with paths[size].open("wb") as writer:
                while remaining > 0:
                    chunk = reader.read(min(8 << 20, remaining))
                    if not chunk:
                        raise ValueError(f"unexpected end of source while writing the {size} MB slice")
                    writer.write(chunk)
                    remaining -= len(chunk)
    return paths


def slice_report(slice_paths: dict[int, Path]) -> list[dict[str, Any]]:
    """Byte sizes plus Attack composition of every slice for slice-bias disclosure."""
    report = []
    for size, path in sorted(slice_paths.items()):
        con = duckdb.connect()
        try:
            rows = int(con.execute(f"SELECT count(*) FROM read_csv('{str(path).replace(chr(39), chr(39)*2)}')").fetchone()[0])
            share = con.execute(
                f"SELECT \"Attack\", count(*) FROM read_csv('{str(path).replace(chr(39), chr(39)*2)}') GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall()
        finally:
            con.close()
        report.append({
            "size_mb_target": size,
            "bytes": path.stat().st_size,
            "rows": rows,
            "attack_share": {attack: int(count) for attack, count in share},
        })
    return report


def run_quality_checks(cleaned_parquet: Path, results_dir: Path, batch_rows: int = 500_000,
                       force: bool = False) -> dict[str, Any]:
    """Full Great Expectations suite over the cleaned dataset in bounded batches."""
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "quality_report.json"
    if not force and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["reused"] = True
        return report

    import great_expectations as gx
    import pandas as pd

    def quote(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    con = duckdb.connect()
    try:
        parquet_sql = str(cleaned_parquet.resolve()).replace("'", "''")
        columns = [row[0] for row in con.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{parquet_sql}'))"
        ).fetchall()]
        total_rows = int(con.execute(f"SELECT count(*) FROM read_parquet('{parquet_sql}')").fetchone()[0])
        select_list = ", ".join(quote(column) for column in columns)
        cursor = con.execute(f"SELECT {select_list} FROM read_parquet('{parquet_sql}')")

        context = gx.get_context(mode="ephemeral")
        datasource = context.data_sources.add_pandas("ids_cleaned")
        asset = datasource.add_dataframe_asset(name="cleaned_batches")
        batch_definition = asset.add_batch_definition_whole_dataframe("whole_batch")

        expectations: list[Any] = [
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column) for column in columns
        ] + [
            gx.expectations.ExpectColumnValuesToBeInSet(column=LABEL_COLUMN, value_set=[0, 1]),
            gx.expectations.ExpectColumnValuesToBeInSet(column="Attack", value_set=ATTACK_DOMAIN),
            gx.expectations.ExpectColumnValuesToBeInSet(column="Attack_subType", value_set=ATTACK_SUBTYPE_DOMAIN),
        ] + [
            gx.expectations.ExpectColumnValuesToBeBetween(column=column, min_value=0, max_value=None)
            for column in VARIANCE_COLUMNS + STD_COLUMNS
        ] + [
            gx.expectations.ExpectColumnValuesToBeBetween(column=column, min_value=-1.0, max_value=1.0)
            for column in PCC_COLUMNS
        ]

        runs = 0
        succeeded = 0
        unexpected: dict[str, int] = {}
        sample: Any = None
        checked = 0
        while True:
            chunk = cursor.fetchmany(batch_rows)
            if not chunk:
                break
            batch = pd.DataFrame(chunk, columns=columns)
            if sample is None:
                sample = batch.head(2).copy()
            batch_object = batch_definition.get_batch(batch_parameters={"dataframe": batch})
            for expectation in expectations:
                result = batch_object.validate(expectation)
                runs += 1
                success = bool(getattr(result, "success", False))
                succeeded += int(success)
                if not success:
                    payload = result.to_json_dict()
                    key = f"{type(expectation).__name__}:{expectation.column}"
                    unexpected[key] = unexpected.get(key, 0) + int(
                        payload.get("result", {}).get("unexpected_count", 0)
                    )
            checked += len(batch)
        if checked != total_rows:
            raise RuntimeError(f"Great Expectations checked {checked:,} of {total_rows:,} rows")

        # Negative control: the suite must actually fail on injected bad data.
        negative = sample.copy()
        negative[LABEL_COLUMN] = -999999
        negative_object = batch_definition.get_batch(batch_parameters={"dataframe": negative})
        negative_result = negative_object.validate(
            gx.expectations.ExpectColumnValuesToBeInSet(column=LABEL_COLUMN, value_set=[0, 1])
        )
        if bool(getattr(negative_result, "success", True)):
            raise RuntimeError("Great Expectations negative control did not detect an invalid label")

        report = {
            "implementation": f"Great Expectations {getattr(gx, '__version__', 'unknown')}",
            "rows_checked": checked,
            "expectations_per_batch": len(expectations),
            "expectations_run": runs,
            "expectations_succeeded": succeeded,
            "unexpected_counts": unexpected,
            "negative_control_failed_as_expected": True,
            "known_unexpected_note": (
                "pcc columns legitimately exceed [-1, 1] in the source (documented in Lab 1); "
                "rows are kept and reported, not dropped, because the aggregation task does not use pcc."
            ),
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["reused"] = False
        return report
    finally:
        con.close()


def ensure_prepared(root: Path, force: bool = False) -> dict[str, Any]:
    """Idempotent entry point used by the notebook: source check → cleaning → GE → slices."""
    workdir = root / "data" / "lab2"
    inputs = workdir / "inputs"
    results = root / "lab2" / "results"
    source = resolve_source(root)
    provenance = verify_source(source)
    stats = build_cleaned(source, inputs, force=force)
    cleaned_csv = inputs / "ids_cleaned.csv"
    cleaned_parquet = inputs / "ids_cleaned.parquet"
    quality = run_quality_checks(cleaned_parquet, results, force=force)
    full_target = 1000
    slices = create_prefix_slices(cleaned_csv, inputs, [100, full_target])
    slices_report = slice_report(slices)
    summary = {
        "source": provenance,
        "cleaning": stats,
        "quality": quality,
        "slices": slices_report,
        "cleaned_csv": str(cleaned_csv),
        "cleaned_parquet": str(cleaned_parquet),
        "slice_paths": {str(size): str(path) for size, path in slices.items()},
        "full_csv": str(cleaned_csv),
        "aggregation": {
            "key_columns": KEY_COLUMNS,
            "mean_columns": MEAN_COLUMNS,
            "max_column": MAX_COLUMN,
            "aggregates": ["count", "mean", "mean", "max"],
        },
    }
    (workdir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true", help="Rebuild cleaned data, quality report and slices")
    args = parser.parse_args()
    summary = ensure_prepared(args.root.resolve(), force=args.force)
    print(json.dumps({
        "raw_rows": summary["cleaning"]["raw_rows"],
        "cleaned_rows": summary["cleaning"]["cleaned_rows"],
        "duplicate_excess_rows": summary["cleaning"]["duplicate_excess_rows"],
        "expectations_run": summary["quality"]["expectations_run"],
        "slices": {row["size_mb_target"]: row["bytes"] for row in summary["slices"]},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
