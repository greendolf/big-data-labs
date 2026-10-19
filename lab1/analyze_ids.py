"""Full-source profile and CSV/Parquet/Avro roundtrip audit for Lab 1 IDS data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import duckdb

DATASET_ID = "azalhowaide/iot-dataset-for-intrusion-detection-systems-ids"
SOURCE_FILENAME = "BoTNeTIoT-L01-v2.csv"
NUMERIC_COLUMNS = (
    "MI_dir_L0.1_weight",
    "MI_dir_L0.1_mean",
    "MI_dir_L0.1_variance",
    "H_L0.1_weight",
    "H_L0.1_mean",
    "H_L0.1_variance",
    "HH_L0.1_weight",
    "HH_L0.1_mean",
    "HH_L0.1_std",
    "HH_L0.1_magnitude",
    "HH_L0.1_radius",
    "HH_L0.1_covariance",
    "HH_L0.1_pcc",
    "HH_jit_L0.1_weight",
    "HH_jit_L0.1_mean",
    "HH_jit_L0.1_variance",
    "HpHp_L0.1_weight",
    "HpHp_L0.1_mean",
    "HpHp_L0.1_std",
    "HpHp_L0.1_magnitude",
    "HpHp_L0.1_radius",
    "HpHp_L0.1_covariance",
    "HpHp_L0.1_pcc",
)
CATEGORY_COLUMNS = ("Device_Name", "Attack", "Attack_subType")
LABEL_COLUMN = "label"
EXPECTED_COLUMNS = (*NUMERIC_COLUMNS, *CATEGORY_COLUMNS, LABEL_COLUMN)
OPTIONS: dict[str, Any] = {
    "threads": 2,
    "memory_limit": "2GB",
    "chunk_rows": 50_000,
    "numeric_columns": list(NUMERIC_COLUMNS),
    "category_columns": list(CATEGORY_COLUMNS),
    "label_column": LABEL_COLUMN,
    "formats": ["csv", "parquet", "avro"],
    "parquet_compression": "zstd",
    "avro_codec": "deflate",
}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _canonical_row(row: Iterable[Any]) -> bytes:
    values: list[Any] = []
    for index, value in enumerate(row):
        if value is None:
            values.append(None)
        elif index == len(EXPECTED_COLUMNS) - 1:
            numeric_label = float(value)
            if not numeric_label.is_integer():
                raise ValueError(f"Non-integral label encountered during roundtrip: {value!r}")
            values.append(int(numeric_label))
        elif index < len(NUMERIC_COLUMNS):
            number = float(value)
            values.append(number.hex() if math.isfinite(number) else str(number))
        else:
            values.append(str(value))
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return len(payload).to_bytes(8, "big") + payload


def _csv_row_values(row: dict[str, str | None]) -> tuple[Any, ...]:
    values: list[Any] = []
    for column in NUMERIC_COLUMNS:
        raw = row.get(column)
        values.append(None if raw is None or raw == "" else float(raw))
    for column in CATEGORY_COLUMNS:
        raw = row.get(column)
        values.append(None if raw is None or raw == "" else raw)
    raw_label = row.get(LABEL_COLUMN)
    if raw_label in (None, ""):
        values.append(None)
    else:
        numeric_label = float(raw_label)
        values.append(int(numeric_label) if numeric_label.is_integer() else numeric_label)
    return tuple(values)


def _row_digest(rows: Iterable[Iterable[Any]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(_canonical_row(row))
        count += 1
    return digest.hexdigest(), count


def _source_digest(source: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), source.stat().st_size


def _verify_download_provenance(source: Path, source_sha: str, source_bytes: int) -> dict[str, Any] | None:
    sidecar = source.with_suffix(source.suffix + ".source.json")
    if not sidecar.exists():
        return None
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    if provenance.get("dataset_id") != DATASET_ID:
        raise ValueError(f"Source sidecar belongs to a different dataset: {provenance.get('dataset_id')!r}")
    if provenance.get("source_sha256") != source_sha or provenance.get("source_bytes") != source_bytes:
        raise ValueError("Source CSV does not match the SHA-256 and byte count in its downloader sidecar")
    return provenance


def _read_header(source: Path) -> list[str]:
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        try:
            header = next(csv.reader(stream))
        except StopIteration as error:
            raise ValueError("Source CSV is empty") from error
    if len(header) != len(set(header)):
        raise ValueError("Source CSV contains duplicate column names")
    if tuple(header) != EXPECTED_COLUMNS:
        missing = [name for name in EXPECTED_COLUMNS if name not in header]
        extra = [name for name in header if name not in EXPECTED_COLUMNS]
        if missing or extra:
            raise ValueError(f"Unexpected IDS schema; missing={missing}, extra={extra}")
        raise ValueError("IDS source columns are in an unexpected order")
    return header


def _write_csv(path: Path, columns: list[str], rows: Iterable[Iterable[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def _profile_numeric(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    expressions = []
    for index, column in enumerate(NUMERIC_COLUMNS):
        name = _quote(column)
        expressions.extend([
            f"count(*) FILTER (WHERE {name} IS NULL) AS {_quote(f'missing_{index}')}",
            f"count(*) FILTER (WHERE isnan({name})) AS {_quote(f'nan_{index}')}",
            f"count(*) FILTER (WHERE isinf({name}) AND {name} > 0) AS {_quote(f'pos_inf_{index}')}",
            f"count(*) FILTER (WHERE isinf({name}) AND {name} < 0) AS {_quote(f'neg_inf_{index}')}",
            f"min({name}) FILTER (WHERE isfinite({name})) AS {_quote(f'min_{index}')}",
            f"max({name}) FILTER (WHERE isfinite({name})) AS {_quote(f'max_{index}')}",
            f"avg({name}) FILTER (WHERE isfinite({name})) AS {_quote(f'mean_{index}')}",
            f"stddev_samp({name}) FILTER (WHERE isfinite({name})) AS {_quote(f'std_{index}')}",
        ])
    expressions.insert(0, "count(*) AS total_rows")
    values = con.execute("SELECT " + ",".join(expressions) + " FROM source_data").fetchone()
    total = int(values[0])
    rows = []
    for index, column in enumerate(NUMERIC_COLUMNS):
        offset = 1 + index * 8
        rows.append({
            "column": column,
            "total": total,
            "missing": int(values[offset]),
            "nan": int(values[offset + 1]),
            "positive_infinity": int(values[offset + 2]),
            "negative_infinity": int(values[offset + 3]),
            "min": values[offset + 4],
            "max": values[offset + 5],
            "mean": values[offset + 6],
            "std": values[offset + 7],
        })
    return rows


def _all_columns_nulls(con: duckdb.DuckDBPyConnection, columns: list[str]) -> list[dict[str, Any]]:
    expressions = ", ".join(
        f"count(*) FILTER (WHERE {_quote(column)} IS NULL) AS {_quote(f'missing_{index}')}"
        for index, column in enumerate(columns)
    )
    result = con.execute(f"SELECT {expressions} FROM source_data").fetchone()
    return [{"column": column, "missing": int(result[index])} for index, column in enumerate(columns)]


def _category_counts(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in (LABEL_COLUMN, *CATEGORY_COLUMNS):
        quoted = _quote(column)
        values = con.execute(
            f"SELECT CAST({quoted} AS VARCHAR), count(*) FROM source_data "
            f"GROUP BY {quoted} ORDER BY count(*) DESC, 1 LIMIT 5000"
        ).fetchall()
        rows.extend(
            {"column": column, "value": value, "rows": int(count)}
            for value, count in values
        )
    return rows


def _duplicate_audit(con: duckdb.DuckDBPyConnection) -> tuple[int, list[dict[str, Any]]]:
    keys = ", ".join(_quote(column) for column in EXPECTED_COLUMNS)
    duplicate_count = con.execute(
        f"SELECT coalesce(sum(n - 1), 0) FROM "
        f"(SELECT count(*) AS n FROM source_data GROUP BY {keys} HAVING count(*) > 1)"
    ).fetchone()[0]
    return int(duplicate_count), []


def _label_audit(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    label = _quote(LABEL_COLUMN)
    attack = _quote("Attack")
    invalid = con.execute(
        f"""SELECT count(*) FROM source_data
            WHERE {label} IS NULL OR {label} NOT IN (0, 1)
               OR ({attack} IS NULL)
               OR ((lower(trim({attack})) = 'normal') != ({label} = 1))"""
    ).fetchone()[0]
    samples = con.execute(
        f"""SELECT __source_row, {label}, {attack}, {_quote('Attack_subType')}, {_quote('Device_Name')}
            FROM source_data
            WHERE {label} IS NULL OR {label} NOT IN (0, 1)
               OR ({attack} IS NULL)
               OR ((lower(trim({attack})) = 'normal') != ({label} = 1))
            LIMIT 20"""
    ).fetchall()
    return {
        "invalid_rows": int(invalid),
        "sample_rows": [list(row) for row in samples],
        "rule": "label=1 iff Attack is Normal; label=0 for non-Normal attacks",
    }


def _numeric_domain_audit(con: duckdb.DuckDBPyConnection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rules = {
        "MI_dir_L0.1_variance": ("nonnegative_variance", 0.0),
        "H_L0.1_variance": ("nonnegative_variance", 0.0),
        "HH_jit_L0.1_variance": ("nonnegative_variance", 0.0),
        "HH_L0.1_std": ("nonnegative_std", 0.0),
        "HpHp_L0.1_std": ("nonnegative_std", 0.0),
        "HH_L0.1_pcc": ("correlation_between_minus1_and_1", 1e-12),
        "HpHp_L0.1_pcc": ("correlation_between_minus1_and_1", 1e-12),
    }
    metrics: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for column, (rule, tolerance) in rules.items():
        value = _quote(column)
        if rule.startswith("correlation"):
            predicate = f"{value} < {-1.0 - tolerance} OR {value} > {1.0 + tolerance}"
        else:
            predicate = f"{value} < {-tolerance}"
        count = int(
            con.execute(
                f"SELECT count(*) FROM source_data WHERE {value} IS NOT NULL AND ({predicate})"
            ).fetchone()[0]
        )
        metrics.append({"column": column, "rule": rule, "tolerance": tolerance, "violations": count})
        if count:
            rows = con.execute(
                f"SELECT __source_row, {value} FROM source_data WHERE {predicate} LIMIT 10"
            ).fetchall()
            examples.extend(
                {"source_row": int(row_id), "column": column, "value": number, "rule": rule}
                for row_id, number in rows
            )
    return metrics, examples


def _run_great_expectations(con: duckdb.DuckDBPyConnection, rows: int) -> dict[str, Any]:
    """Validate the entire critical label column in bounded batches with GX."""
    try:
        import great_expectations as gx
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("Great Expectations and pandas are required for the full audit") from error

    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("ids_profile")
    asset = datasource.add_dataframe_asset(name="label_batches")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_batch")
    expectation_runs = 0
    successful_runs = 0
    unexpected = 0
    batch_sample: Any = None
    batch_rows = 100_000
    cursor = con.execute(f"SELECT {_quote(LABEL_COLUMN)} FROM source_data ORDER BY __source_row")
    checked_rows = 0
    while True:
        chunk = cursor.fetchmany(batch_rows)
        if not chunk:
            break
        batch = pd.DataFrame(chunk, columns=[LABEL_COLUMN])
        if batch_sample is None:
            batch_sample = batch.head(1).copy()
        batch_object = batch_definition.get_batch(batch_parameters={"dataframe": batch})
        for expectation in (
            gx.expectations.ExpectColumnValuesToNotBeNull(column=LABEL_COLUMN),
            gx.expectations.ExpectColumnValuesToBeInSet(column=LABEL_COLUMN, value_set=[0, 1]),
        ):
            result = batch_object.validate(expectation)
            expectation_runs += 1
            success = bool(getattr(result, "success", False))
            successful_runs += int(success)
            if not success:
                result_dict = result.to_json_dict()
                unexpected += int(result_dict.get("result", {}).get("unexpected_count", 0))
        checked_rows += len(batch)
    if checked_rows != rows:
        raise RuntimeError(f"Great Expectations checked {checked_rows:,} of {rows:,} label values")
    negative_control = {"tested": True, "expected_to_fail": True, "failed_as_expected": False, "unexpected_values": 0}
    if batch_sample is not None:
        negative = batch_sample.copy()
        negative[LABEL_COLUMN] = -999999
        negative_batch = batch_definition.get_batch(batch_parameters={"dataframe": negative})
        negative_result = negative_batch.validate(
            gx.expectations.ExpectColumnValuesToBeInSet(column=LABEL_COLUMN, value_set=[0, 1])
        )
        negative_payload = negative_result.to_json_dict()
        negative_control["failed_as_expected"] = not bool(negative_result.success)
        negative_control["unexpected_values"] = int(negative_payload.get("result", {}).get("unexpected_count", 0))
        if not negative_control["failed_as_expected"] or not negative_control["unexpected_values"]:
            raise RuntimeError("Great Expectations negative control did not detect an invalid label")
    return {
        "implementation": f"Great Expectations {getattr(gx, '__version__', 'unknown')}",
        "column": LABEL_COLUMN,
        "rows_checked": checked_rows,
        "expectations_run": expectation_runs,
        "expectations_succeeded": successful_runs,
        "expectations_passed": expectation_runs == successful_runs,
        "unexpected_values": unexpected,
        "rules": ["not null", "in set {0, 1}"],
        "negative_control": negative_control,
    }


def _avro_names() -> dict[str, str]:
    """Map arbitrary CSV headers to Avro-spec names and keep the mapping reversible."""
    names = {column: re.sub(r"[^A-Za-z0-9_]", "_", column) for column in EXPECTED_COLUMNS}
    if len(set(names.values())) != len(names):
        raise ValueError("IDS headers collide after mapping to Avro record field names")
    return names


def _avro_schema() -> dict[str, Any]:
    safe_names = _avro_names()
    fields = []
    for column in EXPECTED_COLUMNS:
        if column in NUMERIC_COLUMNS:
            avro_type: Any = ["null", "double"]
        elif column == LABEL_COLUMN:
            avro_type = ["null", "long"]
        else:
            avro_type = ["null", "string"]
        fields.append({"name": safe_names[column], "type": avro_type, "default": None})
    return {"type": "record", "name": "BotnetIotRecord", "fields": fields}


def _csv_dict_records(source: Path) -> Iterable[dict[str, str | None]]:
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _to_typed_record(row: dict[str, str | None]) -> dict[str, Any]:
    typed = {}
    for column in NUMERIC_COLUMNS:
        value = row[column]
        typed[column] = None if value in (None, "") else float(value)
    for column in CATEGORY_COLUMNS:
        value = row[column]
        typed[column] = None if value in (None, "") else value
    raw_label = row[LABEL_COLUMN]
    if raw_label in (None, ""):
        typed[LABEL_COLUMN] = None
    else:
        number = float(raw_label)
        if not number.is_integer():
            raise ValueError(f"Non-integral label value: {raw_label!r}")
        typed[LABEL_COLUMN] = int(number)
    return typed


def _write_avro(source: Path, target: Path) -> dict[str, Any]:
    try:
        from fastavro import writer
    except ImportError as error:
        raise RuntimeError("fastavro is required to write the Avro comparison") from error

    source_digest = hashlib.sha256()
    source_rows = 0
    safe_names = _avro_names()

    def records() -> Iterable[dict[str, Any]]:
        nonlocal source_rows
        for raw in _csv_dict_records(source):
            values = _to_typed_record(raw)
            source_digest.update(_canonical_row(values[column] for column in EXPECTED_COLUMNS))
            source_rows += 1
            yield {safe_names[column]: value for column, value in values.items()}

    with target.open("wb") as stream:
        writer(stream, _avro_schema(), records(), codec="deflate")
    return {"rows": source_rows, "source_value_sha256": source_digest.hexdigest()}


def _read_avro_digest(path: Path) -> tuple[str, int]:
    from fastavro import reader

    digest = hashlib.sha256()
    count = 0
    safe_names = _avro_names()
    with path.open("rb") as stream:
        for record in reader(stream):
            digest.update(_canonical_row(record[safe_names[column]] for column in EXPECTED_COLUMNS))
            count += 1
    return digest.hexdigest(), count


def _artifact_paths(results: Path, runtime: Path) -> dict[str, str]:
    return {
        "profile": "profile.json",
        "numeric_stats": "numeric_stats.csv",
        "category_counts": "category_counts.csv",
        "quality_metrics": "quality_metrics.csv",
        "quality_examples": "quality_examples.csv",
        "duplicate_examples": "duplicate_examples.csv",
        "format_profile": "format_profile.csv",
        "serialization_comparison": "serialization_comparison.csv",
        "parquet": str((runtime / "ids.parquet").resolve()),
        "avro": str((runtime / "ids.avro").resolve()),
    }


def _cache_is_valid(
    profile_path: Path,
    source_sha: str,
    module_sha: str,
    options: dict[str, Any],
    runtime: Path,
) -> bool:
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if profile.get("status") not in {"complete", "complete_with_quality_findings"}:
        return False
    provenance = profile.get("provenance", {})
    if (provenance.get("source_sha256"), provenance.get("module_sha256"), provenance.get("options")) != (
        source_sha,
        module_sha,
        options,
    ):
        return False
    artifacts = profile.get("artifacts", {})
    for key in ("numeric_stats", "category_counts", "quality_metrics", "quality_examples", "duplicate_examples", "format_profile", "serialization_comparison"):
        if not (profile_path.parent / artifacts.get(key, "")).is_file():
            return False
    return (runtime / "ids.parquet").is_file() and (runtime / "ids.avro").is_file()


def analyze_source(source: Path, runtime: Path, results: Path, force: bool = False) -> dict[str, Any]:
    """Run the complete source profile and full-value CSV/Parquet/Avro audit.

    The source remains untouched. Large format copies and DuckDB spill files go
    under ``runtime``; only compact CSV/JSON summaries go under ``results``.
    This is a full-source operation and may take several minutes.
    """
    source = Path(source).resolve()
    runtime = Path(runtime).resolve()
    results = Path(results).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    runtime.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    profile_path = results / "profile.json"
    analysis_started = time.perf_counter()

    def progress(message: str) -> None:
        elapsed = time.perf_counter() - analysis_started
        print(f"[Lab 1 IDS] +{elapsed:,.1f}s {message}", flush=True)

    source_sha, source_bytes = _source_digest(source)
    download_provenance = _verify_download_provenance(source, source_sha, source_bytes)
    progress(f"source SHA-256 verified ({source_bytes:,} bytes)")
    module_sha = _sha256_file(Path(__file__).resolve())
    if not force and _cache_is_valid(profile_path, source_sha, module_sha, OPTIONS, runtime):
        progress("reusing complete cached profile")
        return json.loads(profile_path.read_text(encoding="utf-8"))

    columns = _read_header(source)
    database_path = runtime / "profile.duckdb"
    spill_path = runtime / "duckdb_tmp"
    spill_path.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database_path))
    con.execute("SET threads=2")
    con.execute("SET memory_limit='2GB'")
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"SET temp_directory={_sql_string(spill_path)}")
    typed_columns = {column: "DOUBLE" for column in NUMERIC_COLUMNS}
    typed_columns.update({column: "VARCHAR" for column in CATEGORY_COLUMNS})
    typed_columns[LABEL_COLUMN] = "BIGINT"
    con.execute(
        "CREATE OR REPLACE TABLE source_data AS "
        "SELECT row_number() OVER () AS __source_row, * FROM read_csv(?, "
        "header=true, auto_detect=false, columns=?, nullstr=[''])",
        [str(source), typed_columns],
    )
    progress("materialized typed CSV once into disk-backed DuckDB table (2 threads, 2 GB limit)")
    try:
        row_count = int(con.execute("SELECT count(*) FROM source_data").fetchone()[0])
        if row_count <= 0:
            raise ValueError("Source CSV contains no data rows")
        numeric_stats = _profile_numeric(con)
        null_stats = _all_columns_nulls(con, columns)
        categories = _category_counts(con)
        label_audit = _label_audit(con)
        gx_result = _run_great_expectations(con, row_count)
        domain_metrics, domain_examples = _numeric_domain_audit(con)
        progress("full-source numeric/category/quality profile and Great Expectations completed")

        # Exact full-row grouping includes all 27 source values; the internal
        # __source_row key is excluded from the duplicate comparison.
        keys = ", ".join(_quote(column) for column in columns)
        duplicate_count = int(
            con.execute(
                f"SELECT coalesce(sum(n - 1), 0) FROM "
                f"(SELECT count(*) AS n FROM source_data GROUP BY {keys} HAVING count(*) > 1)"
            ).fetchone()[0]
        )
        duplicate_samples = con.execute(
            f"SELECT {keys}, count(*) AS occurrences, min(__source_row) AS first_source_row, "
            f"max(__source_row) AS another_source_row FROM source_data "
            f"GROUP BY {keys} HAVING count(*) > 1 ORDER BY occurrences DESC LIMIT 10"
        ).fetchall()
        duplicate_examples = [
            {
                "occurrences": int(row[-3]),
                "values": dict(zip(columns, row[: len(columns)])),
                "source_rows": [int(row[-2]), int(row[-1])],
            }
            for row in duplicate_samples
        ]
        progress(f"exact full-row duplicate audit completed ({duplicate_count:,} excess rows)")

        format_profile = [
            {"field": "timestamp_columns", "value": 0, "note": "The original header contains no timestamp column."},
            {"field": "numeric_columns", "value": len(NUMERIC_COLUMNS), "note": "23 engineered numeric features; HH_jit_L0.1_mean is a jitter statistic, not a timestamp."},
            {"field": "categorical_columns", "value": len(CATEGORY_COLUMNS) + 1, "note": "Device_Name, Attack, Attack_subType, and label."},
        ]
        text_bytes_expression = " + ".join(
            f"coalesce(sum(strlen(coalesce({_quote(column)}, ''))), 0)" for column in CATEGORY_COLUMNS
        )
        text_bytes = int(con.execute(f"SELECT {text_bytes_expression} FROM source_data").fetchone()[0])
        nested_cells = int(
            con.execute(
                "SELECT " + " + ".join(
                    f"count(*) FILTER (WHERE {_quote(column)} LIKE '%{{%' OR {_quote(column)} LIKE '%[%')"
                    for column in CATEGORY_COLUMNS
                ) + " FROM source_data"
            ).fetchone()[0]
        )
        whitespace_values = int(
            con.execute(
                "SELECT " + " + ".join(
                    f"count(*) FILTER (WHERE {_quote(column)} IS NOT NULL AND {_quote(column)} != trim({_quote(column)}))"
                    for column in CATEGORY_COLUMNS
                ) + " FROM source_data"
            ).fetchone()[0]
        )
        format_profile.extend([
            {"field": "categorical_text_bytes", "value": text_bytes, "note": "UTF-8 payload bytes across the three categorical text fields; CSV delimiters/quotes excluded."},
            {"field": "categorical_text_share_of_csv_bytes", "value": text_bytes / source_bytes if source_bytes else 0.0, "note": "Lower-bound text payload share; excludes CSV framing and numeric text."},
            {"field": "json_like_categorical_cells", "value": nested_cells, "note": "Categorical fields containing brace or bracket delimiters; a screening count, not JSON parsing."},
            {"field": "categorical_values_with_outer_whitespace", "value": whitespace_values, "note": "Text values differing from trim(value); useful format consistency check."},
        ])
        invalid_numeric = []
        for item in numeric_stats:
            if item["missing"] or item["nan"] or item["positive_infinity"] or item["negative_infinity"]:
                invalid_numeric.append(item["column"])

        # Full typed Parquet conversion and full bidirectional multiset equality.
        parquet_path = runtime / "ids.parquet"
        avro_path = runtime / "ids.avro"
        parquet_path.unlink(missing_ok=True)
        avro_path.unlink(missing_ok=True)
        quoted_columns = ", ".join(_quote(column) for column in columns)
        started = time.perf_counter()
        parquet_literal = "'" + str(parquet_path).replace("'", "''") + "'"
        con.execute(
            "COPY (SELECT " + quoted_columns + " FROM source_data) TO " + parquet_literal
            + " (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        parquet_write_seconds = time.perf_counter() - started
        progress(f"full Parquet write completed in {parquet_write_seconds:,.1f}s")
        started = time.perf_counter()
        parquet_rows = int(con.execute("SELECT count(*) FROM read_parquet(?)", [str(parquet_path)]).fetchone()[0])
        parquet_read_seconds = time.perf_counter() - started
        parquet_mismatch = int(
            con.execute(
                "SELECT count(*) FROM ((SELECT " + quoted_columns + " FROM source_data EXCEPT ALL "
                "SELECT " + quoted_columns + " FROM read_parquet(" + parquet_literal + ")) UNION ALL "
                "(SELECT " + quoted_columns + " FROM read_parquet(" + parquet_literal + ") EXCEPT ALL "
                "SELECT " + quoted_columns + " FROM source_data)) AS differences"
            ).fetchone()[0]
        )

        # Avro writer streams one CSV record at a time; no full dataset DataFrame
        # or record list is materialized.
        started = time.perf_counter()
        avro_written = _write_avro(source, avro_path)
        avro_write_seconds = time.perf_counter() - started
        started = time.perf_counter()
        avro_digest, avro_rows = _read_avro_digest(avro_path)
        avro_read_seconds = time.perf_counter() - started
        avro_values_equal = (
            avro_written["rows"] == avro_rows == row_count
            and avro_written["source_value_sha256"] == avro_digest
        )
        if parquet_rows != row_count or parquet_mismatch != 0 or not avro_values_equal:
            raise RuntimeError(
                "Full-value serialization check failed: "
                f"rows={row_count}, parquet_rows={parquet_rows}, parquet_mismatch={parquet_mismatch}, "
                f"avro_rows={avro_rows}, avro_equal={avro_values_equal}"
            )
        progress(f"Parquet and Avro full-value roundtrips verified; Avro write {avro_write_seconds:,.1f}s, read {avro_read_seconds:,.1f}s")

        comparisons = [
            {"format": "CSV", "bytes": source_bytes, "write_seconds": None, "read_seconds": None, "rows": row_count, "all_values_equal": None, "method": "original source file; no self-roundtrip claimed"},
            {"format": "Parquet", "bytes": parquet_path.stat().st_size, "write_seconds": parquet_write_seconds, "read_seconds": parquet_read_seconds, "rows": parquet_rows, "all_values_equal": parquet_mismatch == 0, "method": "DuckDB full typed conversion; bidirectional EXCEPT ALL"},
            {"format": "Avro", "bytes": avro_path.stat().st_size, "write_seconds": avro_write_seconds, "read_seconds": avro_read_seconds, "rows": avro_rows, "all_values_equal": avro_values_equal, "method": "streaming typed records; full-row canonical digest"},
        ]
        _write_csv(results / "numeric_stats.csv", list(numeric_stats[0]), (row.values() for row in numeric_stats))
        _write_csv(results / "category_counts.csv", ["column", "value", "rows"], ((row["column"], row["value"], row["rows"]) for row in categories))
        quality_rows = [
            {"metric": f"missing:{row['column']}", "value": row["missing"]}
            for row in null_stats
        ]
        quality_rows.extend(
            {"metric": f"domain_violations:{row['column']}:{row['rule']}", "value": row["violations"]}
            for row in domain_metrics
        )
        quality_rows.extend([
            {"metric": "label_category_mismatch_rows", "value": label_audit["invalid_rows"]},
            {"metric": "full_row_duplicate_excess_rows", "value": duplicate_count},
            {"metric": "numeric_domain_examples", "value": json.dumps(domain_examples, ensure_ascii=False)},
        ])
        _write_csv(results / "quality_metrics.csv", ["metric", "value"], ((row["metric"], row["value"]) for row in quality_rows))
        quality_examples: list[dict[str, Any]] = []
        for item in null_stats:
            if item["missing"]:
                column = item["column"]
                for source_row, value in con.execute(
                    f"SELECT __source_row, {_quote(column)} FROM source_data "
                    f"WHERE {_quote(column)} IS NULL ORDER BY __source_row LIMIT 5"
                ).fetchall():
                    quality_examples.append({"source_row": int(source_row), "rule": "missing_value", "column": column, "value": value, "details": "null/empty source cell"})
        for item in numeric_stats:
            if item["nan"] or item["positive_infinity"] or item["negative_infinity"]:
                column = item["column"]
                for source_row, value in con.execute(
                    f"SELECT __source_row, {_quote(column)} FROM source_data "
                    f"WHERE isnan({_quote(column)}) OR isinf({_quote(column)}) "
                    "ORDER BY __source_row LIMIT 5"
                ).fetchall():
                    quality_examples.append({"source_row": int(source_row), "rule": "nonfinite_numeric", "column": column, "value": str(value), "details": "NaN or infinity in a numeric feature"})
        quality_examples.extend(domain_examples)
        for sample in label_audit["sample_rows"]:
            quality_examples.append({
                "source_row": sample[0],
                "rule": "attack_label_mismatch",
                "column": LABEL_COLUMN,
                "value": sample[1],
                "details": json.dumps({"Attack": sample[2], "Attack_subType": sample[3], "Device_Name": sample[4]}, ensure_ascii=False),
            })
        _write_csv(
            results / "quality_examples.csv",
            ["source_row", "rule", "column", "value", "details"],
            ((row.get("source_row"), row["rule"], row["column"], row.get("value"), row.get("details", "")) for row in quality_examples),
        )
        _write_csv(results / "duplicate_examples.csv", ["example", "occurrences", "values_json", "source_rows"], ((i + 1, row["occurrences"], json.dumps(row["values"], ensure_ascii=False, default=str), json.dumps(row["source_rows"])) for i, row in enumerate(duplicate_examples)))
        _write_csv(results / "format_profile.csv", ["field", "value", "note"], ((row["field"], row["value"], row["note"]) for row in format_profile))
        _write_csv(results / "serialization_comparison.csv", list(comparisons[0]), (row.values() for row in comparisons))

        finished_sha, finished_bytes = _source_digest(source)
        if finished_sha != source_sha or finished_bytes != source_bytes:
            raise RuntimeError("Source CSV changed while the analysis was running")
        profile = {
            "status": "complete",
            "source": {
                "path": source.name,
                "dataset_id": DATASET_ID,
                "source_file": SOURCE_FILENAME,
                "sha256": source_sha,
                "bytes": source_bytes,
                "rows": row_count,
                "columns": len(columns),
                "column_names": columns,
                "download_provenance": download_provenance,
            },
            "schema": {
                "numeric_columns": list(NUMERIC_COLUMNS),
                "categorical_columns": list(CATEGORY_COLUMNS),
                "label_column": LABEL_COLUMN,
                "timestamp_columns": [],
                "dtypes": {**{column: "float64" for column in NUMERIC_COLUMNS}, "Device_Name": "string", "Attack": "string", "Attack_subType": "string", "label": "int64"},
            },
            "numeric_stats": numeric_stats,
            "quality": {
                "nulls_by_column": null_stats,
                "numeric_nonfinite_columns": invalid_numeric,
                "numeric_domain_metrics": domain_metrics,
                "numeric_domain_examples": domain_examples,
                "duplicates": {
                    "full_row_count_excluding_first_occurrence": duplicate_count,
                    "examples": duplicate_examples,
                    "warning": "No event ID is provided. Repeated feature rows can represent legitimate repeated traffic; deduplication would drop observations and was not applied.",
                },
                "label_category_consistency": label_audit,
                "great_expectations": gx_result,
                "critical_label_gx": gx_result,
            },
            "categories": {"counts": categories},
            "variety": {
                "structured_columns": len(NUMERIC_COLUMNS) + len(CATEGORY_COLUMNS) + 1,
                "numeric_columns": len(NUMERIC_COLUMNS),
                "categorical_text_columns": len(CATEGORY_COLUMNS),
                "timestamp_columns": 0,
                "nested_structures": 0,
                "note": "The file is flat tabular CSV; categorical values are text labels. No JSON or timestamp fields are present.",
            },
            "formats": {row["format"].lower(): row for row in comparisons},
            "artifacts": _artifact_paths(results, runtime),
            "provenance": {"source_sha256": source_sha, "module_sha256": module_sha, "options": OPTIONS},
        }
        # A report is complete only when every required full-source check passed.
        has_missing = any(row["missing"] for row in null_stats)
        has_domain_findings = any(row["violations"] for row in domain_metrics)
        if not gx_result["expectations_passed"] or label_audit["invalid_rows"] or invalid_numeric or has_missing or has_domain_findings or duplicate_count:
            profile["status"] = "complete_with_quality_findings"
        temp_profile = profile_path.with_suffix(".json.partial")
        temp_profile.write_text(json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp_profile, profile_path)
        progress(f"profile and compact result tables written; status={profile['status']}")
        return profile
    finally:
        con.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/ids") / SOURCE_FILENAME)
    parser.add_argument("--runtime", type=Path, default=Path("data/lab1"))
    parser.add_argument("--results", type=Path, default=Path("lab1/results"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    profile = analyze_source(args.source, args.runtime, args.results, args.force)
    print(json.dumps({"status": profile["status"], "source": profile["source"], "artifacts": profile["artifacts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
