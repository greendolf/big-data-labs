"""CI smoke: весь пайплайн на синтетике без реальных данных (запуск в GitHub Actions).

python -m lab3.smoke  → 0 если все проверки прошли, 1 иначе.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from lab3 import warehouse as wh
from lab3.batching import build_batches, make_dirty
from lab3.dbt_runner import run_dbt
from lab3.fixture_data import synthetic_source
from lab3.pipeline import process_batch


def main() -> int:
    tmp = Path("data/lab3/_smoke_ci")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    src = tmp / "src.parquet"
    synthetic_source(6_000).to_parquet(src, index=False)

    manifest = build_batches(src, tmp / "batches", n_batches=3, late_share=0.02)
    b1 = tmp / "batches" / "b001.parquet"
    b1.rename(tmp / "b001_clean.parquet")
    make_dirty(tmp / "b001_clean.parquet", b1, n_null_key=4, n_bad_label=2, n_negative=2)

    wh_path = tmp / "warehouse.duckdb"
    wh.init_warehouse(wh_path)
    totals = {"inserted": 0, "dlq": 0}
    for b in manifest:
        m = process_batch(tmp / "batches" / f"{b['batch_id']}.parquet", b["batch_id"],
                          warehouse=wh_path)
        totals["inserted"] += m["inserted"]
        totals["dlq"] += m["dlq"]
    replay = process_batch(tmp / "batches" / "b002.parquet", "b002", warehouse=wh_path)

    checks = {
        "rows_loaded": totals["inserted"],
        "rows_expected": 6_000 - 8,  # 8 грязных строк фикстуры уходят в DLQ, не в raw
        "dlq_rows": totals["dlq"],
        "replay_inserted": replay["inserted"],
        "replay_updated": replay["updated"],
    }
    dbt_res = run_dbt("build", warehouse=wh_path, timeout=600)
    checks["dbt_rc"] = dbt_res["returncode"]
    con = wh.connect(wh_path)
    checks["marts_devices"] = int(con.execute(
        "SELECT count(*) FROM dbt_marts.dim_device_profile").fetchone()[0])
    con.close()
    print("SMOKE CHECKS:", json.dumps(checks, indent=2))

    ok = (checks["rows_loaded"] == checks["rows_expected"] and checks["dlq_rows"] == 8
          and checks["replay_inserted"] == 0 and checks["replay_updated"] > 0
          and checks["dbt_rc"] == 0 and checks["marts_devices"] > 0)
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
