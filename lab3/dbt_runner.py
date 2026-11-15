"""ЛР3: запуск dbt из пайплайна (субпроцесс, JSON-вывод).

dbt и пайплайн делят один duckdb-файл: вызов всегда ПОСЛЕ закрытия писателя
(DBT_LAB3_WAREHOUSE задаётся явно). Парсим `--json` вывод для метрик прогонов.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DBT_DIR = Path("lab3/dbt")


def dbt_executable() -> str:
    cand = Path(sys.executable).with_name("dbt.exe")
    return str(cand if cand.exists() else "dbt")


def run_dbt(*args: str, warehouse: Path, target: str = "dev",
            vars: dict[str, Any] | None = None, timeout: int = 900) -> dict[str, Any]:
    env = dict(os.environ)
    env["DBT_LAB3_WAREHOUSE"] = str(Path(warehouse).resolve())
    env["DBT_PROFILES_DIR"] = str((DBT_DIR / "profiles").resolve())
    cmd = [dbt_executable(), *args, "--target", target,
           "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR / "profiles")]
    if vars:
        cmd += ["--vars", json.dumps(vars)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          env=env, timeout=timeout)
    out = proc.stdout or ""
    payload: dict[str, Any] | None = None
    results_file = DBT_DIR / "target" / "run_results.json"
    if results_file.exists():
        try:
            payload = json.loads(results_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
    return {
        "returncode": proc.returncode,
        "result": payload,
        "stdout_tail": out[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }
