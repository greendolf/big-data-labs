"""Execute the Lab 2 report with this Python interpreter and local runtime files.

All runtime state (slice files, Spark temporary directories, Jupyter runtime
files) stays under the ignored data/lab2 directory. The CPU budget is pinned
to two threads so every engine competes under the same limits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Override the source Kaggle CSV path")
    parser.add_argument("--timeout", type=int, default=6 * 60 * 60, help="Timeout per code cell, seconds")
    parser.add_argument("--force", action="store_true", help="Rebuild cleaned data, quality report, slices")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    runtime = root / "data" / "lab2"
    runtime.mkdir(parents=True, exist_ok=True)
    temp_dir = runtime / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for variable in ("TEMP", "TMP", "TMPDIR"):
        os.environ[variable] = str(temp_dir)
    spark_tmp_dir = runtime / "spark-tmp"
    spark_tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SPARK_LOCAL_DIRS"] = str(spark_tmp_dir)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        os.environ[variable] = "2"
    os.environ["PYSPARK_PYTHON"] = str(Path(sys.executable).resolve())
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    # local[2] alone does not limit JVM common pools; keep local Spark within the same CPU budget.
    java_tool_options = os.environ.get("JAVA_TOOL_OPTIONS", "").strip()
    active_processor_limit = "-XX:ActiveProcessorCount=2"
    if active_processor_limit not in java_tool_options.split():
        os.environ["JAVA_TOOL_OPTIONS"] = " ".join(filter(None, [java_tool_options, active_processor_limit]))

    # Resolve JAVA_HOME: explicit environment first, then the ignored local
    # data/tools/lab2_java_home.txt fallback, then the Windows registry.
    java_home = os.environ.get("JAVA_HOME")
    if not (java_home and (Path(java_home) / "bin" / "java.exe").is_file()):
        java_home = None
    if not java_home:
        java_home_file = root / "data" / "tools" / "lab2_java_home.txt"
        if java_home_file.is_file():
            candidate = java_home_file.read_text(encoding="utf-8").strip()
            if (Path(candidate) / "bin" / "java.exe").is_file():
                java_home = candidate
    if not java_home and sys.platform == "win32":
        import winreg

        for key_path in (r"SOFTWARE\JavaSoft\JDK", r"SOFTWARE\JavaSoft\Java Development Kit"):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                    for index in range(winreg.QueryInfoKey(key)[0]):
                        version = winreg.EnumKey(key, index)
                        with winreg.OpenKey(key, version) as jdk_key:
                            candidate, _ = winreg.QueryValueEx(jdk_key, "JavaHome")
                        if (Path(candidate) / "bin" / "java.exe").is_file() and any(
                            marker in version for marker in ("21", "17")
                        ):
                            java_home = candidate
                            break
            except OSError:
                continue
            if java_home:
                break
    if java_home:
        os.environ["JAVA_HOME"] = str(java_home)
        os.environ["PATH"] = str(Path(java_home) / "bin") + os.pathsep + os.environ.get("PATH", "")
    print(f"JAVA_HOME={java_home or 'not resolved; Spark cells will fail'}")

    # Hadoop 3.x on Windows needs winutils.exe for permission-setting mkdirs
    # (Spark event log directory). Ship it under the ignored data/tools.
    hadoop_home = root / "data" / "tools" / "hadoop"
    if (hadoop_home / "bin" / "winutils.exe").is_file():
        os.environ["HADOOP_HOME"] = str(hadoop_home)

    for variable, directory in {
        "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
        "JUPYTER_CONFIG_DIR": "jupyter-config",
        "IPYTHONDIR": "ipython",
        "MPLCONFIGDIR": "matplotlib",
    }.items():
        path = runtime / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if args.csv:
        os.environ["LAB2_SOURCE_CSV"] = str(args.csv.resolve())
    os.environ["LAB2_FORCE"] = "1" if args.force else "0"

    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    class ProgressClient(NotebookClient):
        """Echo kernel stream output so a long headless run stays visible."""

        def output(self, outs, msg, display_id, cell_index):
            out = super().output(outs, msg, display_id, cell_index)
            if out is not None and out.get("output_type") == "stream":
                for line in out.get("text", "").splitlines():
                    if line.strip():
                        print(f"[kernel] {line}", flush=True)
            return out

    notebook_path = root / "lab2" / "lab2_engines_benchmark.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    # An explicit temporary kernel ensures the notebook uses this interpreter.
    with tempfile.TemporaryDirectory(prefix="kernel-", dir=runtime) as kernel_root:
        spec_dir = Path(kernel_root) / "lab2"
        spec_dir.mkdir()
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Lab 2 local environment",
            "language": "python",
        }), encoding="utf-8")
        manager = KernelManager(
            kernel_name="lab2",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[kernel_root]),
        )
        client = ProgressClient(
            notebook, km=manager, timeout=args.timeout, allow_errors=False,
            resources={"metadata": {"path": str(root)}},
        )

        original_execute_cell = client.execute_cell
        code_total = sum(cell.cell_type == "code" for cell in notebook.cells)
        cell_counter = {"n": 0}

        def run_cell_with_progress(cell, *args, **kwargs):
            if cell.cell_type == "code":
                cell_counter["n"] += 1
                source = cell.source.splitlines()
                first = next((line.strip() for line in source
                              if line.strip() and not line.strip().startswith("#")), "")
                print(f"[cell {cell_counter['n']}/{code_total}] {first[:80]}", flush=True)
            return original_execute_cell(cell, *args, **kwargs)

        client.execute_cell = run_cell_with_progress
        stop_heartbeat = __import__("threading").Event()

        def heartbeat() -> None:
            import time as _time

            started = _time.time()
            while not stop_heartbeat.wait(60):
                print(f"[runner] elapsed {_time.time() - started:,.0f} s", flush=True)

        heartbeat_thread = __import__("threading").Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            client.execute()
        except Exception:
            nbformat.write(notebook, runtime / "failed_execution.ipynb")
            raise
        finally:
            stop_heartbeat.set()
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)

    png_outputs = sum(
        1
        for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )
    if png_outputs < 5:
        nbformat.write(notebook, runtime / "failed_execution.ipynb")
        raise RuntimeError(f"Expected at least 5 inline PNG chart outputs; got {png_outputs}")
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    errors = [
        (index, output.get("ename"), output.get("evalue"))
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    if errors:
        raise RuntimeError(f"Notebook contains error outputs: {errors}")
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {code_cells} code cells successfully with {png_outputs} inline PNG outputs: {notebook_path}")


if __name__ == "__main__":
    main()
