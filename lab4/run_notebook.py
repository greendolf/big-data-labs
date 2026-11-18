"""ЛР4: выполнить отчётный notebook этим интерпретатором (headless, с логом прогресса).

Весь рантайм — под игнорируемым data/lab4 (jupyter-каталоги, временные каталоги).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # консоль cp1251 не переносит юникод лога
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=4 * 60 * 60, help="Timeout per code cell, seconds")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    runtime = root / "data" / "lab4"
    runtime.mkdir(parents=True, exist_ok=True)
    temp_dir = runtime / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for variable in ("TEMP", "TMP", "TMPDIR"):
        os.environ[variable] = str(temp_dir)
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

    notebook_path = root / "lab4" / "lab4_ml_production.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    with tempfile.TemporaryDirectory(prefix="kernel-", dir=runtime) as kernel_root:
        spec_dir = Path(kernel_root) / "lab4"
        spec_dir.mkdir()
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Lab 4 local environment",
            "language": "python",
        }), encoding="utf-8")
        manager = KernelManager(
            kernel_name="lab4",
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
        try:
            client.execute()
        except Exception:
            nbformat.write(notebook, runtime / "failed_execution.ipynb")
            raise
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)

    png_outputs = sum(
        1
        for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )
    if png_outputs < 4:
        nbformat.write(notebook, runtime / "failed_execution.ipynb")
        raise RuntimeError(f"Expected at least 4 inline PNG chart outputs; got {png_outputs}")
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
