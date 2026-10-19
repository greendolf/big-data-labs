"""Run the complete Lab 1 report with bounded local resources."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


THREAD_LIMITS = {
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "NUMEXPR_MAX_THREADS": "2",
    "POLARS_MAX_THREADS": "2",
    "ARROW_NUM_THREADS": "2",
    "RAYON_NUM_THREADS": "2",
    "LOKY_MAX_CPU_COUNT": "2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="Kaggle CSV path (defaults to data/raw/ids/BoTNeTIoT-L01-v2.csv)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild profile and artifacts")
    parser.add_argument("--timeout", type=int, default=4 * 60 * 60, help="Timeout per cell, seconds")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    runtime = root / "data" / "lab1"
    runtime.mkdir(parents=True, exist_ok=True)
    source = (args.source or root / "data" / "raw" / "ids" / "BoTNeTIoT-L01-v2.csv").resolve()
    if not source.is_file():
        parser.error(
            f"Kaggle CSV not found: {source}. Download the dataset while signed in to Kaggle, "
            "then run lab1/download_dataset.py --archive <downloaded-archive.zip>."
        )

    for variable, directory in {
        "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
        "JUPYTER_CONFIG_DIR": "jupyter-config",
        "IPYTHONDIR": "ipython",
        "MPLCONFIGDIR": "matplotlib",
        "TMP": "tmp",
        "TEMP": "tmp",
        "TMPDIR": "tmp",
    }.items():
        path = runtime / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    for variable, value in THREAD_LIMITS.items():
        os.environ[variable] = value
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["LAB1_SOURCE_CSV"] = str(source)
    os.environ["LAB1_RUNTIME_DIR"] = str(runtime)
    os.environ["LAB1_RESULTS_DIR"] = str(root / "lab1" / "results")
    os.environ["LAB1_FORCE"] = "1" if args.force else "0"

    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    notebook_path = root / "lab1" / "lab1_big_data_scenario.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    # Keep kernel and temporary Spark/Arrow/Parquet files under ignored data/.
    with tempfile.TemporaryDirectory(prefix="kernel-", dir=runtime) as kernel_root:
        spec_dir = Path(kernel_root) / "lab1"
        spec_dir.mkdir()
        (spec_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Lab 1 bounded local environment",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        manager = KernelManager(
            kernel_name="lab1",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[kernel_root]),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=args.timeout,
            allow_errors=False,
            resources={"metadata": {"path": str(root)}},
        )
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
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    errors = [
        (index, output.get("ename"), output.get("evalue"))
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    if errors:
        raise RuntimeError(f"Notebook contains error outputs: {errors}")
    print(f"Executed {code_cells} code cells successfully with {png_outputs} inline PNG outputs: {notebook_path}")


if __name__ == "__main__":
    main()
