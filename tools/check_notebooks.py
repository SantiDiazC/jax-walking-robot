"""Execute tutorials with a real kernel using this Python environment.

Run after installing .[analysis,notebooks]. No executed notebook or widget state
is written to source files; simulation artifacts remain in ignored outputs/.
"""
import argparse
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path,
                        default=[root / "tutorials/01_run_a_design.ipynb"])
    args = parser.parse_args()
    print("Kernel Python:", sys.executable, flush=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    with tempfile.TemporaryDirectory(prefix="robot-notebook-check-") as temporary:
        environment["MPLCONFIGDIR"] = str(Path(temporary) / "matplotlib")
        environment["JUPYTER_RUNTIME_DIR"] = str(Path(temporary) / "runtime")
        environment["IPYTHONDIR"] = str(Path(temporary) / "ipython")
        for filename in args.notebooks:
            notebook = nbformat.read(filename, as_version=4)
            nbformat.validate(notebook)
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            client = NotebookClient(notebook, km=manager, timeout=600,
                                    resources={"metadata": {"path": str(root)}},
                                    on_cell_executed=lambda cell_index, **kwargs:
                                    print(f"  Cell {cell_index}: executed", flush=True))
            client.execute(env=environment, cleanup_kc=True)
            # Widget callbacks can capture errors without failing their cell.
            state = notebook.metadata.get("widgets", {}).get("application/vnd.jupyter.widget-state+json", {})
            outputs = [output for cell in notebook.cells for output in cell.get("outputs", [])]
            for widget in state.get("state", {}).values():
                outputs.extend(widget.get("state", {}).get("outputs", []))
            errors = [output for output in outputs if output.get("output_type") == "error"]
            if errors:
                raise RuntimeError(f"Notebook or widget callback failed in {filename}: {errors}")
            print("PASS:", filename.name, flush=True)


if __name__ == "__main__":
    main()
