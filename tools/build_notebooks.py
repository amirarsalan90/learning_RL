"""Build the notebooks in notebooks/ from tools/lessons/*.py.

Each lesson file defines CELLS, a list of ("md", text) and ("code", text) cells.

    uv run --group dev python tools/build_notebooks.py                 # write all notebooks
    uv run --group dev python tools/build_notebooks.py --execute 01_policy_gradient_bandit   # run it, keep outputs

Smoke-test every notebook on CPU with a tiny random model (outputs are not kept):

    uv run --group dev python tools/make_tiny_model.py /tmp/tiny-qwen
    RLCOURSE_MODEL=/tmp/tiny-qwen RLCOURSE_SMOKE=1 uv run --group dev python tools/build_notebooks.py --check
"""

import argparse
import ast
import importlib.util
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
LESSONS = ROOT / "tools" / "lessons"
META = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}


def _quiet_tuple(src):
    """End a cell with ';' when its last line is a bare tuple like `fig.tight_layout(), plt.show()`,
    so Jupyter doesn't print `(None, None)` under the plot."""
    try:
        last = ast.parse(src).body[-1]
    except (SyntaxError, IndexError):
        return src
    if isinstance(last, ast.Expr) and isinstance(last.value, ast.Tuple):
        return src.rstrip() + ";"
    return src


def load_cells(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cells = []
    for kind, text in module.CELLS:
        text = text.strip("\n")
        cells.append(nbformat.v4.new_markdown_cell(text) if kind == "md" else nbformat.v4.new_code_cell(_quiet_tuple(text)))
    return nbformat.v4.new_notebook(cells=cells, metadata=META)


def execute(nb, cwd):
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=3600, resources={"metadata": {"path": str(cwd)}}).execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run the notebooks and keep their outputs")
    parser.add_argument("--check", action="store_true", help="run the notebooks, but write them without outputs")
    parser.add_argument("lessons", nargs="*", help="lesson names (default: all)")
    args = parser.parse_args()
    paths = [LESSONS / f"{n}.py" for n in args.lessons] or sorted(LESSONS.glob("[0-9]*.py"))
    for path in paths:
        out = ROOT / "notebooks" / f"{path.stem}.ipynb"
        if args.execute or args.check:
            nb = load_cells(path)
            execute(nb, out.parent)
            print(f"ran   {out.relative_to(ROOT)} without errors")
            if args.execute:
                nbformat.write(nb, out)
                print(f"wrote {out.relative_to(ROOT)} with outputs")
                continue
        nbformat.write(load_cells(path), out)
        print(f"wrote {out.relative_to(ROOT)}")
