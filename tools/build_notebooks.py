"""Build the exercise and solution notebooks from tools/lessons/*.py.

Each lesson file defines CELLS, a list built with md(), code() and think().
Inside code cells, anything between `### BEGIN SOLUTION` and `### END SOLUTION`
is kept in solutions/ and replaced by a TODO in notebooks/.

    uv run --group dev python tools/build_notebooks.py            # write solutions/, and notebooks/ that don't exist yet
    uv run --group dev python tools/build_notebooks.py --execute  # also run every solution notebook top to bottom

notebooks/ is where you work, so an existing exercise notebook is never
overwritten unless you pass --force.
"""

import argparse
import importlib.util
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
LESSONS = ROOT / "tools" / "lessons"

SOLUTION_RE = re.compile(r"^([ \t]*)### BEGIN SOLUTION\n.*?^[ \t]*### END SOLUTION\n?", re.S | re.M)


def _exercise_source(src):
    return SOLUTION_RE.sub(lambda m: f"{m.group(1)}# YOUR CODE HERE\n{m.group(1)}raise NotImplementedError\n", src)


def _solution_source(src):
    return re.sub(r"^[ \t]*### (BEGIN|END) SOLUTION\n?", "", src, flags=re.M)


def _cells(cells, solution):
    out = []
    for kind, text, answer in cells:
        text = text.strip("\n")
        if kind == "md":
            out.append(nbformat.v4.new_markdown_cell(text))
        elif kind == "code":
            out.append(nbformat.v4.new_code_cell(_solution_source(text) if solution else _exercise_source(text)))
        elif kind == "think":
            out.append(nbformat.v4.new_markdown_cell(text))
            reply = f"**Answer.** {answer.strip()}" if solution else "*Your answer:* "
            out.append(nbformat.v4.new_markdown_cell(reply))
    return out


def build(lesson_path, force=False, execute=False):
    spec = importlib.util.spec_from_file_location(lesson_path.stem, lesson_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = f"{lesson_path.stem}.ipynb"
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}

    solution_nb = nbformat.v4.new_notebook(cells=_cells(module.CELLS, solution=True), metadata=meta)
    solution_path = ROOT / "solutions" / name
    if execute:
        from nbclient import NotebookClient

        NotebookClient(solution_nb, timeout=600, resources={"metadata": {"path": str(solution_path.parent)}}).execute()
        print(f"executed {solution_path.relative_to(ROOT)} without errors")
        for cell in solution_nb.cells:  # keep committed notebooks free of outputs
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
    nbformat.write(solution_nb, solution_path)
    print(f"wrote {solution_path.relative_to(ROOT)}")

    exercise_path = ROOT / "notebooks" / name
    if exercise_path.exists() and not force:
        print(f"kept  {exercise_path.relative_to(ROOT)} (exists; --force to overwrite your work)")
    else:
        nbformat.write(nbformat.v4.new_notebook(cells=_cells(module.CELLS, solution=False), metadata=meta), exercise_path)
        print(f"wrote {exercise_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("lessons", nargs="*", help="lesson names, e.g. 01_policy_gradient_bandit (default: all)")
    args = parser.parse_args()
    paths = [LESSONS / f"{n}.py" for n in args.lessons] or sorted(LESSONS.glob("[0-9]*.py"))
    for path in paths:
        build(path, force=args.force, execute=args.execute)
