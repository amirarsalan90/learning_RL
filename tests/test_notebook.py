"""Execute the lesson's Python cells without requiring a Jupyter installation.

This validates the scientific code; it is not a Jupyter kernel/UI integration test.
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest


class NotebookTests(unittest.TestCase):
    def test_notebook_structure_and_python_cells(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "01_reinforce_bandit.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")
        namespace = {"__name__": "__notebook__"}
        seen_ids = set()
        output = io.StringIO()
        with redirect_stdout(output):
            for cell in notebook["cells"]:
                self.assertNotIn(cell["id"], seen_ids)
                seen_ids.add(cell["id"])
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    exec(compile("".join(cell["source"]), f"{path}:{cell['id']}", "exec"), namespace)
        self.assertEqual(len(namespace["rows"]), 3 * 5 * 51)
        self.assertIn("<svg", namespace["svg"])
        self.assertIn("P(optimal)", output.getvalue())
