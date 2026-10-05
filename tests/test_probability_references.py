"""The probability layer's references: today's code still writes them, byte for byte.

``tests/data/write_probability_references.py`` writes, from today's
parameter layer and forward model, the prior draws, densities and
predictions the probability layer must reproduce. Rewriting them here and
comparing bytes shows the stored files are what today's code computes, so a
later difference is the new code's.
"""

from __future__ import annotations

from conftest import load_script

SCRIPT = "tests/data/write_probability_references.py"


def test_the_script_reproduces_its_files_byte_for_byte(tmp_path):
    references = load_script(SCRIPT)
    written = references.write_references(tmp_path)
    stored = sorted(references.DIRECTORY.glob("*.nc"))
    assert [path.name for path in stored] == sorted(path.name for path in written)
    for path in stored:
        assert (tmp_path / path.name).read_bytes() == path.read_bytes(), path.name
