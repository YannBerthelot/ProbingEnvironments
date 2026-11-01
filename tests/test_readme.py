"""The README's examples run as written.

Contract (RELEASING.md §3): every ```python block in README.md executes, in
order and in one shared namespace, without raising. The check examples train
PPO, so this is a medium test.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.mark.medium
def test_the_readme_python_examples_run():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.S)
    assert blocks, "README.md has no python examples"
    namespace: dict = {}
    for block in blocks:
        # Executing the README's own code is what this test is for.
        exec(  # pylint: disable=exec-used
            compile(block, str(README), "exec"), namespace
        )
