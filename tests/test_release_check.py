"""The release workflow's first gate: is this version ready to tag?

Contract (``~/.claude/RELEASING.md`` §1 and §2, step 1, written by the owner):
a release ``X.Y.Z`` goes ahead only when

* the version is plain semver, three integers;
* CHANGELOG.md has an entry headed for it, ``## X.Y.Z`` or Keep a
  Changelog's ``## [X.Y.Z] - date``, as this project's changelog is written;
* the tag ``vX.Y.Z`` does not exist yet;
* the commit to release is the head of main, or, for a patch to an older
  minor, the head of that minor's ``release/X.Y`` branch (the rule the owner
  chose for TargetGym on 2026-11-01; every package releases the same way).

``release_check.problems`` returns what blocks the release, empty when
nothing does. Each case below breaks one rule, so it must report exactly one
problem.
"""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "_release_check", ROOT / "scripts" / "release_check.py"
)
release_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_check)

CHANGELOG = """# Changelog

## [Unreleased]

## [0.7.0] - 2026-11-02

### Changed

- Something.
"""
TAGS = ["0.4.0", "0.5.1", "v0.5.2"]
HEAD = "a" * 40


def _problems(**changes):
    args = dict(
        version="0.7.0",
        changelog=CHANGELOG,
        tags=TAGS,
        commit=HEAD,
        branch="main",
        branch_head=HEAD,
    )
    args.update(changes)
    return release_check.problems(**args)


def test_a_ready_release_has_no_problems():
    assert _problems() == []


def test_a_plain_heading_counts_as_the_entry():
    assert _problems(changelog="# Changelog\n\n## 0.7.0\n\n- Something.\n") == []


@pytest.mark.parametrize("version", ["0.7", "v0.7.0", "0.7.0rc1", "0.7.0 ", "a.b.c"])
def test_a_version_that_is_not_three_integers_is_refused(version):
    assert len(_problems(version=version)) >= 1


@pytest.mark.parametrize(
    "changelog",
    [
        "# Changelog\n\n## [Unreleased]\n\n- Something.\n",
        # A longer version that merely contains this one.
        "# Changelog\n\n## [10.7.0] - 2026-11-02\n",
        "# Changelog\n\n## [0.7.00] - 2026-11-02\n",
        "# Changelog\n\n## 0.7.00\n",
        # A subsection, not an entry.
        "# Changelog\n\n### 0.7.0\n",
        # Mentioned in prose only.
        "# Changelog\n\nSee 0.7.0 below.\n",
    ],
    ids=[
        "unreleased-only",
        "longer-version",
        "padded",
        "padded-plain",
        "subsection",
        "prose",
    ],
)
def test_a_missing_changelog_entry_is_refused(changelog):
    assert len(_problems(changelog=changelog)) == 1


def test_an_existing_tag_is_refused():
    assert len(_problems(tags=[*TAGS, "v0.7.0"])) == 1


def test_a_commit_behind_its_branch_is_refused():
    assert len(_problems(commit="b" * 40)) == 1


def test_a_patch_comes_from_its_minor_release_branch():
    changelog = "# Changelog\n\n## [0.6.2]\n\n- Something fixed.\n"
    assert _problems(version="0.6.2", changelog=changelog, branch="release/0.6") == []


def test_a_release_branch_releases_only_its_own_minor():
    assert len(_problems(branch="release/0.6")) == 1


@pytest.mark.parametrize("branch", ["feature/gate", "release/0.6.x", "releases/0.7"])
def test_any_other_branch_is_refused(branch):
    assert len(_problems(branch=branch)) == 1


# The GitHub Release's notes (RELEASING.md §2, step 7): "the version's
# CHANGELOG entry as notes". Contract for ``release_check.notes``: the text
# under the version's heading, up to the next entry's heading, without the
# heading itself or surrounding blank lines; subsections belong to the entry.
# A version with no entry has no notes, and asking for them is an error.

TWO_ENTRIES = """# Changelog

## [Unreleased]

- Not released.

## [0.7.0] - 2026-11-02

### Changed

- Something.

### Fixed

- Something else.

## [0.6.0]

- Older.
"""


def test_the_notes_are_the_entry_without_its_heading_or_its_neighbours():
    assert release_check.notes("0.7.0", TWO_ENTRIES) == (
        "### Changed\n\n- Something.\n\n### Fixed\n\n- Something else."
    )


def test_the_last_entry_runs_to_the_end_of_the_changelog():
    assert release_check.notes("0.6.0", TWO_ENTRIES) == "- Older."


def test_a_plain_heading_has_notes_too():
    assert release_check.notes("0.7.0", "## 0.7.0\n\n- Something.\n") == "- Something."


@pytest.mark.parametrize("version", ["0.8.0", "7.0", "Unreleased"])
def test_a_version_with_no_entry_has_no_notes(version):
    with pytest.raises(ValueError):
        release_check.notes(version, TWO_ENTRIES)
