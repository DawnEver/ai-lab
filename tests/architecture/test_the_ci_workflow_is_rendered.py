"""The GitHub Actions workflow is RENDERED from TOML (`dev_lab.dev.ciworkflows`): no drift, no hand-written file."""

from __future__ import annotations

from pathlib import Path

from dev_lab.dev.ciworkflows import drift, workflows

ROOT = Path(__file__).resolve().parents[2]


def test_the_repo_runs_the_family_pr_lifecycle() -> None:
    # dev-lab #92: a GitHub repo renders its forge's whole base set; the judge prepares its own bundle (D73/#772).
    # review-again.yml is no longer rendered: the review role is bound off CI (removed on integrate/main 2026-11-20).
    assert set(workflows(ROOT)) == {
        'ci.yml',
        'continue.yml',
        'merge-tick.yml',
    }


def test_every_workflow_matches_its_toml() -> None:
    assert drift(ROOT) == []
