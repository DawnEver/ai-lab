"""Every fixture under the method's ``conformance/`` holds for this executor; the Node one runs the same files.

The fixtures are the single source of the rules the data cannot state. They are read IN PLACE and every
file is a case: a new fixture is exercised without a line changing here, and an empty kind directory fails.
"""

import json
from pathlib import Path

import pytest

from ai_lab.review.method import load_method
from ai_lab.review.rules import merge, pick, render, render_prompt
from tests.conftest import method_path

_ROOT = method_path() / 'conformance'
_KINDS = ('merge', 'pick', 'prompt')


def _cases(kind: str) -> list:
    files = sorted((_ROOT / kind).glob('*.json'))
    # Never a skip: a missing directory is one failing case naming it.
    return [pytest.param(f, id=f.stem) for f in files] or [pytest.param(_ROOT / kind, id=f'{kind}-absent')]


def _fixture(path: Path) -> dict:
    assert path.is_file(), f'no conformance fixtures at {path}: set SHARP_REVIEW_METHOD'
    data = json.loads(path.read_text(encoding='utf-8'))
    assert set(data) == {'description', 'input', 'expected'}, path
    return data


@pytest.fixture(scope='module')
def method(method_dir):
    return load_method(method_dir)


def test_every_kind_has_fixtures():
    assert {p.name for p in _ROOT.iterdir() if p.is_dir()} == set(_KINDS)


@pytest.mark.parametrize('path', _cases('pick'))
def test_pick(path, method):
    case = _fixture(path)
    got = pick(method, [tuple(w) for w in case['input']['weights']], case['input']['count'], case['input']['rands'])
    assert got == case['expected'], case['description']


@pytest.mark.parametrize('path', _cases('prompt'))
def test_prompt(path, method):
    case = _fixture(path)
    given = case['input']
    if 'template' in given:
        got = render(method, given['template'], given['vars'])
    else:
        got = render_prompt(method, method.profile(given['profile']), given['manifest'])
    assert got == case['expected'], case['description']


@pytest.mark.parametrize('path', _cases('merge'))
def test_merge(path, method):
    case = _fixture(path)
    given = case['input']
    got = merge(
        method, range_=given['range'], date=given['date'], profiles=given['profiles'],
        reviewers=given['reviewers'], results=given['results'],
    )  # fmt: skip
    assert got == case['expected'], case['description']
