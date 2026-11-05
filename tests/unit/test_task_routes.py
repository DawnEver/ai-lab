"""A task routes its calls (user ruling 2026-11-05): ``routes.toml`` names each task's chain and what it falls over on.

The tasks are DERIVED from the routes; a call walks every key slot of one provider, then the next on a failure of
a listed class; one TOML bundle per task replaces that task's slots. Every secret here is a fake.
"""

import io
import json

import pytest
from lab_commons.credentials import api_keys

from ai_lab import ProviderError, ResponseRequest, Text, Unsupported, auth, budget, connect
from ai_lab.client import ERROR_CLASSES, FALL_OVER_ON, error_class, routes, tasks, user_routes_file
from ai_lab.errors import DailyTokenCapReached
from ai_lab.review.method import REVIEW_TASK, configured_providers, default_config
from tests.unit.test_one_account import _post

_ROWS = """
[alpha]
wire = "openai_chat_json"
url = "https://alpha.test"
openai_path = "/v1"
key_name = "ALPHA_API_KEY"
auth = "api"
models = { base = "a-base" }

[beta]
wire = "openai_chat_json"
url = "https://beta.test"
openai_path = "/v1"
key_name = "BETA_API_KEY"
auth = "api"
models = { base = "b-base" }

[local]
wire = "openai_chat_json"
url = "http://localhost:1"
openai_path = "/v1"
auth = "local"
"""

_REQUEST = ResponseRequest('i', (Text('t'),), {'type': 'object'})


def _raw(text='{"a": 1}'):
    return {'choices': [{'message': {'content': text}}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}


def _limited():
    return ProviderError('HTTP 429 from u: slow down (after 3 attempts)', kind='rate_limit', status=429)


def _server():
    return ProviderError('HTTP 500 from u: boom (after 3 attempts)', status=500)


@pytest.fixture
def routed(credential_store):
    """``write(routes_text)``: user rows alpha/beta/local, each keyed row holding one ``work`` task key."""
    root = user_routes_file().parent
    root.mkdir(parents=True, exist_ok=True)
    (root / 'providers.toml').write_text(_ROWS, encoding='utf-8')
    credential_store('alpha.test', 'sk-alpha-work', 'api-key-work')
    credential_store('beta.test', 'sk-beta-work', 'api-key-work')

    def write(text):
        user_routes_file().write_text(text, encoding='utf-8')

    write('[task.work]\nchain = ["alpha", "beta:b-named"]\n')
    return write


def _ledger():
    return [json.loads(line) for line in budget.ledger_file().read_text('utf-8').splitlines()]


# --- the error classes ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('error', 'cls'),
    [
        (ProviderError('x', kind='auth', status=401), 'auth'),
        (ProviderError('x', kind='auth', status=403), 'auth'),
        (ProviderError('x', kind='exhausted', status=402), 'quota'),
        (DailyTokenCapReached('x', scope='alpha'), 'quota'),
        (ProviderError('x', kind='rate_limit', status=429), 'rate_limit'),
        (ProviderError('x', kind='rate_limit', status=529), 'rate_limit'),
        (ProviderError('x', status=503), 'server'),
        (ProviderError('x', status=400), None),
        (ProviderError('x', kind='timeout'), None),
    ],
)
def test_every_failure_has_one_class_from_the_closed_set(error, cls):
    assert error_class(error) == cls
    assert cls is None or cls in ERROR_CLASSES


def test_server_is_a_legal_class_but_not_a_default():
    assert 'server' in ERROR_CLASSES
    assert {'quota', 'auth', 'rate_limit'} == FALL_OVER_ON


# --- the chain ----------------------------------------------------------------------------------------------


def test_a_429_on_the_first_provider_falls_to_the_second_and_the_ledger_names_it(routed):
    post = _post(_limited(), _raw())
    got = connect('alpha:a-base', task='work', post=post).respond(_REQUEST)
    assert [c.url for c in post.calls] == [
        'https://alpha.test/v1/chat/completions',
        'https://beta.test/v1/chat/completions',
    ]
    assert (got.provider, got.model, got.value) == ('beta', 'b-named', {'a': 1})
    assert post.calls[1].headers['Authorization'] == 'Bearer sk-beta-work'
    assert [(row['provider'], row.get('fallback_from')) for row in _ledger()] == [('beta', 'alpha')]


def test_planted_control_without_a_task_a_429_never_moves(routed, credential_store):
    credential_store('alpha.test', 'sk-alpha-default')
    post = _post(_limited(), _raw())
    with pytest.raises(ProviderError, match='rate_limit'):
        connect('alpha:a-base', post=post).respond(_REQUEST)
    assert len(post.calls) == 1


def test_a_500_does_not_fall_over_by_default(routed):
    post = _post(_server(), _raw())
    with pytest.raises(ProviderError, match='HTTP 500'):
        connect('alpha:a-base', task='work', post=post).respond(_REQUEST)
    assert len(post.calls) == 1


def test_a_500_falls_over_when_the_task_lists_server(routed):
    routed('[task.work]\nchain = ["alpha", "beta"]\nfall_over_on = ["server"]\n')
    post = _post(_server(), _raw())
    got = connect('alpha:a-base', task='work', post=post).respond(_REQUEST)
    assert (got.provider, got.model) == ('beta', 'b-base')


def test_the_files_default_applies_to_a_task_that_names_none(routed):
    routed('fall_over_on = ["server"]\n[task.work]\nchain = ["alpha", "beta"]\n')
    assert routes()['work'].fall_over_on == {'server'}


def test_an_empty_fall_over_on_never_moves(routed):
    routed('[task.work]\nchain = ["alpha", "beta"]\nfall_over_on = []\n')
    post = _post(_limited(), _raw())
    with pytest.raises(ProviderError):
        connect('alpha:a-base', task='work', post=post).respond(_REQUEST)
    assert len(post.calls) == 1


def test_a_task_alone_starts_at_the_head_of_its_chain(routed):
    routed('[task.work]\nchain = ["alpha:a-named", "beta"]\n')
    assert connect(task='work').name == 'alpha:a-named'
    routed('[task.work]\nchain = ["beta", "alpha"]\n')
    assert connect(task='work').name == 'beta:b-base'


def test_a_chain_entry_that_cannot_stand_in_is_skipped_and_named(routed):
    """``local`` declares no models, so it cannot take the role; ``beta`` serves. Without beta, both are named."""
    routed('[task.work]\nchain = ["alpha", "local", "beta"]\n')
    post = _post(_limited(), _raw())
    assert connect('alpha:a-base', task='work', post=post).respond(_REQUEST).provider == 'beta'
    routed('[task.work]\nchain = ["alpha", "local"]\n')
    with pytest.raises(ProviderError, match=r'rate_limit.*fallback unavailable: local: local declares no models'):
        connect('alpha:a-base', task='work', post=_post(_limited())).respond(_REQUEST)


# --- the routes file is checked by name ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ('text', 'named'),
    [
        ('[task.work]\nchain = ["alpha", "nowhere"]\n', "unknown provider 'nowhere'"),
        ('[task.work]\nchain = ["alpha"]\nfall_over_on = ["timeout"]\n', "'timeout' is not an error class"),
        ('[task.work]\nchain = ["alpha"]\nfallback = ["beta"]\n', "unknown key 'fallback'"),
        ('fallover = ["auth"]\n[task.work]\nchain = ["alpha"]\n', "unknown key 'fallover'"),
        ('[task.work]\nchain = []\n', 'chain is empty'),
        ('[task.work]\nchain = ["alpha", "alpha:x"]\n', 'appears twice'),
        ('[task.work-2]\nchain = ["alpha"]\n', 'not a task name'),
    ],
)
def test_a_bad_routes_file_is_refused_by_name(routed, text, named):
    routed(text)
    with pytest.raises(Unsupported, match=named):
        routes()


def test_the_tasks_are_derived_from_the_routes(routed):
    assert {'pr-review', 'optimization', 'session', 'live', 'work'} <= tasks()
    with pytest.raises(Unsupported, match="'nightly' is not a routed task"):
        connect('alpha:a-base', task='nightly')
    routed('[task.nightly]\nchain = ["alpha"]\n')
    assert connect('alpha:a-base', task='nightly').task == 'nightly'


def test_the_packaged_routes_give_every_task_a_chain_and_live_never_falls_over():
    table = routes()
    assert {name for name, _ in table['live'].chain} == {'anthropic', 'openai'}
    assert table['live'].fall_over_on == frozenset()
    assert [name for name, _ in table['pr-review'].chain] == ['anthropic', 'deepseek']


# --- the review's default provider is the route's head -------------------------------------------------------


def test_the_reviews_default_provider_is_the_head_of_its_route(routed):
    assert configured_providers({}, 'headless') == ['anthropic']
    routed(f'[task.{REVIEW_TASK}]\nchain = ["beta", "alpha"]\n')
    assert default_config()['providers'] == {'interactive': ['beta'], 'headless': ['beta']}
    assert configured_providers({'providers': {'headless': ['alpha']}}, 'headless') == ['alpha']


# --- the bundle ----------------------------------------------------------------------------------------------


def _bundle(monkeypatch, text, *args):
    monkeypatch.setattr('sys.stdin', io.StringIO(text))
    return auth.main(['login', '--task', 'work', '--bundle', '-', *args])


def test_a_bundle_stores_each_providers_keys_in_slot_order_and_prints_none(routed, monkeypatch, capsys):
    text = '[keys]\nalpha = ["sk-a1", "sk-a2"]\nbeta = ["sk-b1"]\n'
    assert _bundle(monkeypatch, text) == 0
    assert api_keys('alpha.test', task='work') == ('sk-a1', 'sk-a2')
    assert api_keys('beta.test', task='work') == ('sk-b1',)
    out = capsys.readouterr()
    assert 'sk-' not in out.out + out.err


def test_a_bundle_replaces_the_tasks_slots_so_a_shorter_one_erases_the_rest(routed, monkeypatch):
    assert _bundle(monkeypatch, '[keys]\nalpha = ["sk-a1", "sk-a2", "sk-a3"]\n') == 0
    assert _bundle(monkeypatch, '[keys]\nalpha = ["sk-a9"]\n') == 0
    assert api_keys('alpha.test', task='work') == ('sk-a9',)
    assert api_keys('beta.test', task='work') == ('sk-beta-work',)  # a provider the bundle omits is untouched


@pytest.mark.parametrize(
    ('text', 'named'),
    [
        ('[keys]\nalpha = ["sk-new"]\nnowhere = ["sk-x"]\n', "unknown provider 'nowhere'"),
        ('[keys]\nalpha = ["sk-new"]\nlocal = ["sk-x"]\n', 'local needs no key'),
        ('[keys]\nalpha = ["sk-new"]\nopenai = ["sk-x"]\n', "openai is not in task 'work'"),
        ('[keys]\nalpha = ["sk-new", "sk-new"]\n', 'given twice'),
        ('alpha = ["sk-new"]\n', 'one [keys] table'),
        ('[keys]\nalpha = []\n', 'one or more'),
    ],
)
def test_a_bad_bundle_is_refused_by_name_and_stores_nothing(routed, monkeypatch, capsys, text, named):
    assert _bundle(monkeypatch, text) == 1
    err = capsys.readouterr().err
    assert named in err
    assert 'sk-new' not in err
    assert api_keys('alpha.test', task='work') == ('sk-alpha-work',)


def test_a_bundle_needs_a_routed_task_and_takes_no_provider(routed, monkeypatch, capsys):
    monkeypatch.setattr('sys.stdin', io.StringIO('[keys]\nalpha = ["sk-new"]\n'))
    assert auth.main(['login', '--task', 'nightly', '--bundle', '-']) == 1
    assert 'not a routed task' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        auth.main(['login', 'alpha', '--task', 'work', '--bundle', '-'])
