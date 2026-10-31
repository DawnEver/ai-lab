"""``ai_lab.review.plan``: method data + project config + manifest -> reviewers with providers and prompts.

The planner sharp-review's Node host ran moved here with the method (PLAN-a-short-robust-toolchain P4); the host
collects the manifest, calls ``python -m ai_lab.review.plan`` and fans the planned reviewers out.
"""

import io
import json

import pytest
from dev_lab.dev.rules import RULES

from ai_lab import Provider, catalogue
from ai_lab.review.method import DEFAULT_CONFIG, load_method
from ai_lab.review.plan import build_plan, effective_profiles, eligible_weights, main, report_path
from ai_lab.review.rules import render_prompt
from tests.review_principles import PRINCIPLES, PRINCIPLES_FILE

MANIFEST = {'mode': 'review', 'range': 'HEAD', 'stats': {}, 'excludedSummary': '', 'diff': '+x'}
BASE = {'manifest': MANIFEST, 'rands': {'pick': [0, 0], 'shuffle': [0, 0]}, 'run_id': 'r', 'started_at': 't',
        'date': '2026-10-20', 'seed': 0}  # fmt: skip


@pytest.fixture(scope='module')
def method(method_dir):
    # The family's registry, given as dev-lab's review command gives it: the prompt budget is paid with its ids.
    return load_method(method_dir, principles=PRINCIPLES, registry=tuple(rule.id for rule in RULES))


def _plan(method, **kw):
    return build_plan(method, **{**BASE, **kw})


@pytest.mark.parametrize('run_mode', ['interactive', 'headless'])
def test_the_shipped_default_config_sends_every_run_mode_to_anthropic_only(method, run_mode):
    # User ruling 2026-10-28: a DEFAULT CONFIGURATION, not method data -- no request, no project config.
    p = _plan(method, run_mode=run_mode, forced='diff', available=['anthropic', 'claude', 'codex', 'kimi'])
    assert {r['provider'] for r in p['reviewers']} == {'anthropic'}


def test_a_project_config_overrides_the_shipped_default(method):
    cfg = {'providers': {'interactive': ['kimi']}}
    p = _plan(method, config=cfg, forced='diff', available=['anthropic', 'kimi'])
    assert [r['provider'] for r in p['reviewers']] == ['kimi']


def test_interactive_picks_two_profiles_over_the_alphabetical_roster_rotated_by_seed(method):
    # Planted control: with the default config emptied the method's roster rotation still runs.
    p = _plan(method, sources=['diff'], available=['kimi', 'claude', 'codex'], seed=1, defaults={})
    assert len(p['reviewers']) == 2
    assert sorted(r['provider'] for r in p['reviewers']) == ['codex', 'kimi']
    for r in p['reviewers']:
        assert (r['status'], r['call_mode']) == ('pending', 'review')
        assert '+x' in r['prompt']


def test_interactive_anthropic_reviewers_carry_the_model_the_method_declares(method):
    assert _plan(method, forced='diff', available=['anthropic'])['reviewers'][0]['model'] == 'opus'


def test_a_diff_profile_on_an_empty_manifest_is_skipped_and_an_explore_profile_still_runs(method):
    empty = {**MANIFEST, 'mode': 'empty', 'diff': None}
    p = _plan(method, manifest=empty, sources=['diff', 'deps'], available=['anthropic'],
              rands={'pick': [0, 0.99], 'shuffle': [0, 0]})  # fmt: skip
    by_name = {r['name']: r for r in p['reviewers']}
    assert by_name['diff']['status'] == 'skipped'
    assert (by_name['deps']['status'], by_name['deps']['call_mode']) == ('pending', 'agent')


def test_a_forced_unknown_profile_is_refused_by_name(method):
    with pytest.raises(ValueError, match='unknown profile nope'):
        _plan(method, forced='nope', available=['claude'])


def test_custom_profiles_add_to_the_method_set_and_weights_retune_it(method):
    profiles = effective_profiles(method, {
        'customProfiles': [{'key': 'hygiene', 'source': 'codebase', 'context': 'explore', 'scope': ['dead code'],
                            'weight': 0.3}],
        'profileWeights': {'architecture': 0},
    })  # fmt: skip
    assert profiles['hygiene']['label'] == 'hygiene'
    assert eligible_weights(profiles, sources=['codebase']) == [('architecture', 0), ('hygiene', 0.3)]


def test_an_invalid_custom_profile_is_refused_by_name_not_skipped(method):
    with pytest.raises(ValueError, match=r"'x'.*lacks context, scope"):
        effective_profiles(method, {'customProfiles': [{'key': 'x', 'source': 'diff'}]})


def test_headless_defaults_to_anthropic_diff_profiles_no_synthesis_stdout_sink(method):
    explore = {**MANIFEST, 'mode': 'explore', 'manifest': 'x.py  +1 -0'}  # headless never inlines the diff
    p = _plan(method, run_mode='headless', sources=[], available=['anthropic'], manifest=explore)
    assert p['reviewers']
    contexts = {row['key']: row['context'] for row in method.profiles}
    for r in p['reviewers']:
        assert (r['provider'], r['status'], contexts[r['name']]) == ('anthropic', 'pending', 'diff')
    assert (p['run_mode'], p['synthesis'], p['sink']) == ('headless', False, 'stdout')


def test_a_requested_provider_that_is_not_available_is_unavailable_never_substituted(method):
    p = _plan(method, run_mode='headless', forced='diff', available=['deepseek'])
    assert [(r['provider'], r['status']) for r in p['reviewers']] == [('anthropic', 'unavailable')]
    assert 'prompt' not in p['reviewers'][0]


def test_requested_providers_override_the_run_mode_and_the_project_config_sits_between(method):
    cfg = {'providers': {'headless': ['kimi']}}
    from_config = _plan(method, run_mode='headless', config=cfg, forced='diff', available=['kimi'])
    assert from_config['reviewers'][0]['provider'] == 'kimi'
    from_flag = _plan(method, run_mode='headless', config=cfg, requested=['deepseek'], forced='diff',
                      available=['kimi', 'deepseek'])  # fmt: skip
    assert from_flag['reviewers'][0]['provider'] == 'deepseek'


def test_interactive_with_an_explicit_provider_list_keeps_it_and_marks_the_missing_one(method):
    p = _plan(method, requested=['codex', 'claude'], sources=['diff'], available=['claude'])
    assert [(r['provider'], r['status']) for r in p['reviewers']] == [('codex', 'unavailable'), ('claude', 'pending')]


def test_a_call_mode_the_run_mode_cannot_execute_is_unsupported(method):
    big = {**MANIFEST, 'mode': 'agent', 'diff': None, 'manifest': '| file |'}
    p = _plan(method, run_mode='headless', manifest=big, forced='diff', available=['anthropic'])
    assert p['reviewers'][0]['status'] == 'unsupported'


def test_an_unknown_run_mode_is_refused_by_name(method):
    with pytest.raises(ValueError, match='unknown run mode batch'):
        _plan(method, run_mode='batch', forced='diff')


def test_the_plan_names_the_top_confidence_label_a_host_counts_by(method):
    assert _plan(method, forced='diff')['high_confidence'] == method.spec['merge']['confidence'][0]['label']


def test_the_module_reads_a_request_on_stdin_and_prints_the_plan(monkeypatch, capsys):
    request = {**BASE, 'forced': 'diff', 'available': ['anthropic']}
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(request)))
    assert main(['--principles', str(PRINCIPLES_FILE), '--rules', '']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert (plan['run_id'], plan['reviewers'][0]['status']) == ('r', 'pending')


def test_the_module_draws_its_own_randomness_when_the_request_carries_none(monkeypatch, capsys):
    request = {k: v for k, v in BASE.items() if k != 'rands'} | {'sources': ['diff'], 'available': ['anthropic']}
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(request)))
    assert main(['--principles', str(PRINCIPLES_FILE), '--rules', '']) == 0
    assert len(json.loads(capsys.readouterr().out)['reviewers']) == 2


@pytest.mark.parametrize('text', ['not json', '[]', '{"date": "d"}'])
def test_the_module_refuses_a_malformed_request_with_exit_2(monkeypatch, capsys, text):
    monkeypatch.setattr('sys.stdin', io.StringIO(text))
    assert main(['--principles', str(PRINCIPLES_FILE), '--rules', '']) == 2
    out = capsys.readouterr()
    assert out.out == ''
    assert out.err.startswith('refused: ')


# -- The reviewer prompt is paid for on every review: its fixed part (everything but the diff and the manifest)
# has a declared ceiling, the current size plus a small margin, that may only go down.

#: Characters of the largest rendered prompt with an empty diff and manifest.
PROMPT_BUDGET_CHARS = 6500  # raised 2026-10-31: the exploratory review's request vocabulary (largest 6423)
#: How far the ceiling may sit above the largest prompt before it must be lowered.
MARGIN_CHARS = 100
EMPTY = {'range': '', 'excludedSummary': '', 'diff': '', 'manifest': ''}


def _fixed_prompts(method, profiles):
    out = {}
    for profile in profiles:
        for mode in method.spec['contexts'][profile['context']]:
            rendered = render_prompt(method, profile, {**EMPTY, 'mode': mode})
            if 'prompt' in rendered:
                out[f'{profile["key"]}/{mode}'] = len(rendered['prompt'])
    return out


def test_every_reviewer_prompt_without_its_diff_fits_the_budget_and_the_budget_is_tight(method):
    prompts = _fixed_prompts(method, method.profiles)
    assert len(prompts) >= 3, 'a scan that renders nothing passes anything'
    assert {k: v for k, v in prompts.items() if v > PROMPT_BUDGET_CHARS} == {}
    largest = max(prompts.values())
    assert PROMPT_BUDGET_CHARS - largest <= MARGIN_CHARS, f'largest is {largest}; lower the ceiling'


def test_planted_a_profile_whose_framing_overflows_the_budget_is_convicted(method):
    planted = {**method.profiles[0], 'key': 'planted', 'framing': 'x' * PROMPT_BUDGET_CHARS}
    assert any(v > PROMPT_BUDGET_CHARS for v in _fixed_prompts(method, [planted]).values())


def test_report_path_is_regenerable_output_under_the_dated_logs_tree():
    assert report_path('2026-10-20') == 'output/logs/26/10/20/sharp-review/sharp-review.md'


def test_interactive_plan_sinks_a_report_at_the_report_path_never_memory(method):
    p = _plan(method, available=['anthropic'])
    assert (p['sink'], p['report']) == ('report', report_path('2026-10-20'))


def _run_mode_providers(method):
    """Every provider a run mode chooses with no request: the shipped default config and each listed mode."""
    named = {p for one in DEFAULT_CONFIG['providers'].values() for p in one}
    return named | {p for mode in method.spec['run_modes'].values() if isinstance(mode['providers'], list)
                    for p in mode['providers']}  # fmt: skip


def test_every_shipped_row_a_run_mode_chooses_declares_its_models(method):
    rows = catalogue()
    chosen = sorted(p for p in _run_mode_providers(method) if p in rows)
    assert chosen == ['anthropic']
    assert {p: rows[p].models.get('base') for p in chosen} == {'anthropic': 'claude-sonnet-5-5'}


def test_planted_a_run_mode_provider_without_models_is_refused_naming_the_remedy(method):
    rows = {'anthropic': Provider('anthropic', 'anthropic_messages', 'https://x.test', 'api', anthropic_path='/v1')}
    with pytest.raises(ValueError, match=r'^provider anthropic declares no models; add \[anthropic\.models\] base='):
        _plan(method, forced='diff', available=['anthropic'], rows=rows)
    # A REQUESTED provider is the caller's choice, not a run mode's: it is planned as given.
    assert _plan(method, forced='diff', available=['anthropic'], requested=['anthropic'], rows=rows)['reviewers']
