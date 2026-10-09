"""Prices: catalogs cross-checked into dated snapshots, read at the call's time, overridden by the user."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_lab.prices import Price, cost, quote
from ai_lab.usage import Usage

MD = 'https://models.dev/api.json'
LL = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'
OR = 'https://openrouter.ai/api/v1/models'


def _catalogs(md_input: float, ll_input: float, calls: list[str]):
    def get(url: str) -> object:
        calls.append(url)
        if url == MD:
            return {'openai': {'models': {'m': {'cost': {'input': md_input, 'output': 0.5, 'cache_read': 0.01}}}}}
        if url == LL:
            return {
                'm': {
                    'litellm_provider': 'openai',
                    'input_cost_per_token': ll_input / 1e6,
                    'output_cost_per_token': 5e-7,
                }
            }
        if url == OR:
            return {'data': [{'id': 'openai/m', 'pricing': {'prompt': '0.00000001', 'completion': '0.0000001'}}]}
        raise OSError(url)  # portkey unreachable: one source failing is not a refusal

    return get


def test_agreeing_catalogs_give_their_median_and_the_reseller_is_ignored(tmp_path) -> None:
    calls: list[str] = []
    q = quote('openai', 'openai', 'm', directory=tmp_path, get=_catalogs(0.10, 0.10, calls), user={})
    assert q.agreed and q.price == Price(0.10, 0.5, 0.01) and set(q.sources) == {'models.dev', 'litellm'}


def test_disagreeing_catalogs_are_flagged(tmp_path) -> None:
    q = quote('deepseek', 'openai', 'm', directory=tmp_path, get=_catalogs(0.15, 0.30, []), user={})
    assert not q.agreed and q.price.input == pytest.approx(0.225)


def test_a_fresh_snapshot_is_reused_and_a_past_call_never_fetches(tmp_path) -> None:
    calls: list[str] = []
    get = _catalogs(0.10, 0.10, calls)
    quote('openai', 'openai', 'm', directory=tmp_path, get=get, user={})
    fetched = len(calls)
    quote('openai', 'openai', 'm', directory=tmp_path, get=get, user={})
    assert len(calls) == fetched, 'a snapshot younger than a day is reused'
    past = datetime.now(UTC) - timedelta(days=30)
    q = quote('openai', 'openai', 'm', at=past, directory=tmp_path, get=get, user={})
    assert len(calls) == fetched and q is not None, 'a past call is priced from the earliest snapshot, never fetched'


def test_the_users_rate_wins_and_an_unmetered_provider_is_unpriced(tmp_path) -> None:
    user = {'openai:m': {'input': 1.0, 'output': 2.0}}
    assert (
        quote('openai', 'openai', 'm', directory=tmp_path, get=_catalogs(0.1, 0.1, []), user=user).origin
        == 'prices.toml'
    )
    assert quote('claude_code', '', 'haiku', directory=tmp_path, get=_catalogs(0.1, 0.1, []), user={}) is None


def test_cost_bills_cached_input_at_its_rate_and_the_clis_own_dollars_win() -> None:
    price = Price(1.0, 4.0, 0.1)
    assert cost(Usage(1_000_000, 500_000, 1_000_000), price) == pytest.approx(2.1)
    assert cost(Usage(10, 10), None) is None
    assert cost(Usage(1, 1, cost_usd=0.5), None) == 0.5
