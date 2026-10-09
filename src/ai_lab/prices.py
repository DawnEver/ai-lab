"""What a token costs: public catalogs cross-checked, snapshotted by time, overridden by the user.

FIRST PRINCIPLES. The bill is tokens times the rate in force when the call was made. Tokens are a
fact of the call (the ledger records them with the call's time); the rate is outside data that
changes. So a rate is never fetched per call: catalogs are read into a dated SNAPSHOT, a cost is
priced against the newest snapshot at or before the call, and a snapshot is refreshed only when it
is older than a day or lacks the model asked about.

WHO BILLS YOU DECIDES WHICH PRICE IS TRUE. A provider row names its billing ``vendor``; the catalogs
are asked about that vendor's model. A reseller (OpenRouter) quotes its own route and is a source
only for the ``openrouter`` vendor -- measured 2026-10-09, it quoted deepseek-v4-flash input at
$0.017/M against the vendor's $0.14-0.30. A provider with no vendor (a local CLI on a subscription)
is unpriced; the ``claude`` CLI reports its own dollars in the usage it returns.

CROSS-CHECK. The vendor-direct catalogs (models.dev, LiteLLM, Portkey) are read; the consensus is
their median, AGREED when they all sit within 5% of it, DISPUTED otherwise -- measured 2026-10-09:
gpt-6-luna agreed across all four, deepseek-v4-flash spread 0.14-0.30. A row in the user's
``prices.toml`` (US dollars per million tokens, keyed ``provider:model``) wins over every catalog:
it is a contract rate.
"""

from __future__ import annotations

import json
import statistics
import tomllib
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lab_commons.paths import config_root, user_cache_path

from ai_lab.client import providers
from ai_lab.usage import Usage

__all__ = [
    'SOURCES',
    'Price',
    'Quote',
    'cost',
    'quote',
    'quote_for',
    'snapshot_dir',
    'user_prices_file',
]

_MILLION = 1_000_000
_AGREE = 0.05
STALE_AFTER = timedelta(days=1)
_TIMEOUT_S = 30


@dataclass(frozen=True, slots=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float
    cached_input: float | None = None


@dataclass(frozen=True, slots=True)
class Quote:
    """The price used, where it came from, and whether the sources agreed."""

    price: Price
    origin: str
    agreed: bool
    sources: Mapping[str, Price]


def _get(url: str) -> object:
    request = urllib.request.Request(url, headers={'User-Agent': 'ai-lab'})  # noqa: S310 -- fixed https catalogs
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
        return json.loads(response.read())


def _per_token(value: object) -> float | None:
    return None if value in (None, '') else float(value) * _MILLION


def _models_dev(vendor: str, model: str, get: Callable[[str], object]) -> Price | None:
    entry = get('https://models.dev/api.json').get(vendor, {}).get('models', {}).get(model, {}).get('cost')
    if not entry or 'input' not in entry:
        return None
    return Price(float(entry['input']), float(entry.get('output', 0.0)), entry.get('cache_read'))


def _litellm(vendor: str, model: str, get: Callable[[str], object]) -> Price | None:
    table = get('https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json')
    entry = table.get(f'{vendor}/{model}') or (
        table.get(model) if table.get(model, {}).get('litellm_provider') == vendor else None
    )
    if not entry or entry.get('input_cost_per_token') is None:
        return None
    return Price(
        _per_token(entry['input_cost_per_token']),
        _per_token(entry.get('output_cost_per_token')) or 0.0,
        _per_token(entry.get('cache_read_input_token_cost')),
    )


def _portkey(vendor: str, model: str, get: Callable[[str], object]) -> Price | None:
    try:
        entry = get(f'https://api.portkey.ai/model-configs/pricing/{vendor}/{model}').get('pay_as_you_go', {})
    except OSError:
        return None
    if 'request_token' not in entry:
        return None
    cents = 1e-2 * _MILLION  # Portkey quotes US cents per token
    cached = entry.get('cache_read_input_token', {}).get('price')
    return Price(
        entry['request_token']['price'] * cents,
        entry.get('response_token', {}).get('price', 0.0) * cents,
        None if cached is None else cached * cents,
    )


def _openrouter(vendor: str, model: str, get: Callable[[str], object]) -> Price | None:
    if vendor != 'openrouter':
        return None
    for entry in get('https://openrouter.ai/api/v1/models').get('data', []):
        if entry.get('id') == model:
            pricing = entry['pricing']
            return Price(
                _per_token(pricing['prompt']),
                _per_token(pricing['completion']),
                _per_token(pricing.get('input_cache_read')),
            )
    return None


#: Every catalog, by name. Each answers only for the vendor it can speak for (OpenRouter: itself).
SOURCES: Mapping[str, Callable[[str, str, Callable[[str], object]], Price | None]] = {
    'models.dev': _models_dev,
    'litellm': _litellm,
    'portkey': _portkey,
    'openrouter': _openrouter,
}


def snapshot_dir() -> Path:
    """Where dated catalog snapshots are kept."""
    return user_cache_path('ai_lab') / 'prices'


def user_prices_file() -> Path:
    """The user's contract rates (it need not exist)."""
    return config_root('ai_lab') / 'prices.toml'


def _stamp(at: datetime) -> str:
    return at.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')


def _snapshots(directory: Path) -> list[tuple[datetime, dict]]:
    if not directory.is_dir():
        return []
    return [
        (datetime.strptime(path.stem, '%Y%m%dT%H%M%SZ').replace(tzinfo=UTC), json.loads(path.read_text('utf-8')))
        for path in sorted(directory.glob('*.json'))
    ]


def _fetch(vendor: str, model: str, get: Callable[[str], object]) -> dict[str, dict]:
    cache: dict[str, object] = {}

    def cached_get(url: str) -> object:
        if url not in cache:
            cache[url] = get(url)
        return cache[url]

    found = {}
    for name, source in SOURCES.items():
        try:
            price = source(vendor, model, cached_get)
        except (OSError, ValueError, KeyError, TypeError):
            price = None
        if price is not None:
            found[name] = asdict(price)
    return found


def _consensus(sources: Mapping[str, Price]) -> tuple[Price, bool]:
    quoted = [asdict(p) for p in sources.values()]

    def middle(field: str) -> float | None:
        values = [q[field] for q in quoted if q[field] is not None]
        return statistics.median(values) if values else None

    median = {field: middle(field) for field in ('input', 'output', 'cached_input')}
    agreed = all(abs(q[f] - median[f]) <= _AGREE * max(median[f], 1e-12) for q in quoted for f in ('input', 'output'))
    return Price(median['input'], median['output'] or 0.0, median['cached_input']), agreed


def quote(
    provider: str,
    vendor: str,
    model: str,
    *,
    at: datetime | None = None,
    directory: Path | None = None,
    get: Callable[[str], object] = _get,
    user: Mapping[str, Mapping[str, float]] | None = None,
) -> Quote | None:
    """The price of ``model`` billed by ``vendor`` at ``at`` (now when ``None``), or ``None`` when unpriced.

    The user's ``prices.toml`` row for ``provider:model`` wins. Otherwise the newest snapshot at or
    before ``at`` holding the model is used; asked about NOW with no such snapshot younger than a day,
    the catalogs are read once and a new snapshot is written.
    """
    user = _user_prices() if user is None else user
    row = user.get(f'{provider}:{model}')
    if row is not None:
        price = Price(row['input'], row['output'], row.get('cached_input'))
        return Quote(price, 'prices.toml', agreed=True, sources={'prices.toml': price})
    if not vendor:
        return None
    directory = snapshot_dir() if directory is None else directory
    key = f'{vendor}/{model}'
    now = datetime.now(UTC)
    when = now if at is None else at
    held = [(t, s['quotes'][key]) for t, s in _snapshots(directory) if key in s.get('quotes', {}) and t <= when]
    if at is None and (not held or now - held[-1][0] > STALE_AFTER):
        fetched = _fetch(vendor, model, get)
        if fetched:
            directory.mkdir(parents=True, exist_ok=True)
            payload = {'fetched_at': now.isoformat(), 'quotes': {key: fetched}}
            (directory / f'{_stamp(now)}.json').write_text(json.dumps(payload, indent=1), 'utf-8')
            held.append((now, fetched))
    if not held:
        later = [(t, s['quotes'][key]) for t, s in _snapshots(directory) if key in s.get('quotes', {})]
        if not later:
            return None
        held = later[:1]
    sources = {name: Price(**p) for name, p in held[-1][1].items()}
    price, agreed = _consensus(sources)
    return Quote(price, f'catalogs@{_stamp(held[-1][0])}', agreed, sources)


def _user_prices() -> dict[str, Mapping[str, float]]:
    path = user_prices_file()
    return tomllib.loads(path.read_text('utf-8')) if path.is_file() else {}


def cost(usage: Usage | None, price: Price | None) -> float | None:
    """US dollars for ``usage`` at ``price``; the CLI's own reported dollars win; ``None`` when unpriced."""
    if usage is None:
        return None
    if usage.cost_usd is not None:
        return usage.cost_usd
    if price is None:
        return None
    fresh = usage.input_tokens - usage.cached_tokens
    cached_rate = price.input if price.cached_input is None else price.cached_input
    return (fresh * price.input + usage.cached_tokens * cached_rate + usage.output_tokens * price.output) / _MILLION


def quote_for(client: str, *, at: datetime | None = None, **options: object) -> Quote | None:
    """:func:`quote` for a client name (``provider:model[@effort]``) -- effort moves tokens, not the rate."""
    provider, _, model = client.partition('@')[0].partition(':')
    row = providers().get(provider)
    return quote(provider, '' if row is None else row.vendor, model, at=at, **options)


def main(argv: list[str] | None = None) -> int:
    """``python -m ai_lab.prices CLIENT [...]`` -- each catalog's quote, the consensus, and agreement."""
    import sys  # noqa: PLC0415 -- the command line only

    for client in sys.argv[1:] if argv is None else argv:
        held = quote_for(client)
        if held is None:
            sys.stdout.write(f'{client}: unpriced (no vendor, or no catalog lists it)\n')
            continue
        verdict = 'agreed' if held.agreed else 'DISPUTED'
        sys.stdout.write(f'{client}: {held.price} from {held.origin}, {verdict}\n')
        for name, price in held.sources.items():
            sys.stdout.write(f'    {name}: {price}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
