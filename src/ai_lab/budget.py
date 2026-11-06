"""Spend caps, scoped by PROVIDER and by API KEY: MONEY per calendar month (UTC), never a number of tokens.

The ONE declaration of what a provider may spend is the ``[ai_lab.spend_cap]`` table of the per-machine file
(:mod:`lab_commons.config`); every other reader (a campaign planner's projection, a run's token allotment)
derives from :func:`cap` / :func:`left_usd` and never restates the number::

    [ai_lab.spend_cap]
    openai = { amount = 50, currency = "USD" }              # one provider, every key of it
    deepseek = { amount = 100, currency = "CNY" }           # billed in yuan: needs [ai_lab.usd_per] CNY
    "anthropic/3f2a9c1b7d4e" = { amount = 10, currency = "USD" }  # one key: <provider>/<fingerprint>
    gemini = { amount = inf, currency = "USD" }             # declared uncapped

    [ai_lab.usd_per]
    CNY = 0.14                                              # US dollars per unit; USD is 1 by definition

NO CODE DEFAULT. A paid (keyed) provider with no declared cap is REFUSED at its first call, naming the row to
declare: a code default silently capped a 50 USD OpenAI account at ~2 USD of tokens (2026-11-06), and a cap
nobody declared is a cap nobody chose. ``amount = inf`` is the honest spelling of "no cap". A keyless provider
(a local server) pays nothing and needs no row.

TOKENS ARE DERIVED. A call is priced by :mod:`ai_lab.prices` at its time; a call that would START with a scope
spent raises :class:`~ai_lab.errors.SpendCapReached` and is never sent, and its ``max_tokens`` is clamped to what
the tightest scope has left at the model's output rate (a call under way may finish past the cap by its input).
A capped call whose model has no price is refused: an unpriced call cannot be held to a money cap.

THE PERIOD is the calendar month (UTC): providers bill and budget monthly, and so does :mod:`ai_lab.spend`.

The ledger is one JSON line per call ``{"provider", "key", "tokens", "model", "input", "output", "cached",
"at", "tag"}`` (``tag``: what the call is charged to, :data:`TAG_ENV` with the call's own tag appended
(``owner/repo#5149/diff``), else whichever of the two is set, else ``null``), appended to
``tokens-<UTC date>.jsonl`` under :func:`state_dir` -- machine-local, never tracked, so every checkout of one box
draws on one cap. Dollars are DERIVED from it (:func:`row_usd`), never stored. A key is named by its FINGERPRINT
(:func:`fingerprint`; ``python -m ai_lab.auth status`` prints it), never by its secret. The ledger is a fast
LOCAL stop: a key held on several hosts is metered per host here, and the key's global truth is its provider's
own usage report (user ruling 2026-10-24, option A).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from lab_commons.config import config_path, section
from platformdirs import user_state_path

from ai_lab import retired
from ai_lab.errors import SpendCapReached, Unsupported
from ai_lab.usage import Usage

if TYPE_CHECKING:
    from ai_lab.prices import Quote

__all__ = [
    'TAG_ENV',
    'Cap',
    'cap',
    'charge',
    'fingerprint',
    'key_scope',
    'ledger_file',
    'left_usd',
    'rate',
    'row_usd',
    'spend_tag',
    'spent_usd',
    'state_dir',
    'usd_per',
]

APP: Final = 'ai_lab'
#: The environment variable naming what a call is charged to (a PR review sets ``<owner>/<repo>#<number>``).
TAG_ENV: Final = 'AI_LAB_SPEND_TAG'
TABLE: Final = 'spend_cap'
RATES: Final = 'usd_per'
#: The currency every price is quoted in.
USD: Final = 'USD'

#: Hex digits of a key's sha256: enough to tell keys apart, never enough to be one.
FINGERPRINT_CHARS: Final = 12

Price = Callable[[str, 'datetime | None'], 'Quote | None']


@dataclass(frozen=True, slots=True)
class Cap:
    """What one scope may spend per calendar month: *amount* of *currency* (``inf``: declared uncapped)."""

    amount: float
    currency: str


def fingerprint(secret: str) -> str:
    """A key's identity everywhere it is named: a sha256 prefix, never the key."""
    return hashlib.sha256(secret.encode()).hexdigest()[:FINGERPRINT_CHARS]


def key_scope(provider: str, key: str) -> str:
    """The table key of one API key: ``<provider>/<fingerprint>``."""
    return f'{provider}/{key}'


def rate(client: str, at: datetime | None = None) -> Quote | None:
    """``provider:model``'s rate at *at* (the current one when no snapshot that old holds it); ``None``: unpriced."""
    from ai_lab.prices import quote_for  # noqa: PLC0415 -- prices reads the catalogue, which imports this module

    return quote_for(client, at=at) or quote_for(client)


def _table() -> dict:
    own = section(APP)
    for name in own:
        retired.refuse('table', name, f'[{APP}.{name}] in {config_path()}')
    return own


def _caps() -> dict[str, Cap]:
    table = _table().get(TABLE, {})
    if not isinstance(table, dict):
        msg = f'[{APP}.{TABLE}] is a table of scope = {{ amount, currency }}, got {table!r}'
        raise ValueError(msg)  # noqa: TRY004 -- a malformed machine file is a value error, not a type error
    caps = {}
    for scope, row in table.items():
        amount = row.get('amount') if isinstance(row, dict) else None
        currency = row.get('currency') if isinstance(row, dict) else None
        if (
            not isinstance(amount, int | float) or isinstance(amount, bool) or math.isnan(amount) or amount < 0
            or not isinstance(currency, str) or set(row) != {'amount', 'currency'}
        ):  # fmt: skip
            msg = (
                f'[{APP}.{TABLE}] {scope} must be {{ amount = <non-negative number or inf>, currency = "<ISO code>" }}'
                f', got {row!r}'
            )
            raise ValueError(msg)
        caps[scope] = Cap(amount, currency)
    return caps


def usd_per(currency: str) -> float:
    """US dollars per unit of *currency*: 1 for USD, else ``[ai_lab.usd_per]``; an undeclared one is refused."""
    if currency == USD:
        return 1.0
    rate = _table().get(RATES, {}).get(currency)
    if not isinstance(rate, int | float) or isinstance(rate, bool) or not rate > 0:
        msg = f'{currency} has no positive rate: declare [{APP}.{RATES}] {currency} = <USD per unit> in {config_path()}'
        raise Unsupported(msg)
    return float(rate)


def _paid(provider: str) -> bool:
    from ai_lab.client import catalogue  # noqa: PLC0415 -- the client imports this module

    row = catalogue().get(provider)
    return row is None or row.keyed


def cap(provider: str) -> Cap | None:
    """*provider*'s declared cap; ``None`` for a keyless provider. A paid one with no row is refused by name."""
    caps = _caps()
    if provider in caps:
        return caps[provider]
    if not _paid(provider):
        return None
    msg = (
        f'{provider} has no spend cap: declare [{APP}.{TABLE}] {provider} = {{ amount = <per month>, '
        f'currency = "USD" }} in {config_path()} (amount = inf for none); no cap is assumed'
    )
    raise Unsupported(msg)


def state_dir() -> Path:
    """``$AI_LAB_HOME/state`` when set, else the platform's per-user local state directory."""
    home = os.environ.get('AI_LAB_HOME')
    return Path(home).expanduser() / 'state' if home else user_state_path(APP, appauthor=False)


def ledger_file() -> Path:
    """Today's (UTC) token ledger."""
    return state_dir() / f'tokens-{datetime.now(UTC).date().isoformat()}.jsonl'


def row_usd(row: dict, price: Price | None = None) -> float | None:
    """One ledger row's US dollars at its call's time; ``None`` when it names no model or the model is unpriced."""
    from ai_lab.prices import cost  # noqa: PLC0415 -- prices reads the catalogue, which imports this module

    if not row.get('model'):
        return None
    at = datetime.fromisoformat(row['at']) if row.get('at') else None
    held = (rate if price is None else price)(f'{row["provider"]}:{row["model"]}', at)
    usage = Usage(row.get('input', 0), row.get('output', 0), row.get('cached', 0))
    return cost(usage, None if held is None else held.price)


def _month_rows(provider: str, key: str | None) -> list[dict]:
    start = datetime.now(UTC).date().replace(day=1)
    rows = []
    for path in sorted(state_dir().glob('tokens-*.jsonl')):
        try:
            day = date.fromisoformat(path.stem.removeprefix('tokens-'))
        except ValueError:
            continue
        if day >= start:
            rows += [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]
    return [r for r in rows if r['provider'] == provider and (key is None or r['key'] == key)]


def spent_usd(provider: str, key: str | None = None) -> float:
    """US dollars this month's calls to *provider* (through *key*, when given) cost; an unpriced call is refused."""
    total = 0.0
    for row in _month_rows(provider, key):
        usd = row_usd(row)
        if usd is None:
            msg = f'{provider}:{row.get("model")} at {row.get("at")} has no price; declare it in ai_lab prices.toml'
            raise Unsupported(msg)
        total += usd
    return total


def left_usd(provider: str, key: str | None = None) -> float:
    """US dollars left this month to a call on *provider* through *key* (``inf``: uncapped).

    :class:`SpendCapReached` naming the scope when one is spent; :class:`Unsupported` for an undeclared cap.
    """
    caps = _caps()
    scopes = [(provider, cap(provider), None)]
    if key is not None and (scope := key_scope(provider, key)) in caps:
        scopes.append((scope, caps[scope], key))
    left = [math.inf]
    for scope, held, by_key in scopes:
        if held is None or math.isinf(held.amount):
            continue
        limit = held.amount * usd_per(held.currency)
        spent = spent_usd(provider, by_key)
        if spent >= limit:
            msg = (
                f'{scope} has spent {spent:.4f} of its {limit:.4f} USD this month (UTC): its spend cap is reached; '
                f'no call starts there until next month, or raise [{APP}.{TABLE}] {scope!r} in {config_path()}'
            )
            raise SpendCapReached(msg, scope=scope)
        left.append(limit - spent)
    return min(left)


def spend_tag(tag: str | None = None) -> str | None:
    """What a call is charged to: :data:`TAG_ENV`, ``/``-joined with the call's own *tag* (a review profile)."""
    return '/'.join(t for t in (os.environ.get(TAG_ENV), tag) if t) or None


def charge(provider: str, key: str | None, usage: Usage, model: str, *, tag: str | None = None) -> None:
    """Append one call's *usage* of *model* on *provider* through *key* to today's ledger, tagged :func:`spend_tag`."""
    row = {
        'provider': provider, 'key': key, 'tokens': usage.input_tokens + usage.output_tokens, 'model': model,
        'input': usage.input_tokens, 'output': usage.output_tokens, 'cached': usage.cached_tokens,
        'at': datetime.now(UTC).isoformat(timespec='seconds'), 'tag': spend_tag(tag),
    }  # fmt: skip
    path = ledger_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps(row) + '\n')
