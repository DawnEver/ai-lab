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

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from lab_commons.config import config_path, section
from platformdirs import user_state_path

from ai_lab import retired
from ai_lab.errors import SpendCapReached, Unsupported
from ai_lab.usage import Usage

if TYPE_CHECKING:
    from ai_lab.prices import Quote

__all__ = [
    'SCOPE_ENV',
    'SEP',
    'TAG_ENV',
    'Allotment',
    'Approval',
    'Cap',
    'allot',
    'approvals',
    'approve',
    'cap',
    'charge',
    'close',
    'current',
    'fingerprint',
    'key_scope',
    'ledger_file',
    'left_usd',
    'rate',
    'revoke',
    'room',
    'row_usd',
    'spend_tag',
    'spent_usd',
    'standing',
    'state_dir',
    'tasks',
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


def _rows(provider: str, key: str | None, *, month: bool = True) -> list[dict]:
    """*provider*'s ledger rows (through *key*, when given): this month's, else every day's."""
    start = datetime.now(UTC).date().replace(day=1) if month else date.min
    rows = []
    for path in sorted(state_dir().glob('tokens-*.jsonl')):
        try:
            day = date.fromisoformat(path.stem.removeprefix('tokens-'))
        except ValueError:
            continue
        if day >= start:
            rows += [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]
    return [r for r in rows if r['provider'] == provider and (key is None or r['key'] == key)]


def _priced(row: dict) -> float:
    usd = row_usd(row)
    if usd is None:
        msg = f'{row["provider"]}:{row.get("model")} at {row.get("at")} has no price; declare it in ai_lab prices.toml'
        raise Unsupported(msg)
    return usd


def spent_usd(provider: str, key: str | None = None) -> float:
    """US dollars this month's calls to *provider* (through *key*, when given) cost; an unpriced call is refused."""
    return sum(_priced(row) for row in _rows(provider, key))


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
        'at': _now(), 'tag': spend_tag(tag), 'scope': current(),
    }  # fmt: skip
    path = ledger_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps(row) + '\n')


# --- Allotments: the levels below a provider's cap ---------------------------------------------------------------
#
# LEVEL 1 is the provider's month (:func:`cap`). LEVEL 2 is an ALLOTMENT carved out of it: a STANDING one, a named
# recurring USD amount per calendar month declared ONCE in ``[ai_lab.standing]`` of the machine file (the PR
# review, a CI job), or a TASK one a caller opens for one task (a campaign, an optimisation run, a diagnosis) and
# closes when done. LEVEL 3 is a child of either (an item, an arm, one PR), opened under it by path
# (``campaign:T3>item:4``, ``review>owner/repo#5``) or derived by :meth:`Allotment.split`; a call's own level is
# its ``max_tokens``, derived from what the tightest enclosing level leaves.
#
# WHAT IS CARVED: the provider's month left, minus what every standing and open task allotment still holds, is
# the room of an unscoped call or a new task; so a campaign can never starve the review, nor the review a
# campaign. A child is carved out of its parent the same way. An opening that does not fit is REFUSED at open,
# naming the level it does not fit in; a call that finds any enclosing level spent is refused naming that level.
#
# WHERE EACH FACT LIVES. A standing amount: the machine file. A task's amount: its ``open`` row in
# ``allotments.jsonl`` under :func:`state_dir` (a later ``open`` of the same path re-allots it; ``close`` releases
# it and its children). What any level SPENT: the token ledger, each row carrying the path of the scope it was
# charged in (:data:`SCOPE_ENV`, which a child process inherits), priced like every other row, never stored. A
# scope reopened by name in another process (an HPC item) reads the same amount and spent -- on the host whose
# ledger holds the calls (the ledger is machine-local, like the cap).
#
# A TASK'S AMOUNT IS A DEFAULT (user ruling 2026-11-06): what a caller passes to ``allot`` is the budget its task
# DECLARES; a person may APPROVE one task, by its exact name, for a different amount (:func:`approve`: who, when,
# why, in the same ``allotments.jsonl``), and ``allot`` then opens it at the approved amount, recording the
# default beside it. An approval that does not fit where the task would open is refused at approve, naming the
# level; :func:`revoke` restores the default.

#: The environment variable naming the scope a call is charged in: the cross-process contract.
SCOPE_ENV: Final = 'AI_LAB_SCOPE'
#: Separates a scope path's levels: ``campaign:T3>item:4``.
SEP: Final = '>'
STANDING: Final = 'standing'


@dataclass(frozen=True, slots=True)
class Allotment:
    """One scope: its *name* (a ``>`` path), the *provider* it draws on, its *usd*, and whether it is *standing*."""

    name: str
    provider: str
    usd: float
    standing: bool = False
    #: The caller's declared default when an approval replaced it; ``None``: the allotment IS the default.
    default_usd: float | None = None

    @property
    def parent(self) -> str | None:
        """The enclosing scope's path; ``None`` at level 2."""
        return self.name.rpartition(SEP)[0] or None

    @property
    def root(self) -> str:
        """The level-2 allotment this scope is carved from."""
        return self.name.split(SEP, 1)[0]

    def spent(self) -> float:
        """US dollars the ledger charged in this scope or below it (this month, under a standing allotment)."""
        return _scope_spent(self.provider, self.name, month=self.root in standing())

    def left(self) -> float:
        """What this scope has left to spend, never below zero."""
        return max(0.0, self.usd - self.spent())

    def allot(self, name: str, usd: float) -> Allotment:
        """Open the child *name* of this scope with *usd* (refused at open when this scope cannot hold it)."""
        return allot(f'{self.name}{SEP}{name}', usd=usd)

    def split(self, names: list[str]) -> list[Allotment]:
        """One child per *names*, each an equal share of what this scope leaves unheld: derived, never copied."""
        share = _free_in(self) / len(names)
        return [self.allot(name, share) for name in names]

    def __enter__(self) -> Self:
        """Charge every call started inside (here, and in child processes through :data:`SCOPE_ENV`) to this scope."""
        _stack.append(os.environ.get(SCOPE_ENV))
        os.environ[SCOPE_ENV] = self.name
        return self

    def __exit__(self, *_: object) -> None:
        """Restore the enclosing scope; the allotment stays open (:func:`close` releases it)."""
        before = _stack.pop()
        if before is None:
            os.environ.pop(SCOPE_ENV, None)
        else:
            os.environ[SCOPE_ENV] = before


_stack: list[str | None] = []


@dataclass(frozen=True, slots=True)
class Approval:
    """*by* approved the task *name* on *provider* for *amount* of *currency* at *at*, because *reason*."""

    name: str
    provider: str
    amount: float
    currency: str
    by: str
    reason: str
    at: str

    @property
    def usd(self) -> float:
        """The approved amount in US dollars, at the currently declared rate."""
        return self.amount * usd_per(self.currency)


def _allotments_file() -> Path:
    return state_dir() / 'allotments.jsonl'


def standing() -> dict[str, Allotment]:
    """Every ``[ai_lab.standing]`` allotment; refused when one provider's standing amounts overrun its cap."""
    table = _table().get(STANDING, {})
    found: dict[str, Allotment] = {}
    for name, row in table.items():
        if not isinstance(row, dict) or set(row) != {'provider', 'amount', 'currency'} or SEP in name:
            msg = f'[{APP}.{STANDING}] {name} must be {{ provider, amount, currency }} with no {SEP!r}, got {row!r}'
            raise ValueError(msg)
        found[name] = Allotment(name, row['provider'], row['amount'] * usd_per(row['currency']), standing=True)
    for provider in {a.provider for a in found.values()}:
        held = cap(provider)
        held_usd = math.inf if held is None else held.amount * usd_per(held.currency)
        total = sum(a.usd for a in found.values() if a.provider == provider)
        if total > held_usd:
            msg = (
                f'[{APP}.{STANDING}] allots {total:.4f} USD a month on {provider}, past its {held_usd:.4f} USD '
                f'[{APP}.{TABLE}] in {config_path()}: lower a standing amount or raise the cap'
            )
            raise ValueError(msg)
    return found


def _store() -> list[dict]:
    path = _allotments_file()
    return [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()] if path.exists() else []


def _append(row: dict) -> None:
    path = _allotments_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps(row) + '\n')


def tasks() -> dict[str, Allotment]:
    """Every open task allotment by path: a path's newest ``open``, unless a later ``close`` released it."""
    found: dict[str, Allotment] = {}
    for row in _store():
        if row['op'] in {'approve', 'revoke'}:
            continue
        if row['op'] == 'open':
            found[row['name']] = Allotment(row['name'], row['provider'], row['usd'], default_usd=row.get('default'))
        else:
            for name in [n for n in found if n == row['name'] or n.startswith(row['name'] + SEP)]:
                del found[name]
    return found


def approvals() -> dict[str, Approval]:
    """Every approval in force by task name: a name's newest ``approve``, unless a later ``revoke`` withdrew it."""
    found: dict[str, Approval] = {}
    for row in _store():
        if row['op'] == 'approve':
            found[row['name']] = Approval(
                row['name'], row['provider'], row['amount'], row['currency'], row['by'], row['reason'], row['at']
            )
        elif row['op'] == 'revoke':
            found.pop(row['name'], None)
    return found


def _all() -> dict[str, Allotment]:
    return {**tasks(), **standing()}


def _scope_spent(provider: str, name: str, *, month: bool) -> float:
    total = 0.0
    for row in _rows(provider, None, month=month):
        scope = row.get('scope') or ''
        if scope == name or scope.startswith(name + SEP):
            total += _priced(row)
    return total


def _held(children: list[Allotment], but: str | None) -> float:
    return sum(c.left() for c in children if c.name != but)


def _children(parent: str | None, provider: str) -> list[Allotment]:
    return [a for a in _all().values() if a.parent == parent and a.provider == provider]


def _free_in(parent: Allotment, but: str | None = None) -> float:
    """What *parent* leaves that none of its open children holds."""
    return parent.left() - _held(_children(parent.name, parent.provider), but)


def _free_on(provider: str, key: str | None = None, but: str | None = None) -> float:
    """The provider's month left that no standing or open task allotment holds."""
    return left_usd(provider, key) - _held(_children(None, provider), but)


def _refused(scope: str, what: str) -> SpendCapReached:
    return SpendCapReached(f'{scope}: {what}', scope=scope)


def allot(name: str, *, usd: float | None = None, provider: str | None = None) -> Allotment:
    """Open the task allotment *name* with *usd* (a level-2 one names its *provider*), or REOPEN it by name.

    Without *usd* the allotment is the one recorded (a standing one, or a task another process opened); an
    unknown name is refused. With *usd* it is opened or re-allotted, and refused with :class:`SpendCapReached`
    naming the level it does not fit in: its parent (what it leaves beside its other children), else its
    provider (the month left beside every standing and open task allotment).
    """
    known = _all()
    if usd is None:
        if name not in known:
            msg = f'no allotment {name!r} is open: open it with usd= (or declare [{APP}.{STANDING}] {name})'
            raise Unsupported(msg)
        return known[name]
    if name in standing():
        msg = f'{name} is a [{APP}.{STANDING}] allotment, declared in {config_path()}; it is never opened as a task'
        raise Unsupported(msg)
    provider, free, level = _fit(name, provider, known)
    approved = approvals().get(name)
    default = None
    if approved is not None and approved.provider == provider:
        default, usd = float(usd), approved.usd
    opened = Allotment(name, provider, float(usd), default_usd=default)
    spent = opened.spent()
    if usd - spent > free + 1e-12 * max(1.0, usd):  # relative: an exact split fits
        raise _refused(level, f'cannot hold {name} at {usd:.4f} USD ({spent:.4f} spent): {free:.4f} USD is free')
    _append({'op': 'open', 'name': name, 'provider': provider, 'usd': float(usd), 'default': default, 'at': _now()})
    return opened


def _fit(name: str, provider: str | None, known: dict[str, Allotment]) -> tuple[str, float, str]:
    """Where *name* opens: its provider, the USD free for it there, and the level that bounds it."""
    parent_name = name.rpartition(SEP)[0] or None
    if parent_name is not None:
        if parent_name not in known:
            msg = f'{name}: its parent {parent_name!r} is not open'
            raise Unsupported(msg)
        parent = known[parent_name]
        if provider not in {None, parent.provider}:
            msg = f"{name} draws on its parent {parent_name}'s provider {parent.provider}, not {provider}"
            raise Unsupported(msg)
        provider, free, level = parent.provider, _free_in(parent, but=name), parent_name
    else:
        if provider is None:
            msg = f'{name} opens a task at level 2: name the provider it draws on'
            raise Unsupported(msg)
        free, level = _free_on(provider, but=name), provider
    return provider, free, level


def approve(task: str, amount: float, currency: str = USD, *, provider: str, by: str, reason: str) -> Approval:
    """Approve the task *task* on *provider* for *amount* of *currency* in place of its declared default.

    Refused with :class:`SpendCapReached` naming the level when the amount does not fit where *task* opens (its
    parent, else the provider month beside every standing and other open task allotment); a standing name and
    an approval without a reason are refused.
    """
    if task in standing():
        msg = f'{task} is a [{APP}.{STANDING}] allotment, declared in {config_path()}; it is never approved as a task'
        raise Unsupported(msg)
    if not reason.strip():
        msg = f'approving {task} needs a reason'
        raise Unsupported(msg)
    approval = Approval(task, provider, float(amount), currency, by, reason, _now())
    _, free, level = _fit(task, provider, _all())
    usd, spent = approval.usd, Allotment(task, provider, approval.usd).spent()
    if usd - spent > free + 1e-12 * max(1.0, usd):
        raise _refused(level, f'cannot approve {task} at {usd:.4f} USD ({spent:.4f} spent): {free:.4f} USD is free')
    _append({
        'op': 'approve', 'name': task, 'provider': provider, 'amount': approval.amount, 'currency': currency,
        'by': by, 'reason': reason, 'at': approval.at,
    })  # fmt: skip
    return approval


def revoke(task: str) -> None:
    """Withdraw *task*'s approval: its next ``allot`` opens at the caller's default again."""
    _append({'op': 'revoke', 'name': task, 'at': _now()})


def close(name: str) -> None:
    """Release the task allotment *name* and every child of it: what it held returns to its parent."""
    _append({'op': 'close', 'name': name, 'at': _now()})


def current() -> str | None:
    """The scope a call started here is charged in (:data:`SCOPE_ENV`); ``None``: unscoped."""
    return os.environ.get(SCOPE_ENV) or None


def room(provider: str, key: str | None = None) -> float:
    """US dollars a call on *provider* through *key* may spend: the TIGHTEST enclosing level's left.

    :class:`SpendCapReached` names the level with no room -- the provider (or key), then each scope of the current
    path from the outermost down. An unscoped call has only what no allotment holds.
    """
    scope = current()
    if scope is None:
        free = _free_on(provider, key)
        if free <= 0:
            raise _refused(provider, f'every dollar its month leaves is held by an allotment ({free:.4f} USD free)')
        return free
    known = _all()
    if scope not in known:
        msg = f'{SCOPE_ENV}={scope} names no open allotment'
        raise Unsupported(msg)
    if known[scope].provider != provider:
        msg = f'{scope} is allotted on {known[scope].provider}; a call on {provider} cannot be charged to it'
        raise Unsupported(msg)
    left = [left_usd(provider, key)]
    if left[0] <= 0:
        raise _refused(provider, 'its month is spent')
    parts = scope.split(SEP)
    for level in (SEP.join(parts[: i + 1]) for i in range(len(parts))):
        held = known[level]
        if (spare := held.left()) <= 0:
            raise _refused(level, f'spent {held.spent():.4f} of its {held.usd:.4f} USD allotment')
        left.append(spare)
    return min(left)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


def main(argv: Sequence[str] | None = None) -> int:
    """``allot NAME [--usd N] [--provider P]`` prints the allotment as JSON; ``close NAME`` releases it.

    ``approve NAME --amount A [--currency C] --provider P --reason R [--by WHO]``, ``revoke NAME`` and
    ``approvals`` (JSON) manage the approvals that replace a task's declared default.

    The door for a caller that imports no ai-lab (dev-lab's review opens ``review>owner/repo#n`` here, then runs
    its commands with :data:`SCOPE_ENV`). A refusal exits 3 with its message and the level on stderr.
    """
    parser = argparse.ArgumentParser(prog='python -m ai_lab.budget', description='open, reopen or close an allotment')
    verbs = parser.add_subparsers(dest='verb', required=True)
    opened = verbs.add_parser('allot')
    opened.add_argument('name')
    opened.add_argument('--usd', type=float)
    opened.add_argument('--provider')
    verbs.add_parser('close').add_argument('name')
    granted = verbs.add_parser('approve', help='approve one task for an amount in place of its declared default')
    granted.add_argument('name')
    granted.add_argument('--amount', type=float, required=True)
    granted.add_argument('--currency', default=USD)
    granted.add_argument('--provider', required=True)
    granted.add_argument('--by', default=os.environ.get('USERNAME') or os.environ.get('USER') or 'unknown')
    granted.add_argument('--reason', required=True)
    verbs.add_parser('revoke').add_argument('name')
    verbs.add_parser('approvals')
    args = parser.parse_args(argv)
    if args.verb == 'close':
        close(args.name)
        return 0
    if args.verb == 'revoke':
        revoke(args.name)
        return 0
    if args.verb == 'approvals':
        listed = {
            n: {'provider': a.provider, 'amount': a.amount, 'currency': a.currency, 'usd': a.usd, 'by': a.by,
                'reason': a.reason, 'at': a.at}
            for n, a in approvals().items()
        }  # fmt: skip
        sys.stdout.write(json.dumps(listed) + '\n')
        return 0
    try:
        if args.verb == 'approve':
            done = approve(
                args.name, args.amount, args.currency, provider=args.provider, by=args.by, reason=args.reason
            )
            sys.stdout.write(json.dumps({'name': done.name, 'provider': done.provider, 'usd': done.usd}) + '\n')
            return 0
        held = allot(args.name, usd=args.usd, provider=args.provider)
    except (SpendCapReached, Unsupported) as refused:
        sys.stderr.write(f'{refused}\n')
        return 3
    doc = {'name': held.name, 'provider': held.provider, 'allotted_usd': held.usd, 'spent_usd': held.spent()}
    sys.stdout.write(json.dumps(doc) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
