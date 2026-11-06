"""The four ways a call cannot produce what was asked, each naming its remedy."""

__all__ = [
    'ACCOUNT_KINDS', 'ERROR_KINDS', 'BatchLapsed', 'ProviderError', 'ReplayMiss', 'SpendCapReached',
    'Unsupported',
]  # fmt: skip

#: The failures another key of the same account may not share: a call moves to the next key slot on one.
ACCOUNT_KINDS = frozenset({'exhausted', 'auth', 'rate_limit'})
#: Every classified failure: the account kinds, and a ``timeout`` that another key would only repeat.
ERROR_KINDS = ACCOUNT_KINDS | {'timeout'}


class Unsupported(Exception):
    """The provider lacks a capability the request needs (a verb, images). Pick a provider that has it."""


class ProviderError(Exception):
    """The provider was reached and failed, or returned something that is not the declared shape.

    ``kind`` is one of :data:`ERROR_KINDS`: an ACCOUNT-level failure (:data:`ACCOUNT_KINDS`) --
    ``'exhausted'`` (no balance or quota), ``'auth'`` (a key refused) or ``'rate_limit'`` (429 or 529 past its
    retries) -- that another key of the same account may not share; ``'timeout'`` (no answer within the
    ceiling, past its retries); or ``''``, any other failure. Neither of the last two moves to another key.
    ``status`` is the HTTP status the failure ended on, ``None`` when there was none (a network failure, a body).
    """

    def __init__(self, message: str, *, kind: str = '', status: int | None = None) -> None:
        """``message`` names what failed; ``kind`` classifies it (never the key); ``status`` is its HTTP status."""
        if kind and kind not in ERROR_KINDS:
            msg = f'unknown kind {kind!r}; the kinds are {sorted(ERROR_KINDS)}'
            raise ValueError(msg)
        super().__init__(message)
        self.kind = kind
        self.status = status


class BatchLapsed(ProviderError):
    """A batched call that never reached the model: its batch ``expired`` (kind ``timeout``) or was ``canceled``.

    Unbilled; ``lapsed`` names which, so a caller may answer the same request another way.
    """

    def __init__(self, message: str, *, lapsed: str) -> None:
        """Carry ``lapsed`` (``'expired'`` or ``'canceled'``)."""
        super().__init__(message, kind='timeout' if lapsed == 'expired' else '')
        self.lapsed = lapsed


class ReplayMiss(Exception):
    """A replay ledger holds no entry for this exact request. Re-record it with mode='record'."""


class SpendCapReached(Exception):
    """A provider's or a key's spend this month reached its ``[ai_lab.spend_cap]``; no call starts there."""

    def __init__(self, message: str, *, scope: str) -> None:
        """Carry ``scope``, the provider or key whose cap was reached."""
        super().__init__(message)
        self.scope = scope
