"""The three ways a call cannot produce what was asked, each naming its remedy."""

__all__ = ['ProviderError', 'ReplayMiss', 'Unsupported']


class Unsupported(Exception):
    """The provider lacks a capability the request needs (a verb, images). Pick a provider that has it."""


class ProviderError(Exception):
    """The provider was reached and failed, or returned something that is not the declared shape.

    ``kind`` names an ACCOUNT-level failure -- ``'exhausted'`` (no balance), ``'auth'`` (401/403) or
    ``'rate_limit'`` (429 past its retries) -- that another key of the same account may not share;
    ``''`` is any other failure, which another key would only repeat.
    """

    def __init__(self, message: str, *, kind: str = '') -> None:
        """``message`` names what failed; ``kind`` classifies it (never the key)."""
        super().__init__(message)
        self.kind = kind


class ReplayMiss(Exception):
    """A replay ledger holds no entry for this exact request. Re-record it with mode='record'."""
