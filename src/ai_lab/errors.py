"""The three ways a call cannot produce what was asked, each naming its remedy."""

__all__ = ['ProviderError', 'ReplayMiss', 'Unsupported']


class Unsupported(Exception):
    """The provider lacks a capability the request needs (a verb, images). Pick a provider that has it."""


class ProviderError(Exception):
    """The provider was reached and failed, or returned something that is not the declared shape."""


class ReplayMiss(Exception):
    """A replay ledger holds no entry for this exact request. Re-record it with mode='record'."""
