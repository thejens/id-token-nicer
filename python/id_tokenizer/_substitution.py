"""Substitute UUIDs with short placeholders and restore them."""

import re

from ._registry import UUID_RE, MemoryRegistry, UuidRegistry

PLACEHOLDER_RE = re.compile(r"\$\{UUID_\d+\}")


def substitute(
    text: str,
    registry: UuidRegistry | None = None,
    *,
    factor: int = 97,
    mix: bool = True,
) -> tuple[str, UuidRegistry]:
    """Replace UUIDs in *text* with ``${UUID_N}`` placeholders.

    *factor* and *mix* are forwarded to :class:`MemoryRegistry` when no
    *registry* is supplied.  Use them to make the placeholder suffixes
    non-sequential (``mix``) and/or detectable when hallucinated
    (``factor``).
    """
    reg: UuidRegistry = (
        registry if registry is not None
        else MemoryRegistry(factor=factor, mix=mix)
    )

    def _replace(m: re.Match[str]) -> str:
        key = reg.store(m.group(0))
        return f"${{{key}}}"

    return UUID_RE.sub(_replace, text), reg


def restore(text: str, registry: UuidRegistry) -> str:
    """Replace ``${UUID_N}`` placeholders with original UUIDs.

    Unknown placeholders are left unchanged.
    """

    def _replace(m: re.Match[str]) -> str:
        key = m.group(0)[2:-1]  # strip ${ and }
        uuid = registry.fetch(key)
        return uuid if uuid is not None else m.group(0)

    return PLACEHOLDER_RE.sub(_replace, text)
