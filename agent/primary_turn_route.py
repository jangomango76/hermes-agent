"""Host-owned, per-user-turn routing for general Hermes plugins.

Plugins may return one typed ``pre_primary_turn_route`` decision before the
first primary request. They receive route facts, IDs, and user input, never
the live agent or credentials. Invalid decisions are ignored without changing
the active runtime.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_HOOK = "pre_primary_turn_route"
_ALLOWED_API_MODES = frozenset({"chat_completions", "anthropic_messages", "codex_responses", "bedrock_converse"})


@dataclass(frozen=True)
class _HostRiskVerdict:
    """Host-only classification; plugin output and prompt text are not authority."""

    risk: str
    verified: bool
    provenance: str


def _host_risk_verdict() -> _HostRiskVerdict | None:
    """No trusted risk producer is wired to this isolated host yet."""
    return None


def _safe_endpoint_origin(base_url: Any) -> str | None:
    """Expose only an HTTP(S) origin, never credentials, path, query or fragment."""
    if not isinstance(base_url, str):
        return None
    try:
        parsed = urlsplit(base_url)
        host, port = parsed.hostname, parsed.port
        if (parsed.scheme not in {"http", "https"} or not host
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or "?" in base_url or "#" in base_url
                or not host.isascii() or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789.-:" for c in host.lower())):
            return None
        # urlsplit accepts whitespace and control characters; those are not safe identities.
        if any(ord(c) <= 32 or ord(c) == 127 for c in base_url):
            return None
        hostname = f"[{host}]" if ":" in host else host
        return f"{parsed.scheme}://{hostname}{f':{port}' if port is not None else ''}"
    except ValueError:
        return None


@dataclass(frozen=True)
class PrimaryTurnRouteDecision:
    """A plugin's requested temporary primary route.

    ``model`` and ``provider`` are required. ``reasoning_effort`` is parsed by
    the host after the canonical model switch; credentials are deliberately
    not part of this extension contract.
    """

    model: str
    provider: str
    base_url: str = ""
    api_mode: str = ""
    reasoning_effort: Any = None


@dataclass(frozen=True)
class _PrimaryTurnRouteState:
    """Host-owned state needed to undo a temporary route after the turn."""

    primary_runtime: Mapping[str, Any]
    reasoning_config: Any
    cached_system_prompt: Any
    fallback_state: Mapping[str, Any]


_FALLBACK_STATE_FIELDS = (
    "_fallback_chain", "_fallback_model", "_fallback_index", "_fallback_activated",
    "_provider_fallback_active", "_provider_fallback_route", "_credential_pool_revert_id",
)


def _capture_route_state(agent: Any) -> _PrimaryTurnRouteState:
    return _PrimaryTurnRouteState(
        primary_runtime=copy.deepcopy(getattr(agent, "_primary_runtime", {}) or {}),
        reasoning_config=copy.deepcopy(getattr(agent, "reasoning_config", None)),
        cached_system_prompt=getattr(agent, "_cached_system_prompt", None),
        fallback_state={name: copy.deepcopy(getattr(agent, name, None)) for name in _FALLBACK_STATE_FIELDS},
    )


def _route_decision(value: Any) -> PrimaryTurnRouteDecision | None:
    """Validate an untrusted plugin return without coercing malformed values."""
    if not isinstance(value, Mapping):
        return None
    allowed = {"model", "provider", "base_url", "api_mode", "reasoning_effort"}
    if set(value) - allowed:
        return None
    model, provider = value.get("model"), value.get("provider")
    if not isinstance(model, str) or not model.strip() or not isinstance(provider, str) or not provider.strip():
        return None
    base_url, api_mode = value.get("base_url", ""), value.get("api_mode", "")
    if not isinstance(base_url, str) or not isinstance(api_mode, str):
        return None
    if api_mode and api_mode not in _ALLOWED_API_MODES:
        return None
    return PrimaryTurnRouteDecision(
        model=model.strip(), provider=provider.strip(), base_url=base_url.strip(), api_mode=api_mode,
        reasoning_effort=value.get("reasoning_effort"),
    )


def apply_primary_turn_route(agent: Any, *, user_message: Any, turn_id: str, task_id: str) -> bool:
    """Apply one valid plugin route before request construction, or leave runtime untouched."""
    # The host currently has no trusted risk producer. Do not even discover a
    # plugin without an explicitly verified host verdict and a safe identity.
    try:
        verdict = _host_risk_verdict()
    except Exception:
        logger.warning("Host primary-turn risk verdict unavailable", exc_info=True)
        return False
    if (type(verdict) is not _HostRiskVerdict or verdict.verified is not True
            or verdict.risk != "low" or not isinstance(verdict.provenance, str)
            or not verdict.provenance.strip()):
        return False
    endpoint_origin = _safe_endpoint_origin(getattr(agent, "base_url", None))
    if endpoint_origin is None:
        return False
    reasoning = getattr(agent, "reasoning_config", None)
    original_effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
    if original_effort is not None and not isinstance(original_effort, str):
        return False
    try:
        from hermes_cli.plugins import has_hook, invoke_hook
        if not has_hook(_HOOK):
            return False
        results = invoke_hook(
            _HOOK,
            turn_id=turn_id,
            task_id=task_id,
            session_id=getattr(agent, "session_id", "") or "",
            platform=getattr(agent, "platform", "") or "",
            user_message=user_message,
            current_route={
                "model": getattr(agent, "model", "") or "",
                "provider": getattr(agent, "provider", "") or "",
                "api_mode": getattr(agent, "api_mode", "") or "",
                "original_effort": original_effort,
                "endpoint_origin": endpoint_origin,
                "risk": {"risk": verdict.risk, "verified": True, "provenance": verdict.provenance},
            },
        )
    except Exception:
        logger.warning("pre-primary-turn route plugin dispatch failed", exc_info=True)
        return False
    decision = next((item for item in results if item is not None), None)
    parsed = _route_decision(decision)
    if parsed is None:
        if decision is not None:
            logger.warning("Ignoring invalid pre_primary_turn_route decision")
        return False
    effort = None
    if parsed.reasoning_effort is not None:
        from hermes_constants import parse_reasoning_effort
        effort = parse_reasoning_effort(parsed.reasoning_effort)
        if effort is None:
            logger.warning("Ignoring invalid pre_primary_turn_route reasoning_effort")
            return False
        # The generic request builder cannot project effort for an unknown
        # provider profile. Do not silently route a requested high-effort turn
        # to a model whose wire request would omit the effort altogether.
        try:
            from providers import get_provider_profile
            profile = get_provider_profile(parsed.provider)
        except Exception:
            logger.warning("Could not resolve pre_primary_turn_route effort provider profile", exc_info=True)
            return False
        if profile is None:
            logger.warning("Ignoring pre_primary_turn_route effort for unknown provider profile")
            return False
    state = _capture_route_state(agent)
    # The canonical switcher retains the old API key when no new one is supplied.
    # Resolve a destination credential before crossing a provider/endpoint boundary;
    # otherwise the old provider's secret would be sent to the new host.
    new_key = None
    resolved_base_url = parsed.base_url
    crosses_endpoint = (
        parsed.provider != getattr(agent, "provider", "")
        or (parsed.base_url and parsed.base_url != getattr(agent, "base_url", ""))
    )
    try:
        if crosses_endpoint:
            from agent.auxiliary_client import resolve_provider_client
            destination, _resolved_model = resolve_provider_client(
                parsed.provider, model=parsed.model, raw_codex=True,
            )
            if destination is None:
                raise ValueError("destination provider is not configured")
            resolved_base_url = str(destination.base_url)
            new_key = getattr(destination, "api_key", "") or ""
            if not resolved_base_url:
                raise ValueError("destination provider has no endpoint")
            # An explicit URL in plugin output is only an identity assertion, not a
            # resolver override: composing a provider's key with an arbitrary URL
            # would disclose that credential to a plugin-controlled host.
            if parsed.base_url and parsed.base_url.rstrip("/") != resolved_base_url.rstrip("/"):
                raise ValueError("destination endpoint differs from configured provider")
        from agent.agent_runtime_helpers import switch_model
        old_key = getattr(agent, "api_key", "")
        if crosses_endpoint:
            # Empty-key local endpoints must not inherit the previous provider's key.
            agent.api_key = ""
        agent._primary_turn_route_state = state
        switch_model(
            agent, new_model=parsed.model, new_provider=parsed.provider,
            api_key=new_key or "", base_url=resolved_base_url, api_mode=parsed.api_mode,
        )
        if effort is not None:
            agent.reasoning_config = effort
            # Fallback restoration reads the acting primary snapshot mid-turn.
            # Keep the override there too, not only on the live agent.
            agent._primary_runtime["reasoning_config"] = copy.deepcopy(effort)
        return True
    except Exception:
        logger.warning("Ignoring failed pre_primary_turn_route decision", exc_info=True)
        if getattr(agent, "_primary_turn_route_state", None) is state:
            if not restore_primary_turn_route(agent):
                # Retain the state and stop this turn. Continuing could send A's
                # credentials to a partially activated B endpoint.
                raise RuntimeError("Cannot start turn after failed temporary route restoration") from None
            if crosses_endpoint:
                agent.api_key = old_key
        return False


def restore_primary_turn_route(agent: Any) -> bool:
    """Restore a completed temporary route through the canonical switch path.

    Usage is recorded before this function runs, preserving the actual model and
    provider that served the request. Failed restores retain state for a retry at
    the next turn boundary.
    """
    state = getattr(agent, "_primary_turn_route_state", None)
    if not isinstance(state, _PrimaryTurnRouteState):
        return False
    runtime = state.primary_runtime
    try:
        model, provider = runtime.get("model"), runtime.get("provider")
        if not isinstance(model, str) or not model or not isinstance(provider, str) or not provider:
            raise ValueError("original primary runtime is incomplete")
        from agent.agent_runtime_helpers import switch_model
        switch_model(
            agent, new_model=model, new_provider=provider,
            api_key=runtime.get("api_key", "") or "",
            base_url=runtime.get("base_url", "") or "",
            api_mode=runtime.get("api_mode", "") or "",
            capabilities=runtime.get("runtime_capabilities"),
        )
        agent._primary_runtime = copy.deepcopy(dict(runtime))
        agent.reasoning_config = copy.deepcopy(state.reasoning_config)
        agent._cached_system_prompt = state.cached_system_prompt
        for name, value in state.fallback_state.items():
            setattr(agent, name, copy.deepcopy(value))
        # The temporary primary must not remain visible to auxiliary routing after cleanup.
        from agent.turn_context import _publish_runtime_main
        _publish_runtime_main(agent)
        agent._primary_turn_route_state = None
        return True
    except Exception:
        logger.warning("Failed to restore pre-primary-turn route", exc_info=True)
        return False
