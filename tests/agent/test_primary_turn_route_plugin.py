"""Behavior tests for the generic pre-primary-turn route plugin seam."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml


@pytest.fixture(autouse=True)
def _stub_destination_resolution(monkeypatch):
    """No unit test may discover a real provider account or network endpoint."""
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda provider, model, **kwargs: (SimpleNamespace(base_url="https://b.example/v1", api_key="key-for-b"), model),
    )


def _agent() -> SimpleNamespace:
    return SimpleNamespace(
        model="model-a",
        provider="provider-a",
        base_url="https://a.example/v1",
        api_mode="chat_completions",
        session_id="session-a",
        platform="cli",
        reasoning_config={"enabled": True, "effort": "medium"},
        _primary_runtime={"model": "model-a"},
    )


def test_invalid_route_decision_is_rejected_without_switching(monkeypatch):
    """Malformed plugin output is fail-closed: it never reaches the switcher."""
    from agent.primary_turn_route import apply_primary_turn_route

    agent = _agent()
    switch_calls = []
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "pre_primary_turn_route")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: [{"model": 42}])
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.switch_model",
        lambda *args, **kwargs: switch_calls.append((args, kwargs)),
    )

    assert apply_primary_turn_route(agent, user_message="route this", turn_id="turn-a", task_id="task-a") is False
    assert switch_calls == []
    assert agent.model == "model-a"


def test_invalid_route_effort_is_rejected_before_switching(monkeypatch):
    """A bad effort cannot leave a half-applied model switch behind."""
    from agent.primary_turn_route import apply_primary_turn_route

    switch_calls = []
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: [{"model": "model-b", "provider": "provider-b", "reasoning_effort": "typo"}],
    )
    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *args, **kwargs: switch_calls.append(kwargs))

    assert apply_primary_turn_route(_agent(), user_message="route", turn_id="turn-a", task_id="task-a") is False
    assert switch_calls == []


def test_cross_provider_route_resolves_its_own_credentials_before_switch(monkeypatch):
    """Never reuse provider A's key or URL to rebuild provider B's client."""
    from agent.primary_turn_route import apply_primary_turn_route

    agent = _agent()
    agent.api_key = "key-for-a"
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: [{"model": "model-b", "provider": "provider-b"}],
    )
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda provider, model, **kwargs: (SimpleNamespace(base_url="https://b.example/v1", api_key="key-for-b"), model),
    )
    calls = []

    def switch(fake_agent, **kwargs):
        calls.append(kwargs)
        fake_agent.model, fake_agent.provider = kwargs["new_model"], kwargs["new_provider"]

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", switch)
    assert apply_primary_turn_route(agent, user_message="route", turn_id="turn-a", task_id="task-a")
    assert calls[0]["api_key"] == "key-for-b"
    assert calls[0]["base_url"] == "https://b.example/v1"
    assert calls[0]["api_key"] != agent.api_key


def test_unconfigured_destination_is_rejected_without_reusing_old_key(monkeypatch):
    from agent.primary_turn_route import apply_primary_turn_route

    agent = _agent()
    agent.api_key = "key-for-a"
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [{"model": "model-b", "provider": "provider-b"}])
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *_a, **_kw: (None, None))
    calls = []
    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *_a, **_kw: calls.append(True))

    assert not apply_primary_turn_route(agent, user_message="route", turn_id="turn-a", task_id="task-a")
    assert calls == []
    assert (agent.model, agent.provider, agent.api_key) == ("model-a", "provider-a", "key-for-a")


def test_user_plugin_routes_with_safe_context_and_host_applies_effort(monkeypatch, tmp_path):
    """A discovered user plugin can route, but only receives the documented data context."""
    import hermes_cli.plugins as plugins_mod
    from agent.primary_turn_route import apply_primary_turn_route

    plugin_dir = tmp_path / "plugins" / "route_demo"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: route_demo\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    def route(**kw):\n"
        "        assert 'agent' not in kw\n"
        "        assert kw['current_route']['model'] == 'model-a'\n"
        "        assert kw['user_message'] == 'please route'\n"
        "        return {'model': 'model-b', 'provider': 'provider-b', 'reasoning_effort': 'high'}\n"
        "    ctx.register_hook('pre_primary_turn_route', route)\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["route_demo"]}}), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugins_mod._plugin_manager = plugins_mod.PluginManager()

    agent = _agent()
    calls = []

    def switch(fake_agent, **kwargs):
        calls.append(kwargs)
        fake_agent.model = kwargs["new_model"]
        fake_agent.provider = kwargs["new_provider"]

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", switch)

    assert apply_primary_turn_route(agent, user_message="please route", turn_id="turn-a", task_id="task-a") is True
    assert calls == [{"new_model": "model-b", "new_provider": "provider-b", "api_key": "key-for-b", "base_url": "https://b.example/v1", "api_mode": ""}]
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}


def test_temporary_route_restores_original_runtime_after_accounting(monkeypatch):
    """The route is only live for this user turn; accounting can observe it before restoration."""
    from agent.primary_turn_route import apply_primary_turn_route, restore_primary_turn_route

    agent = _agent()
    agent._primary_runtime = {
        "model": "model-a", "provider": "provider-a", "base_url": "https://a.example/v1",
        "api_mode": "chat_completions", "api_key": "not-exposed", "runtime_capabilities": {},
    }
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: [{"model": "model-b", "provider": "provider-b", "reasoning_effort": "high"}],
    )
    observed = []

    def switch(fake_agent, **kwargs):
        observed.append((kwargs["new_model"], kwargs["new_provider"], dict(fake_agent.reasoning_config)))
        fake_agent.model = kwargs["new_model"]
        fake_agent.provider = kwargs["new_provider"]

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", switch)

    assert apply_primary_turn_route(agent, user_message="route", turn_id="turn-a", task_id="task-a") is True
    # Usage accounting runs while the routed identity is still active.
    assert (agent.model, agent.provider, agent.reasoning_config["effort"]) == ("model-b", "provider-b", "high")
    assert restore_primary_turn_route(agent) is True
    assert (agent.model, agent.provider, agent.reasoning_config["effort"]) == ("model-a", "provider-a", "medium")
    assert [item[:2] for item in observed] == [("model-b", "provider-b"), ("model-a", "provider-a")]


def test_restored_route_republishes_original_auxiliary_main(monkeypatch):
    """Auxiliary routing outside the turn must not see its temporary provider as main."""
    from agent.primary_turn_route import apply_primary_turn_route, restore_primary_turn_route

    agent = _agent()
    agent._primary_runtime.update(provider="provider-a", base_url="https://a.example/v1", api_mode="chat_completions")
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [{"model": "model-b", "provider": "provider-b"}])

    def switch(fake_agent, **kwargs):
        fake_agent.model, fake_agent.provider = kwargs["new_model"], kwargs["new_provider"]

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", switch)
    seen = []
    monkeypatch.setattr("agent.turn_context._publish_runtime_main", lambda fake_agent: seen.append(fake_agent.model))

    assert apply_primary_turn_route(agent, user_message="route", turn_id="turn-a", task_id="task-a")
    assert restore_primary_turn_route(agent)
    assert seen == ["model-a"]


def test_real_turn_routes_once_records_b_usage_and_restores_a(monkeypatch, tmp_path):
    """A temp-home plugin exercises the real loop's A→B→A lifecycle and usage path."""
    import hermes_cli.plugins as plugins_mod
    from run_agent import AIAgent

    plugin_dir = tmp_path / "plugins" / "turn_route"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: turn_route\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_hook('pre_primary_turn_route', lambda **kw: {'model': 'model-b', 'provider': 'provider-b'})\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["turn_route"]}}), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugins_mod._plugin_manager = plugins_mod.PluginManager()

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://a.example/v1", provider="provider-a",
            model="model-a", quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client.chat.completions.create.return_value = SimpleNamespace(
        id="route-test", model="model-b",
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="routed", tool_calls=None), finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )
    switched = []

    def switch(fake_agent, **kwargs):
        switched.append((kwargs["new_model"], kwargs["new_provider"]))
        fake_agent.model = kwargs["new_model"]
        fake_agent.provider = kwargs["new_provider"]

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", switch)
    result = agent.run_conversation("route this")

    assert result["model"] == "model-b"
    assert agent.session_total_tokens == 18
    assert (agent.model, agent.provider) == ("model-a", "provider-a")
    assert agent._cached_system_prompt == "You are helpful."
    assert switched == [("model-b", "provider-b"), ("model-a", "provider-a")]


def test_failed_restore_never_starts_next_turn_on_previous_temporary_route(monkeypatch):
    """A failed client rebuild must not leak the old routed identity into another turn."""
    from agent.conversation_loop import _run_conversation_turn

    agent = SimpleNamespace(_primary_turn_route_state=object(), _try_refresh_env_client_credentials=lambda: None)
    monkeypatch.setattr("agent.conversation_loop.begin_fast_mode_turn", lambda *_args: None)
    monkeypatch.setattr("agent.primary_turn_route.restore_primary_turn_route", lambda _agent: False)
    monkeypatch.setattr(
        "agent.conversation_loop.build_turn_context",
        lambda *_args, **_kwargs: pytest.fail("entered turn context with stale route"),
    )
    with pytest.raises(RuntimeError, match="temporary route"):
        _run_conversation_turn(agent, "next")
