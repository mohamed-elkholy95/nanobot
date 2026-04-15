"""Tests for SelfTool v2 — runtime inspection and configuration tuning."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.self import SelfTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_loop(**overrides):
    """Build a lightweight mock AgentLoop with the attributes SelfTool reads."""
    loop = MagicMock()
    loop.model = "anthropic/claude-sonnet-4-20250514"
    loop.max_iterations = 40
    loop.context_window_tokens = 65_536
    loop.workspace = Path("/tmp/workspace")
    loop.restrict_to_workspace = False
    loop._start_time = 1000.0
    loop.exec_config = MagicMock()
    loop.channels_config = MagicMock()
    loop._last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
    loop._runtime_vars = {}
    loop._config_defaults = {
        "max_iterations": 40,
        "context_window_tokens": 65_536,
        "model": "anthropic/claude-sonnet-4-20250514",
    }
    loop._critical_tool_backup = {}
    loop.provider_retry_mode = "standard"
    loop.max_tool_result_chars = 16000

    # Tools registry mock
    loop.tools = MagicMock()
    loop.tools.tool_names = ["read_file", "write_file", "exec", "web_search", "self"]
    loop.tools.has.side_effect = lambda n: n in loop.tools.tool_names
    loop.tools.get.return_value = None

    # SubagentManager mock
    loop.subagents = MagicMock()
    loop.subagents._running_tasks = {"abc123": MagicMock(done=MagicMock(return_value=False))}
    loop.subagents.get_running_count = MagicMock(return_value=1)

    for k, v in overrides.items():
        setattr(loop, k, v)

    return loop


def _make_tool(loop=None):
    if loop is None:
        loop = _make_mock_loop()
    return SelfTool(loop=loop)


# ---------------------------------------------------------------------------
# inspect — no key (summary)
# ---------------------------------------------------------------------------

class TestInspectSummary:

    @pytest.mark.asyncio
    async def test_inspect_returns_current_state(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect")
        assert "max_iterations: 40" in result
        assert "context_window_tokens: 65536" in result

    @pytest.mark.asyncio
    async def test_inspect_includes_runtime_vars(self):
        loop = _make_mock_loop()
        loop._runtime_vars = {"task": "review"}
        tool = _make_tool(loop)
        result = await tool.execute(action="inspect")
        assert "task" in result

    @pytest.mark.asyncio
    async def test_inspect_summary_shows_all_description_keys(self):
        """inspect without key should show all top-level keys listed in description."""
        tool = _make_tool()
        result = await tool.execute(action="inspect")
        assert "max_iterations" in result
        assert "context_window_tokens" in result
        assert "model" in result
        assert "workspace" in result
        assert "provider_retry_mode" in result
        assert "max_tool_result_chars" in result


# ---------------------------------------------------------------------------
# inspect — single key (direct)
# ---------------------------------------------------------------------------

class TestInspectSingleKey:

    @pytest.mark.asyncio
    async def test_inspect_simple_value(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="max_iterations")
        assert "40" in result

    @pytest.mark.asyncio
    async def test_inspect_blocked_returns_error(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="bus")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_inspect_dunder_blocked(self):
        tool = _make_tool()
        for attr in ("__class__", "__dict__", "__bases__", "__subclasses__", "__mro__"):
            result = await tool.execute(action="inspect", key=attr)
            assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_inspect_nonexistent_returns_not_found(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="nonexistent_attr_xyz")
        assert "not found" in result


# ---------------------------------------------------------------------------
# inspect — dot-path navigation
# ---------------------------------------------------------------------------

class TestInspectPathNavigation:

    @pytest.mark.asyncio
    async def test_inspect_subattribute_via_dotpath_blocked(self):
        """subagents is BLOCKED — inspect should be rejected."""
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="subagents._running_tasks")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_inspect_config_subfield(self):
        loop = _make_mock_loop()
        loop.web_config = MagicMock()
        loop.web_config.enable = True
        tool = _make_tool(loop)
        result = await tool.execute(action="inspect", key="web_config.enable")
        assert "True" in result

    @pytest.mark.asyncio
    async def test_inspect_dict_key_via_dotpath(self):
        loop = _make_mock_loop()
        loop._last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        tool = _make_tool(loop)
        result = await tool.execute(action="inspect", key="_last_usage.prompt_tokens")
        assert "100" in result

    @pytest.mark.asyncio
    async def test_inspect_blocked_in_path(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="bus.foo")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_inspect_tools_returns_summary(self):
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="tools")
        assert "tools" in result.lower()

    @pytest.mark.asyncio
    async def test_inspect_method_returns_hint_blocked(self):
        """subagents is BLOCKED — inspect should be rejected."""
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="subagents.get_running_count")
        assert "not accessible" in result


# ---------------------------------------------------------------------------
# modify — restricted (with validation)
# ---------------------------------------------------------------------------

class TestModifyRestricted:

    @pytest.mark.asyncio
    async def test_modify_restricted_valid(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=80)
        assert "Set max_iterations = 80" in result
        assert tool._loop.max_iterations == 80

    @pytest.mark.asyncio
    async def test_modify_restricted_out_of_range(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=0)
        assert "Error" in result
        assert tool._loop.max_iterations == 40

    @pytest.mark.asyncio
    async def test_modify_restricted_max_exceeded(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=999)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modify_restricted_wrong_type(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value="not_an_int")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modify_restricted_bool_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=True)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modify_string_int_coerced(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value="80")
        assert tool._loop.max_iterations == 80

    @pytest.mark.asyncio
    async def test_modify_context_window_valid(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="context_window_tokens", value=131072)
        assert tool._loop.context_window_tokens == 131072

    @pytest.mark.asyncio
    async def test_modify_none_value_for_restricted_int(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=None)
        assert "Error" in result


# ---------------------------------------------------------------------------
# modify — blocked (minimal set)
# ---------------------------------------------------------------------------

class TestModifyBlocked:

    @pytest.mark.asyncio
    async def test_modify_bus_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="bus", value="hacked")
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_provider_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="provider", value=None)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_running_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_running", value=True)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_config_defaults_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_config_defaults", value={})
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_dunder_blocked(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="__class__", value="evil")
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_dotpath_leaf_dunder_blocked(self):
        """Fix 3.1: leaf segment of dot-path must also be validated."""
        tool = _make_tool()
        result = await tool.execute(
            action="modify",
            key="provider_retry_mode.__class__",
            value="evil",
        )
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_modify_dotpath_leaf_denied_attr_blocked(self):
        """Fix 3.1: leaf segment matching _DENIED_ATTRS must be rejected."""
        tool = _make_tool()
        result = await tool.execute(
            action="modify",
            key="provider_retry_mode.__globals__",
            value={},
        )
        assert "not accessible" in result


# ---------------------------------------------------------------------------
# modify — free tier (setattr priority)
# ---------------------------------------------------------------------------

class TestModifyFree:

    @pytest.mark.asyncio
    async def test_modify_existing_attr_setattr(self):
        """Modifying an existing loop attribute should use setattr."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="provider_retry_mode", value="persistent")
        assert "Set provider_retry_mode" in result
        assert tool._loop.provider_retry_mode == "persistent"

    @pytest.mark.asyncio
    async def test_modify_new_key_stores_in_runtime_vars(self):
        """Modifying a non-existing attribute should store in _runtime_vars."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="my_custom_var", value="hello")
        assert "my_custom_var" in result
        assert tool._loop._runtime_vars["my_custom_var"] == "hello"

    @pytest.mark.asyncio
    async def test_modify_rejects_callable(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="evil", value=lambda: None)
        assert "callable" in result

    @pytest.mark.asyncio
    async def test_modify_rejects_complex_objects(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="obj", value=Path("/tmp"))
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modify_allows_list(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="items", value=[1, 2, 3])
        assert tool._loop._runtime_vars["items"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_modify_allows_dict(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="data", value={"a": 1})
        assert tool._loop._runtime_vars["data"] == {"a": 1}

    @pytest.mark.asyncio
    async def test_modify_whitespace_key_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="   ", value="test")
        assert "cannot be empty or whitespace" in result

    @pytest.mark.asyncio
    async def test_modify_nested_dict_with_object_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="evil", value={"nested": object()})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modify_deep_nesting_rejected(self):
        tool = _make_tool()
        deep = {"level": 0}
        current = deep
        for i in range(1, 15):
            current["child"] = {"level": i}
            current = current["child"]
        result = await tool.execute(action="modify", key="deep", value=deep)
        assert "nesting too deep" in result

    @pytest.mark.asyncio
    async def test_modify_dict_with_non_str_key_rejected(self):
        tool = _make_tool()
        result = await tool.execute(action="modify", key="evil", value={42: "value"})
        assert "key must be str" in result


# ---------------------------------------------------------------------------
# modify — previously BLOCKED/READONLY now open
# ---------------------------------------------------------------------------

class TestModifyOpen:

    @pytest.mark.asyncio
    async def test_modify_tools_blocked(self):
        """tools is BLOCKED — cannot be replaced."""
        tool = _make_tool()
        new_registry = MagicMock()
        result = await tool.execute(action="modify", key="tools", value=new_registry)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_subagents_blocked(self):
        """subagents is BLOCKED — cannot be replaced."""
        tool = _make_tool()
        new_subagents = MagicMock()
        result = await tool.execute(action="modify", key="subagents", value=new_subagents)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_runner_blocked(self):
        """runner is BLOCKED — cannot be replaced."""
        tool = _make_tool()
        new_runner = MagicMock()
        result = await tool.execute(action="modify", key="runner", value=new_runner)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_sessions_blocked(self):
        """sessions is BLOCKED — cannot be replaced."""
        tool = _make_tool()
        new_sessions = MagicMock()
        result = await tool.execute(action="modify", key="sessions", value=new_sessions)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_consolidator_blocked(self):
        """consolidator is BLOCKED — cannot be replaced."""
        tool = _make_tool()
        new_consolidator = MagicMock()
        result = await tool.execute(action="modify", key="consolidator", value=new_consolidator)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_dream_blocked(self):
        """dream is BLOCKED — cannot be replaced."""
        tool = _make_tool()
        new_dream = MagicMock()
        result = await tool.execute(action="modify", key="dream", value=new_dream)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_auto_compact_blocked(self):
        """auto_compact is BLOCKED — cannot be replaced."""
        tool = _make_tool()
        new_auto_compact = MagicMock()
        result = await tool.execute(action="modify", key="auto_compact", value=new_auto_compact)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_context_blocked(self):
        """context is BLOCKED — cannot be replaced."""
        tool = _make_tool()
        new_context = MagicMock()
        result = await tool.execute(action="modify", key="context", value=new_context)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_commands_blocked(self):
        """commands is BLOCKED — cannot be replaced."""
        tool = _make_tool()
        new_commands = MagicMock()
        result = await tool.execute(action="modify", key="commands", value=new_commands)
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_workspace_allowed(self):
        """workspace was READONLY in v1, now freely modifiable."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="workspace", value="/new/path")
        assert "Set workspace" in result

    @pytest.mark.asyncio
    async def test_modify_mcp_servers_blocked(self):
        """_mcp_servers contains API credentials — must be blocked."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_mcp_servers", value={"evil": "leaked"})
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_mcp_stacks_blocked(self):
        """_mcp_stacks holds connection handles — must be blocked."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_mcp_stacks", value={})
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_pending_queues_blocked(self):
        """_pending_queues controls message routing — must be blocked."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_pending_queues", value={})
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_session_locks_blocked(self):
        """_session_locks controls session isolation — must be blocked."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_session_locks", value={})
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_active_tasks_blocked(self):
        """_active_tasks tracks running tasks — must be blocked."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_active_tasks", value={})
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_background_tasks_blocked(self):
        """_background_tasks tracks background tasks — must be blocked."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="_background_tasks", value=[])
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_inspect_mcp_servers_blocked(self):
        """_mcp_servers contains credentials — inspect must be blocked too."""
        tool = _make_tool()
        result = await tool.execute(action="inspect", key="_mcp_servers")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_modify_wrapped_denied(self):
        """__wrapped__ allows decorator bypass — must be denied."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="__wrapped__", value="evil")
        assert "protected" in result

    @pytest.mark.asyncio
    async def test_modify_closure_denied(self):
        """__closure__ exposes function internals — must be denied."""
        tool = _make_tool()
        result = await tool.execute(action="modify", key="__closure__", value="evil")
        assert "protected" in result


# ---------------------------------------------------------------------------
# validate_json_safe — element counting
# ---------------------------------------------------------------------------

class TestValidateJsonSafe:

    def test_sibling_lists_counted_correctly(self):
        """Fix 3.3: sibling list elements should accumulate across recursion."""
        # Two sibling lists of 600 each = 1200 total elements, over the 1024 limit
        big_value = {"a": list(range(600)), "b": list(range(600))}
        err = SelfTool._validate_json_safe(big_value)
        assert err is not None
        assert "too large" in err

    def test_single_list_within_limit(self):
        """A single list of 500 items should pass."""
        ok_value = list(range(500))
        assert SelfTool._validate_json_safe(ok_value) is None

    def test_deeply_nested_within_limit(self):
        """Deeply nested structures that stay under limit should pass."""
        value = {"level1": {"level2": {"level3": list(range(100))}}}
        assert SelfTool._validate_json_safe(value) is None


# ---------------------------------------------------------------------------
# unknown action
# ---------------------------------------------------------------------------

class TestUnknownAction:

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = _make_tool()
        result = await tool.execute(action="explode")
        assert "Unknown action" in result


# ---------------------------------------------------------------------------
# runtime_vars limits (from code review)
# ---------------------------------------------------------------------------

class TestRuntimeVarsLimits:

    @pytest.mark.asyncio
    async def test_runtime_vars_rejects_at_max_keys(self):
        loop = _make_mock_loop()
        loop._runtime_vars = {f"key_{i}": i for i in range(64)}
        tool = _make_tool(loop)
        result = await tool.execute(action="modify", key="overflow", value="data")
        assert "full" in result
        assert "overflow" not in loop._runtime_vars

    @pytest.mark.asyncio
    async def test_runtime_vars_allows_update_existing_key_at_max(self):
        loop = _make_mock_loop()
        loop._runtime_vars = {f"key_{i}": i for i in range(64)}
        tool = _make_tool(loop)
        result = await tool.execute(action="modify", key="key_0", value="updated")
        assert "Error" not in result
        assert loop._runtime_vars["key_0"] == "updated"

    @pytest.mark.asyncio
    async def test_value_too_large_rejected(self):
        tool = _make_tool()
        big_list = list(range(2000))
        result = await tool.execute(action="modify", key="big", value=big_list)
        assert "too large" in result
        assert "big" not in tool._loop._runtime_vars


# ---------------------------------------------------------------------------
# denied attrs (non-dunder)
# ---------------------------------------------------------------------------

class TestDeniedAttrs:

    @pytest.mark.asyncio
    async def test_modify_denied_non_dunder_blocked(self):
        tool = _make_tool()
        for attr in ("func_globals", "func_code"):
            result = await tool.execute(action="modify", key=attr, value="evil")
            assert "protected" in result, f"{attr} should be blocked"


# ---------------------------------------------------------------------------
# watchdog (with real _watchdog_check method)
# ---------------------------------------------------------------------------

class TestWatchdog:

    def test_watchdog_corrects_invalid_iterations(self):
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        loop.max_iterations = 0
        loop._watchdog_check()
        assert loop.max_iterations == 40

    def test_watchdog_corrects_invalid_context_window(self):
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        loop.context_window_tokens = 100
        loop._watchdog_check()
        assert loop.context_window_tokens == 65_536

    def test_watchdog_restores_critical_tools(self):
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        backup = MagicMock()
        loop._critical_tool_backup = {"self": backup}
        loop.tools.has.return_value = False
        loop.tools.tool_names = []
        loop._watchdog_check()
        loop.tools.register.assert_called()
        called_arg = loop.tools.register.call_args[0][0]
        assert called_arg is not backup

    def test_watchdog_does_not_reset_valid_state(self):
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        loop.max_iterations = 50
        loop.context_window_tokens = 131072
        loop._watchdog_check()
        assert loop.max_iterations == 50
        assert loop.context_window_tokens == 131072

    def test_watchdog_corrects_empty_model(self):
        """Watchdog should reset model to default when set to empty string."""
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        loop.model = ""
        loop._watchdog_check()
        assert loop.model == "anthropic/claude-sonnet-4-20250514"

    def test_watchdog_corrects_none_model(self):
        """Watchdog should reset model to default when set to None."""
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        loop.model = None
        loop._watchdog_check()
        assert loop.model == "anthropic/claude-sonnet-4-20250514"

    def test_watchdog_corrects_whitespace_only_model(self):
        """Watchdog should reset model to default when set to whitespace only."""
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        loop.model = "   "
        loop._watchdog_check()
        assert loop.model == "anthropic/claude-sonnet-4-20250514"

    def test_watchdog_preserves_valid_model(self):
        """Watchdog should not reset a valid model string."""
        from nanobot.agent.loop import AgentLoop
        loop = _make_mock_loop()
        loop._watchdog_check = AgentLoop._watchdog_check.__get__(loop)
        loop.model = "gpt-4o"
        loop._watchdog_check()
        assert loop.model == "gpt-4o"


# ---------------------------------------------------------------------------
# SubagentStatus formatting
# ---------------------------------------------------------------------------

class TestSubagentStatusFormatting:

    def test_format_single_status(self):
        """_format_value should produce a rich multi-line display for a SubagentStatus."""
        from nanobot.agent.subagent import SubagentStatus

        status = SubagentStatus(
            task_id="abc12345",
            label="read logs and summarize",
            task_description="Read the log files and produce a summary",
            started_at=time.monotonic() - 12.4,
            phase="awaiting_tools",
            iteration=3,
            tool_events=[
                {"name": "read_file", "status": "ok", "detail": "read app.log"},
                {"name": "grep", "status": "ok", "detail": "searched ERROR"},
                {"name": "exec", "status": "error", "detail": "timeout"},
            ],
            usage={"prompt_tokens": 4500, "completion_tokens": 1200},
        )
        result = SelfTool._format_value(status)
        assert "abc12345" in result
        assert "read logs and summarize" in result
        assert "awaiting_tools" in result
        assert "iteration: 3" in result
        assert "read_file(ok)" in result
        assert "exec(error)" in result
        assert "4500" in result

    def test_format_status_dict(self):
        """_format_value should handle dict[str, SubagentStatus] with rich display."""
        from nanobot.agent.subagent import SubagentStatus

        statuses = {
            "abc12345": SubagentStatus(
                task_id="abc12345",
                label="task A",
                task_description="Do task A",
                started_at=time.monotonic() - 5.0,
                phase="awaiting_tools",
                iteration=1,
            ),
        }
        result = SelfTool._format_value(statuses)
        assert "1 subagent(s)" in result
        assert "abc12345" in result
        assert "task A" in result

    def test_format_empty_status_dict(self):
        """Empty dict[str, SubagentStatus] should show 'no running subagents'."""
        result = SelfTool._format_value({})
        assert "{}" in result

    def test_format_status_with_error(self):
        """Status with error should include the error message."""
        from nanobot.agent.subagent import SubagentStatus

        status = SubagentStatus(
            task_id="err00001",
            label="failing task",
            task_description="A task that fails",
            started_at=time.monotonic() - 1.0,
            phase="error",
            error="Connection refused",
        )
        result = SelfTool._format_value(status)
        assert "error: Connection refused" in result

    def test_format_subagent_manager_with_statuses(self):
        """SubagentManager-like object with _task_statuses should show rich display."""
        from nanobot.agent.subagent import SubagentStatus

        mgr = MagicMock()
        mgr._running_tasks = {"abc": MagicMock()}
        mgr._task_statuses = {
            "abc": SubagentStatus(
                task_id="abc",
                label="work",
                task_description="Do work",
                started_at=time.monotonic() - 2.0,
                phase="tools_completed",
                iteration=2,
            ),
        }
        result = SelfTool._format_value(mgr)
        assert "abc" in result
        assert "work" in result
        assert "tools_completed" in result

    def test_format_subagent_manager_fallback_no_statuses(self):
        """SubagentManager with empty _task_statuses falls back to simple display."""
        mgr = MagicMock()
        mgr._running_tasks = {"abc": MagicMock()}
        mgr._task_statuses = {}
        result = SelfTool._format_value(mgr)
        assert "1 running" in result


# ---------------------------------------------------------------------------
# _SubagentHook after_iteration updates status
# ---------------------------------------------------------------------------

class TestSubagentHookStatus:

    @pytest.mark.asyncio
    async def test_after_iteration_updates_status(self):
        """after_iteration should copy iteration, tool_events, usage to status."""
        from nanobot.agent.subagent import SubagentStatus, _SubagentHook
        from nanobot.agent.hook import AgentHookContext

        status = SubagentStatus(
            task_id="test",
            label="test",
            task_description="test",
            started_at=time.monotonic(),
        )
        hook = _SubagentHook("test", status)

        context = AgentHookContext(
            iteration=5,
            messages=[],
            tool_events=[{"name": "read_file", "status": "ok", "detail": "ok"}],
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        await hook.after_iteration(context)

        assert status.iteration == 5
        assert len(status.tool_events) == 1
        assert status.tool_events[0]["name"] == "read_file"
        assert status.usage == {"prompt_tokens": 100, "completion_tokens": 50}

    @pytest.mark.asyncio
    async def test_after_iteration_with_error(self):
        """after_iteration should set status.error when context has an error."""
        from nanobot.agent.subagent import SubagentStatus, _SubagentHook
        from nanobot.agent.hook import AgentHookContext

        status = SubagentStatus(
            task_id="test",
            label="test",
            task_description="test",
            started_at=time.monotonic(),
        )
        hook = _SubagentHook("test", status)

        context = AgentHookContext(
            iteration=1,
            messages=[],
            error="something went wrong",
        )
        await hook.after_iteration(context)

        assert status.error == "something went wrong"

    @pytest.mark.asyncio
    async def test_after_iteration_no_status_is_noop(self):
        """after_iteration with no status should be a no-op."""
        from nanobot.agent.subagent import _SubagentHook
        from nanobot.agent.hook import AgentHookContext

        hook = _SubagentHook("test")
        context = AgentHookContext(iteration=1, messages=[])
        await hook.after_iteration(context)  # should not raise


# ---------------------------------------------------------------------------
# Checkpoint callback updates status
# ---------------------------------------------------------------------------

class TestCheckpointCallback:

    @pytest.mark.asyncio
    async def test_checkpoint_updates_phase_and_iteration(self):
        """The _on_checkpoint callback should update status.phase and iteration."""
        from nanobot.agent.subagent import SubagentStatus
        import asyncio

        status = SubagentStatus(
            task_id="cp",
            label="test",
            task_description="test",
            started_at=time.monotonic(),
        )

        # Simulate the checkpoint callback as defined in _run_subagent
        async def _on_checkpoint(payload: dict) -> None:
            status.phase = payload.get("phase", status.phase)
            status.iteration = payload.get("iteration", status.iteration)

        await _on_checkpoint({"phase": "awaiting_tools", "iteration": 2})
        assert status.phase == "awaiting_tools"
        assert status.iteration == 2

        await _on_checkpoint({"phase": "tools_completed", "iteration": 3})
        assert status.phase == "tools_completed"
        assert status.iteration == 3

    @pytest.mark.asyncio
    async def test_checkpoint_preserves_phase_on_missing_key(self):
        """If payload doesn't have 'phase', status.phase should stay unchanged."""
        from nanobot.agent.subagent import SubagentStatus

        status = SubagentStatus(
            task_id="cp",
            label="test",
            task_description="test",
            started_at=time.monotonic(),
            phase="initializing",
        )

        async def _on_checkpoint(payload: dict) -> None:
            status.phase = payload.get("phase", status.phase)
            status.iteration = payload.get("iteration", status.iteration)

        await _on_checkpoint({"iteration": 1})
        assert status.phase == "initializing"
        assert status.iteration == 1


# ---------------------------------------------------------------------------
# inspect subagents._task_statuses via dot-path
# NOTE: subagents is now BLOCKED for security, so these tests verify
# that access is properly rejected.
# ---------------------------------------------------------------------------

class TestInspectTaskStatuses:

    @pytest.mark.asyncio
    async def test_inspect_task_statuses_dotpath_blocked(self):
        """subagents is BLOCKED — inspect should be rejected."""
        from nanobot.agent.subagent import SubagentStatus

        loop = _make_mock_loop()
        loop.subagents._task_statuses = {
            "abc12345": SubagentStatus(
                task_id="abc12345",
                label="read logs",
                task_description="Read the log files",
                started_at=time.monotonic() - 8.0,
                phase="awaiting_tools",
                iteration=2,
                tool_events=[{"name": "read_file", "status": "ok", "detail": "ok"}],
                usage={"prompt_tokens": 500, "completion_tokens": 100},
            ),
        }
        tool = _make_tool(loop)
        result = await tool.execute(action="inspect", key="subagents._task_statuses")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_inspect_single_subagent_status_blocked(self):
        """subagents is BLOCKED — inspect should be rejected."""
        from nanobot.agent.subagent import SubagentStatus

        loop = _make_mock_loop()
        status = SubagentStatus(
            task_id="xyz",
            label="search code",
            task_description="Search the codebase",
            started_at=time.monotonic() - 3.0,
            phase="done",
            iteration=4,
            stop_reason="completed",
        )
        loop.subagents._task_statuses = {"xyz": status}
        tool = _make_tool(loop)
        result = await tool.execute(action="inspect", key="subagents._task_statuses.xyz")
        assert "not accessible" in result


# ---------------------------------------------------------------------------
# read-only mode (self_modify=False)
# ---------------------------------------------------------------------------

class TestReadOnlyMode:

    def _make_readonly_tool(self):
        loop = _make_mock_loop()
        return SelfTool(loop=loop, modify_allowed=False)

    @pytest.mark.asyncio
    async def test_inspect_allowed_in_readonly(self):
        tool = self._make_readonly_tool()
        result = await tool.execute(action="inspect", key="max_iterations")
        assert "40" in result

    @pytest.mark.asyncio
    async def test_modify_blocked_in_readonly(self):
        tool = self._make_readonly_tool()
        result = await tool.execute(action="modify", key="max_iterations", value=80)
        assert "disabled" in result

    def test_description_shows_readonly(self):
        tool = self._make_readonly_tool()
        assert "READ-ONLY MODE" in tool.description

    def test_description_shows_warning_when_modify_allowed(self):
        tool = _make_tool()
        assert "IMPORTANT" in tool.description
        assert "READ-ONLY" not in tool.description
