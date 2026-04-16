"""Tests for the optional ProfilingHook."""

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.profiling import ProfilingHook, is_profiling_enabled


@pytest.fixture()
def hook():
    return ProfilingHook()


@pytest.fixture()
def ctx():
    return AgentHookContext(iteration=1, messages=[])


async def test_before_iteration_records_start_time(hook, ctx):
    assert hook._iter_t0 == 0.0
    await hook.before_iteration(ctx)
    assert hook._iter_t0 > 0.0


async def test_after_iteration_runs_without_error(hook, ctx):
    await hook.before_iteration(ctx)
    await hook.after_iteration(ctx)  # should not raise


async def test_before_execute_tools_runs_without_error(hook, ctx):
    from nanobot.providers.base import ToolCallRequest

    ctx.tool_calls = [ToolCallRequest(id="t1", name="read_file", arguments="{}")]
    await hook.before_execute_tools(ctx)  # should not raise


def test_profiling_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NANOBOT_PROFILING", raising=False)
    assert is_profiling_enabled() is False


def test_profiling_enabled_with_env(monkeypatch):
    monkeypatch.setenv("NANOBOT_PROFILING", "1")
    assert is_profiling_enabled() is True


def test_profiling_enabled_with_true_string(monkeypatch):
    monkeypatch.setenv("NANOBOT_PROFILING", "true")
    assert is_profiling_enabled() is True
