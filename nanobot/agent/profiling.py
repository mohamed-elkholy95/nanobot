"""Optional profiling hook for agent loop timing.

Enable by setting the environment variable ``NANOBOT_PROFILING=1``.
Timings are emitted at DEBUG level via loguru.
"""

import os
import time

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext


def is_profiling_enabled() -> bool:
    """Check at runtime whether profiling is active."""
    return os.environ.get("NANOBOT_PROFILING", "").strip() in ("1", "true", "yes")


class ProfilingHook(AgentHook):
    """Logs wall-clock time for each LLM iteration and tool batch."""

    def __init__(self) -> None:
        super().__init__()
        self._iter_t0: float = 0.0

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._iter_t0 = time.perf_counter()

    async def after_iteration(self, context: AgentHookContext) -> None:
        elapsed_ms = (time.perf_counter() - self._iter_t0) * 1000
        tool_names = [tc.name for tc in context.tool_calls] if context.tool_calls else []
        logger.debug(
            "[profiling] iteration {}: {:.0f}ms | tools: {}",
            context.iteration,
            elapsed_ms,
            tool_names or "none",
        )

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        tool_names = [tc.name for tc in context.tool_calls]
        logger.debug("[profiling] executing tools: {}", tool_names)
