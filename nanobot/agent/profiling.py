"""Optional profiling hook for agent loop timing.

Enable by setting the environment variable ``NANOBOT_PROFILING=1``.
The variable is read once when ``AgentLoop`` is instantiated, so it
must be set before loop creation (not dynamically at runtime).

Timings are emitted at DEBUG level via loguru.
"""

import os
import time

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext


def is_profiling_enabled() -> bool:
    """Check whether profiling is active."""
    return os.environ.get("NANOBOT_PROFILING", "").strip() in ("1", "true", "yes")


class ProfilingHook(AgentHook):
    """Logs wall-clock time for each LLM iteration."""

    def __init__(self) -> None:
        super().__init__()
        self._iter_t0: float | None = None

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._iter_t0 = time.perf_counter()

    async def after_iteration(self, context: AgentHookContext) -> None:
        if self._iter_t0 is None:
            return
        elapsed_ms = (time.perf_counter() - self._iter_t0) * 1000
        tool_names = [tc.name for tc in context.tool_calls] if context.tool_calls else []
        logger.debug(
            "[profiling] iteration {}: {:.0f}ms | tools: {}",
            context.iteration,
            elapsed_ms,
            tool_names or "none",
        )
