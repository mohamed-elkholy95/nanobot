"""Tests for the /clear slash command (PR #3467 + Re-bin's review concerns)."""

from __future__ import annotations

import asyncio
import contextvars
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.session.manager import (
    Session,
    SessionManager,
    enter_turn_generation_guard,
    exit_turn_generation_guard,
)


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager"):
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


class TestClearCommandBehavior:
    """User-visible behavior of /clear (carried over from PR #3467)."""

    @pytest.mark.asyncio
    async def test_clear_reports_message_count(self):
        loop, _bus = _make_loop()
        session = MagicMock()
        session.get_history.return_value = [{"role": "user"}] * 5
        session.messages = [{"role": "user"}] * 5
        session.last_consolidated = 0
        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/clear")
        response = await loop._process_message(msg)

        assert response is not None
        assert "5 messages" in response.content
        session.clear.assert_called_once()
        loop.sessions.save.assert_called()
        loop.sessions.invalidate.assert_called()

    @pytest.mark.asyncio
    async def test_clear_uses_singular_for_one_message(self):
        loop, _bus = _make_loop()
        session = MagicMock()
        session.get_history.return_value = [{"role": "user"}]
        session.messages = [{"role": "user"}]
        session.last_consolidated = 0
        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/clear")
        response = await loop._process_message(msg)

        assert response is not None
        assert "1 message" in response.content
        assert "1 messages" not in response.content

    @pytest.mark.asyncio
    async def test_clear_empty_session(self):
        loop, _bus = _make_loop()
        session = MagicMock()
        session.get_history.return_value = []
        session.messages = []
        session.last_consolidated = 0
        loop.sessions.get_or_create.return_value = session

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/clear")
        response = await loop._process_message(msg)

        assert response is not None
        assert "0 messages" in response.content

    @pytest.mark.asyncio
    async def test_clear_does_not_cancel_tasks(self):
        """/clear must NOT cancel running tasks — that's /new's job.

        The whole point of /clear vs /new is that active work keeps running;
        the turn-generation guard in SessionManager.save catches the late
        write-back instead.
        """
        loop, _bus = _make_loop()
        session = MagicMock()
        session.get_history.return_value = []
        session.messages = []
        session.last_consolidated = 0
        loop.sessions.get_or_create.return_value = session
        loop._cancel_active_tasks = MagicMock()

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/clear")
        await loop._process_message(msg)

        loop._cancel_active_tasks.assert_not_called()

    @pytest.mark.asyncio
    async def test_help_includes_clear(self):
        loop, _bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/help")

        response = await loop._process_message(msg)

        assert response is not None
        assert "/clear" in response.content


class TestClearArchivesUnconsolidatedSnapshot:
    """Re-bin's 2nd concern: unconsolidated tail must be archived, not silently dropped."""

    @pytest.mark.asyncio
    async def test_clear_schedules_archive_of_unconsolidated_tail(self):
        loop, _bus = _make_loop()
        session = MagicMock()
        # 3 unconsolidated messages past last_consolidated=2
        all_msgs = [
            {"role": "user", "content": "old1"},
            {"role": "assistant", "content": "old2"},
            {"role": "user", "content": "new1"},
            {"role": "assistant", "content": "new2"},
            {"role": "user", "content": "new3"},
        ]
        session.messages = all_msgs
        session.last_consolidated = 2
        session.get_history.return_value = all_msgs[2:]
        loop.sessions.get_or_create.return_value = session

        archive_mock = MagicMock()
        loop.consolidator.archive = MagicMock(return_value=archive_mock)
        loop._schedule_background = MagicMock()

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/clear")
        response = await loop._process_message(msg)

        assert response is not None
        loop.consolidator.archive.assert_called_once_with(all_msgs[2:])
        loop._schedule_background.assert_called_once_with(archive_mock)

    @pytest.mark.asyncio
    async def test_clear_skips_archive_when_nothing_unconsolidated(self):
        loop, _bus = _make_loop()
        session = MagicMock()
        session.messages = [{"role": "user", "content": "old"}]
        session.last_consolidated = 1  # everything is consolidated already
        session.get_history.return_value = []
        # Real metadata avoids _restore_pending_user_turn appending an
        # "interrupted" message (which would skew the snapshot).
        session.metadata = {}
        loop.sessions.get_or_create.return_value = session

        loop.consolidator.archive = MagicMock()
        loop._schedule_background = MagicMock()

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/clear")
        await loop._process_message(msg)

        loop.consolidator.archive.assert_not_called()
        loop._schedule_background.assert_not_called()


class TestClearWriteBackRaceGuard:
    """Re-bin's primary regression: in-flight turn must not resurrect cleared
    history via late ``SessionManager.save``."""

    @pytest.mark.asyncio
    async def test_late_save_after_clear_does_not_resurrect_history(self, tmp_path: Path):
        mgr = SessionManager(tmp_path)
        session = Session(key="test:race")
        session.add_message("user", "hi")
        session.add_message("assistant", "hello")
        mgr.save(session)
        original_count = len(session.messages)

        # In-flight turn captures generation 0 at its start.
        captured_token = enter_turn_generation_guard(session)
        try:
            # /clear runs concurrently in a separate async task with a
            # fresh contextvar context (so /clear's own save is unguarded).
            async def _do_clear():
                session.clear()
                mgr.save(session)
                mgr.invalidate(session.key)

            await asyncio.get_running_loop().create_task(
                _do_clear(),
                context=contextvars.Context(),
            )

            # The in-flight turn now wakes up (post-_run_agent_loop) and
            # appends its turn results, then calls save.  The guard should
            # detect that session.generation has advanced and refuse to
            # write — leaving disk in /clear's empty state.
            session.add_message("assistant", "stale-tool-result-must-not-resurrect")
            mgr.save(session)
        finally:
            exit_turn_generation_guard(captured_token)

        # Reload the file from disk: it must reflect /clear's empty state,
        # not the late append from the in-flight turn.
        mgr.invalidate(session.key)
        reloaded = mgr.get_or_create(session.key)
        contents = [m.get("content") for m in reloaded.messages]
        assert "stale-tool-result-must-not-resurrect" not in contents, (
            f"Late save resurrected stale content: {contents}"
        )
        assert reloaded.messages == [], (
            f"Disk should be in cleared state but contains {original_count - 0} "
            f"messages: {reloaded.messages}"
        )
        assert reloaded.generation >= 1, "Generation should have advanced past clear"

    @pytest.mark.asyncio
    async def test_save_proceeds_when_generation_matches_guard(self, tmp_path: Path):
        """Sanity check: a normal turn-scoped save (no concurrent /clear)
        must not be impacted by the guard."""
        mgr = SessionManager(tmp_path)
        session = Session(key="test:happy")
        session.add_message("user", "hi")
        mgr.save(session)

        token = enter_turn_generation_guard(session)
        try:
            session.add_message("assistant", "hello")
            mgr.save(session)
        finally:
            exit_turn_generation_guard(token)

        mgr.invalidate(session.key)
        reloaded = mgr.get_or_create(session.key)
        contents = [m.get("content") for m in reloaded.messages]
        assert contents == ["hi", "hello"]


class TestSessionGenerationFieldRoundTrip:
    """The new ``generation`` field must persist across save/load."""

    def test_generation_round_trips_through_save_and_load(self, tmp_path: Path):
        mgr = SessionManager(tmp_path)
        session = Session(key="test:roundtrip")
        session.add_message("user", "hi")
        session.clear()  # bumps generation to 1
        session.add_message("user", "hello again")
        mgr.save(session)

        mgr.invalidate(session.key)
        reloaded = mgr.get_or_create(session.key)
        assert reloaded.generation == 1

    def test_legacy_files_without_generation_load_as_zero(self, tmp_path: Path):
        # Pre-existing session files don't have the generation field.
        path = tmp_path / "test_legacy.jsonl"
        path.write_text(
            '{"_type":"metadata","key":"test:legacy","created_at":"2026-01-01T00:00:00",'
            '"updated_at":"2026-01-01T00:00:00","metadata":{},"last_consolidated":0}\n'
            '{"role":"user","content":"hi"}\n',
            encoding="utf-8",
        )
        mgr = SessionManager(tmp_path)
        # Override path resolution for this isolated test
        target = mgr._get_session_path("test:legacy")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text(), encoding="utf-8")

        loaded = mgr.get_or_create("test:legacy")
        assert loaded.generation == 0
        assert len(loaded.messages) == 1
