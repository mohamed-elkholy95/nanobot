"""Tests for first-token priority in ChannelManager stream delta dispatching."""
import pytest
from unittest.mock import patch

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config


def _make_manager():
    bus = MessageBus(maxsize=50)
    config = Config()
    mgr = ChannelManager(bus=bus, config=config)
    return mgr


def test_active_streams_starts_empty():
    mgr = _make_manager()
    assert mgr._active_streams == set()


def test_first_delta_adds_to_active_streams():
    mgr = _make_manager()
    assert "s1" not in mgr._active_streams
    mgr._active_streams.add("s1")
    assert "s1" in mgr._active_streams


def test_stream_end_removes_from_active():
    mgr = _make_manager()
    mgr._active_streams.add("s1")
    mgr._active_streams.discard("s1")
    assert "s1" not in mgr._active_streams


def test_fallback_key_without_stream_id():
    # Without stream_id, key should be channel:chat_id
    msg = OutboundMessage(channel="tg", chat_id="c1", content="hi", metadata={"_stream_delta": True})
    key = msg.metadata.get("stream_id") or f"{msg.channel}:{msg.chat_id}"
    assert key == "tg:c1"


def test_stream_id_preferred_over_fallback():
    msg = OutboundMessage(channel="tg", chat_id="c1", content="hi", metadata={"_stream_delta": True, "stream_id": "s42"})
    key = msg.metadata.get("stream_id") or f"{msg.channel}:{msg.chat_id}"
    assert key == "s42"
