"""Tests for MessageBus bounded queues with backpressure."""

import asyncio

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus


def _make_inbound(content: str = "hello") -> InboundMessage:
    return InboundMessage(
        channel="test",
        sender_id="u1",
        chat_id="c1",
        content=content,
    )


def _make_outbound(content: str = "reply") -> OutboundMessage:
    return OutboundMessage(
        channel="test",
        chat_id="c1",
        content=content,
    )


@pytest.mark.asyncio
async def test_default_maxsize_is_bounded():
    bus = MessageBus()
    assert bus.inbound.maxsize == 100
    assert bus.outbound.maxsize == 100


@pytest.mark.asyncio
async def test_custom_maxsize():
    bus = MessageBus(maxsize=10)
    assert bus.inbound.maxsize == 10
    assert bus.outbound.maxsize == 10


@pytest.mark.asyncio
async def test_publish_inbound_returns_true_on_success():
    bus = MessageBus(maxsize=5, timeout=0.01)
    result = await bus.publish_inbound(_make_inbound())
    assert result is True
    assert bus.inbound_size == 1


@pytest.mark.asyncio
async def test_publish_inbound_returns_false_on_full_queue():
    bus = MessageBus(maxsize=2, timeout=0.01)
    assert await bus.publish_inbound(_make_inbound("m1")) is True
    assert await bus.publish_inbound(_make_inbound("m2")) is True
    # Queue is now full — third publish should return False
    assert await bus.publish_inbound(_make_inbound("m3")) is False
    assert bus.inbound_size == 2


@pytest.mark.asyncio
async def test_publish_outbound_returns_true_on_success():
    bus = MessageBus(maxsize=5, timeout=0.01)
    result = await bus.publish_outbound(_make_outbound())
    assert result is True
    assert bus.outbound_size == 1


@pytest.mark.asyncio
async def test_publish_outbound_returns_false_on_full_queue():
    bus = MessageBus(maxsize=2, timeout=0.01)
    assert await bus.publish_outbound(_make_outbound("r1")) is True
    assert await bus.publish_outbound(_make_outbound("r2")) is True
    # Queue is now full — third publish should return False
    assert await bus.publish_outbound(_make_outbound("r3")) is False
    assert bus.outbound_size == 2


@pytest.mark.asyncio
async def test_consume_inbound_drains_queue():
    bus = MessageBus(maxsize=5, timeout=0.01)
    await bus.publish_inbound(_make_inbound("a"))
    await bus.publish_inbound(_make_inbound("b"))
    assert bus.inbound_size == 2

    msg1 = await bus.consume_inbound()
    assert msg1.content == "a"
    assert bus.inbound_size == 1

    msg2 = await bus.consume_inbound()
    assert msg2.content == "b"
    assert bus.inbound_size == 0
