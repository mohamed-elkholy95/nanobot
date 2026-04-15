"""Async message queue for decoupled channel-agent communication."""

import asyncio
import os

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage

_DEFAULT_MAX_QUEUE_SIZE = int(os.environ.get("NANOBOT_MAX_QUEUE_SIZE", "100"))


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.

    Both queues are bounded to prevent unbounded memory growth when
    producers outpace consumers.
    """

    def __init__(
        self,
        maxsize: int = _DEFAULT_MAX_QUEUE_SIZE,
        timeout: float = 5.0,
    ):
        self._timeout = timeout
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=maxsize)
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=maxsize)

    async def publish_inbound(self, msg: InboundMessage) -> bool:
        """Publish a message from a channel to the agent.

        Returns True on success, False if the queue is full after timeout.
        """
        try:
            await asyncio.wait_for(self.inbound.put(msg), timeout=self._timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Inbound queue full (size {}), message dropped after {}s timeout",
                self.inbound.qsize(),
                self._timeout,
            )
            return False

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> bool:
        """Publish a response from the agent to channels.

        Blocks until space is available. Agent replies must never be
        silently dropped — if the dispatcher stalls, the agent should
        stall too (visible, diagnosable). The bounded queue still
        provides memory safety.
        """
        await self.outbound.put(msg)
        return True

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
