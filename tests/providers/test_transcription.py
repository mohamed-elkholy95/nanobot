"""Tests for transcription retry behavior on transient errors (B10)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nanobot.providers.transcription import GroqTranscriptionProvider, OpenAITranscriptionProvider


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    p = tmp_path / "voice.ogg"
    p.write_bytes(b"OggS\x00fake-audio-bytes")
    return p


def _response(status: int, payload: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://example.test/audio/transcriptions")
    return httpx.Response(status_code=status, json=payload or {}, request=request)


# ---------------------------------------------------------------------------
# OpenAI provider — retry on transient HTTP + network errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_retries_on_5xx_then_succeeds(audio_file: Path) -> None:
    """Transient 503 is retried; a subsequent 200 yields the text."""
    provider = OpenAITranscriptionProvider(api_key="sk-test")
    post = AsyncMock(side_effect=[_response(503), _response(200, {"text": "hello"})])
    with patch("httpx.AsyncClient.post", post), patch("asyncio.sleep", AsyncMock()):
        result = await provider.transcribe(audio_file)
    assert result == "hello"
    assert post.await_count == 2


@pytest.mark.asyncio
async def test_openai_retries_on_429_then_succeeds(audio_file: Path) -> None:
    provider = OpenAITranscriptionProvider(api_key="sk-test")
    post = AsyncMock(side_effect=[_response(429), _response(200, {"text": "rate ok"})])
    with patch("httpx.AsyncClient.post", post), patch("asyncio.sleep", AsyncMock()):
        result = await provider.transcribe(audio_file)
    assert result == "rate ok"
    assert post.await_count == 2


@pytest.mark.asyncio
async def test_openai_retries_on_connect_error(audio_file: Path) -> None:
    """Network-level transient errors are retried."""
    provider = OpenAITranscriptionProvider(api_key="sk-test")
    post = AsyncMock(side_effect=[httpx.ConnectError("boom"), _response(200, {"text": "ok"})])
    with patch("httpx.AsyncClient.post", post), patch("asyncio.sleep", AsyncMock()):
        result = await provider.transcribe(audio_file)
    assert result == "ok"
    assert post.await_count == 2


@pytest.mark.asyncio
async def test_openai_does_not_retry_on_auth_error(audio_file: Path) -> None:
    """401 is the user's misconfiguration — retrying wastes time and rate-limit quota."""
    provider = OpenAITranscriptionProvider(api_key="sk-test")
    post = AsyncMock(return_value=_response(401, {"error": {"message": "bad key"}}))
    with patch("httpx.AsyncClient.post", post), patch("asyncio.sleep", AsyncMock()):
        result = await provider.transcribe(audio_file)
    assert result == ""
    assert post.await_count == 1


@pytest.mark.asyncio
async def test_openai_gives_up_after_max_attempts(audio_file: Path) -> None:
    """Persistent 503 returns "" after the final retry — never hangs."""
    provider = OpenAITranscriptionProvider(api_key="sk-test")
    post = AsyncMock(return_value=_response(503))
    sleep = AsyncMock()
    with patch("httpx.AsyncClient.post", post), patch("asyncio.sleep", sleep):
        result = await provider.transcribe(audio_file)
    assert result == ""
    # 4 attempts total (initial + 3 retries) with 3 sleeps between them.
    assert post.await_count == 4
    assert sleep.await_count == 3


@pytest.mark.asyncio
async def test_openai_backoff_grows_exponentially(audio_file: Path) -> None:
    """Verify the backoff schedule is exponential (1s, 2s, 4s)."""
    provider = OpenAITranscriptionProvider(api_key="sk-test")
    post = AsyncMock(return_value=_response(503))
    sleep = AsyncMock()
    with patch("httpx.AsyncClient.post", post), patch("asyncio.sleep", sleep):
        await provider.transcribe(audio_file)
    delays = [call.args[0] for call in sleep.await_args_list]
    assert delays == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# Groq provider — same semantics (both go through the shared helper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groq_retries_on_5xx_then_succeeds(audio_file: Path) -> None:
    provider = GroqTranscriptionProvider(api_key="gsk-test")
    post = AsyncMock(side_effect=[_response(502), _response(200, {"text": "groq ok"})])
    with patch("httpx.AsyncClient.post", post), patch("asyncio.sleep", AsyncMock()):
        result = await provider.transcribe(audio_file)
    assert result == "groq ok"
    assert post.await_count == 2


@pytest.mark.asyncio
async def test_groq_does_not_retry_on_auth_error(audio_file: Path) -> None:
    provider = GroqTranscriptionProvider(api_key="gsk-test")
    post = AsyncMock(return_value=_response(403))
    with patch("httpx.AsyncClient.post", post), patch("asyncio.sleep", AsyncMock()):
        result = await provider.transcribe(audio_file)
    assert result == ""
    assert post.await_count == 1


# ---------------------------------------------------------------------------
# Regression: missing file / missing key must still short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_missing_api_key_short_circuits(tmp_path: Path) -> None:
    provider = OpenAITranscriptionProvider(api_key=None)
    # Ensure env var doesn't accidentally satisfy it.
    with patch.dict("os.environ", {}, clear=True):
        provider = OpenAITranscriptionProvider(api_key=None)
        post = AsyncMock()
        with patch("httpx.AsyncClient.post", post):
            assert await provider.transcribe(tmp_path / "voice.ogg") == ""
        assert post.await_count == 0


@pytest.mark.asyncio
async def test_openai_missing_file_short_circuits() -> None:
    provider = OpenAITranscriptionProvider(api_key="sk-test")
    post = AsyncMock()
    with patch("httpx.AsyncClient.post", post):
        assert await provider.transcribe("/nonexistent/path/voice.ogg") == ""
    assert post.await_count == 0
