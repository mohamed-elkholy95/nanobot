from unittest.mock import MagicMock

from nanobot.providers.openai_compat_provider import OpenAICompatProvider


def _make_provider():
    p = OpenAICompatProvider.__new__(OpenAICompatProvider)
    p._responses_api_failures = {}
    p._spec = None
    p._effective_base = None
    p._client = MagicMock()
    p.default_model = "gpt-4o"
    p.api_base = None
    return p


def test_failure_counter_starts_empty():
    p = _make_provider()
    assert p._responses_api_failures == {}


def test_failure_key_includes_reasoning_effort():
    p = _make_provider()
    assert p._responses_failure_key("gpt-5", None) == ("gpt-5", None)
    assert p._responses_failure_key("gpt-5", "high") == ("gpt-5", "high")


def test_skip_after_three_failures():
    p = _make_provider()
    key = ("gpt-5", None)
    p._responses_api_failures[key] = 2
    assert not p._should_skip_responses_api("gpt-5", None)
    p._responses_api_failures[key] = 3
    assert p._should_skip_responses_api("gpt-5", None)


def test_success_resets_counter():
    p = _make_provider()
    p._responses_api_failures[("gpt-5", None)] = 5
    p._record_responses_api_success("gpt-5", None)
    assert p._responses_api_failures.get(("gpt-5", None), 0) == 0


def test_failure_increments():
    p = _make_provider()
    p._record_responses_api_failure("gpt-5", "high")
    p._record_responses_api_failure("gpt-5", "high")
    assert p._responses_api_failures[("gpt-5", "high")] == 2
