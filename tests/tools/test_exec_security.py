"""Tests for exec tool internal URL blocking."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from nanobot.agent.tools.shell import ExecTool


def _fake_resolve_private(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 0))]


def _fake_resolve_localhost(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]


def _fake_resolve_public(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


@pytest.mark.asyncio
async def test_exec_blocks_curl_metadata():
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_private):
        result = await tool.execute(
            command='curl -s -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/'
        )
    assert "Error" in result
    assert "internal" in result.lower() or "private" in result.lower()


@pytest.mark.asyncio
async def test_exec_blocks_wget_localhost():
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_localhost):
        result = await tool.execute(command="wget http://localhost:8080/secret -O /tmp/out")
    assert "Error" in result


@pytest.mark.asyncio
async def test_exec_allows_normal_commands():
    tool = ExecTool(timeout=5)
    result = await tool.execute(command="echo hello")
    assert "hello" in result
    assert "Error" not in result.split("\n")[0]


@pytest.mark.asyncio
async def test_exec_allows_curl_to_public_url():
    """Commands with public URLs should not be blocked by the internal URL check."""
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public):
        guard_result = tool._guard_command("curl https://example.com/api", "/tmp")
    assert guard_result is None


@pytest.mark.asyncio
async def test_exec_blocks_chained_internal_url():
    """Internal URLs buried in chained commands should still be caught."""
    tool = ExecTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_private):
        result = await tool.execute(
            command="echo start && curl http://169.254.169.254/latest/meta-data/ && echo done"
        )
    assert "Error" in result


# --- #2989: block writes to nanobot internal state files -----------------


@pytest.mark.parametrize(
    "command",
    [
        "cat foo >> history.jsonl",
        "echo '{}' > history.jsonl",
        "echo '{}' > memory/history.jsonl",
        "echo '{}' > ./workspace/memory/history.jsonl",
        "tee -a history.jsonl < foo",
        "tee history.jsonl",
        "cp /tmp/fake.jsonl history.jsonl",
        "mv backup.jsonl memory/history.jsonl",
        "dd if=/dev/zero of=memory/history.jsonl",
        "sed -i 's/old/new/' history.jsonl",
        "echo x > .dream_cursor",
        "cp /tmp/x memory/.dream_cursor",
    ],
)
def test_exec_blocks_writes_to_history_jsonl(command):
    """Direct writes to history.jsonl / .dream_cursor must be blocked (#2989)."""
    tool = ExecTool()
    result = tool._guard_command(command, "/tmp")
    assert result is not None
    assert "dangerous pattern" in result.lower()


@pytest.mark.parametrize(
    "command",
    [
        "cat history.jsonl",
        "wc -l history.jsonl",
        "tail -n 5 history.jsonl",
        "grep foo history.jsonl",
        "cp history.jsonl /tmp/history.backup",
        "ls memory/",
        "echo history.jsonl",
    ],
)
def test_exec_allows_reads_of_history_jsonl(command):
    """Read-only access to history.jsonl must still be allowed."""
    tool = ExecTool()
    result = tool._guard_command(command, "/tmp")
    assert result is None


# --- Hardening: shell expansion must not defeat #2989 protections --------
#
# The literal-substring deny patterns above were bypassable by splitting both
# the command and the protected filename via expansion, e.g.
# `$(echo tee) /tmp/h/hist$(printf ory).jsonl`. The new defense-in-depth
# check refuses any command that combines shell expansion with a reference
# to a .jsonl path or .dream_cursor.


@pytest.mark.parametrize(
    "command",
    [
        # Verified-live bypasses against the original patterns.
        "$(echo tee) /tmp/h/hist$(printf ory).jsonl <<< PWNED",
        "`printf tee` /tmp/h/hist`printf ory`.jsonl <<< PWNED",
        "${EDITOR:-tee} /tmp/h/history.jsonl <<< PWNED",
        # Variants that target the cursor file via expansion.
        "$(echo cp) src /tmp/.dream_cursor",
        "echo x > /tmp/$(echo .dream_cursor)",
    ],
)
def test_exec_blocks_expansion_targeting_protected_files(command):
    """Hardens #2989: expansion + .jsonl/.dream_cursor target must be blocked."""
    tool = ExecTool()
    result = tool._guard_command(command, "/tmp")
    assert result is not None
    assert "dangerous pattern" in result.lower()


@pytest.mark.parametrize(
    "command",
    [
        # Normal $() use that does not target nanobot state files.
        "echo $(date)",
        "cp src/$(basename dst).txt /tmp/",
        "tar czf backup-$(date +%s).tar.gz src/",
        "ls /tmp/$(whoami)",
        "echo ${HOME}/notes.md",
        "git log --since=$(date -d '1 week ago' +%Y-%m-%d)",
        # .log / .txt with expansion is fine — only .jsonl is protected.
        "tee /tmp/build-$(date +%s).log",
    ],
)
def test_exec_allows_legitimate_command_substitution(command):
    """Regression: innocent uses of $() / backticks / ${} must still execute."""
    tool = ExecTool()
    result = tool._guard_command(command, "/tmp")
    assert result is None, f"unexpectedly blocked: {command!r} → {result!r}"


# --- #2826: working_dir must not escape the configured workspace ---------


@pytest.mark.asyncio
async def test_exec_blocks_working_dir_outside_workspace(tmp_path):
    """An LLM-supplied working_dir outside the workspace must be rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = await tool.execute(command="rm calendar.ics", working_dir="/etc")
    assert "outside the configured workspace" in result


@pytest.mark.asyncio
async def test_exec_blocks_absolute_rm_via_hijacked_working_dir(tmp_path):
    """Regression for #2826: `rm /abs/path` via working_dir hijack."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim_dir = tmp_path / "outside"
    victim_dir.mkdir()
    victim = victim_dir / "file.ics"
    victim.write_text("data")

    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True)
    result = await tool.execute(
        command=f"rm {victim}",
        working_dir=str(victim_dir),
    )
    assert "outside the configured workspace" in result
    assert victim.exists(), "victim file must not have been deleted"


@pytest.mark.asyncio
async def test_exec_allows_working_dir_within_workspace(tmp_path):
    """A working_dir that is a subdirectory of the workspace is fine."""
    workspace = tmp_path / "workspace"
    subdir = workspace / "project"
    subdir.mkdir(parents=True)
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True, timeout=5)
    result = await tool.execute(command="echo ok", working_dir=str(subdir))
    assert "ok" in result
    assert "outside the configured workspace" not in result


@pytest.mark.asyncio
async def test_exec_allows_working_dir_equal_to_workspace(tmp_path):
    """Passing working_dir equal to the workspace root must be allowed."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True, timeout=5)
    result = await tool.execute(command="echo ok", working_dir=str(workspace))
    assert "ok" in result
    assert "outside the configured workspace" not in result


@pytest.mark.asyncio
async def test_exec_ignores_workspace_check_when_not_restricted(tmp_path):
    """Without restrict_to_workspace, the LLM may still choose any working_dir."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=False, timeout=5)
    result = await tool.execute(command="echo ok", working_dir=str(other))
    assert "ok" in result
    assert "outside the configured workspace" not in result
