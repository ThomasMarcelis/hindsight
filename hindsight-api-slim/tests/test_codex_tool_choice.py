"""
Regression tests for Codex provider tool_choice normalization.

The reflect agent forces tool selection via named tool_choice dicts on early iterations:
  {"type": "function", "function": {"name": "recall"}}

The Codex Responses API expects the function name at the top level instead:
  {"type": "function", "name": "recall"}

Without normalization, Codex rejects the request with:
  400 Unknown parameter: 'tool_choice.function'
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.codex_llm import CodexLLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Recall semantic memories",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def build_llm(reasoning_effort: str = "low") -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value="refresh"),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4-mini",
            reasoning_effort=reasoning_effort,
        )


@pytest.mark.asyncio
async def test_codex_proxy_mode_skips_local_oauth_load_and_refresh():
    with (
        patch.object(CodexLLM, "_load_codex_auth", side_effect=AssertionError("local OAuth should not be loaded")),
        patch.object(
            CodexLLM,
            "_load_codex_refresh_token",
            side_effect=AssertionError("local OAuth refresh token should not be loaded"),
        ),
    ):
        llm = CodexLLM(
            provider="openai-codex",
            api_key="proxy-token",
            base_url="http://127.0.0.1:8787/backend-api",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )

    assert llm.base_url == "http://127.0.0.1:8787/backend-api"
    assert llm._codex_proxy_mode is True
    assert llm.access_token == "proxy-token"
    assert llm.account_id == "codex-proxy"
    assert llm.refresh_token is None

    with patch.object(llm, "_refresh_oauth_tokens", new_callable=AsyncMock) as refresh:
        await llm._ensure_fresh_token()

    refresh.assert_not_called()


@pytest.mark.asyncio
async def test_codex_normalizes_legacy_named_tool_choice_shape():
    llm = build_llm()
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = (None, [])
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "recall the memory"}],
                tools=TOOLS,
                tool_choice={"type": "function", "function": {"name": "recall"}},
                max_retries=0,
            )
        sent_payload = mock_post.call_args.kwargs["json"]

    assert sent_payload["tool_choice"] == {"type": "function", "name": "recall"}


@pytest.mark.asyncio
async def test_codex_forced_tool_choice_still_yields_tool_calls():
    llm = build_llm()
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    tool_call = {"id": "call-1", "name": "recall", "arguments": {"query": "memory"}}
    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = (None, [tool_call])
            result = await llm.call_with_tools(
                messages=[{"role": "user", "content": "recall the memory"}],
                tools=TOOLS,
                tool_choice={"type": "function", "function": {"name": "recall"}},
                max_retries=0,
            )
        sent_payload = mock_post.call_args.kwargs["json"]

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "recall"
    assert sent_payload["tool_choice"] == {"type": "function", "name": "recall"}


@pytest.mark.asyncio
async def test_codex_call_sends_reasoning_effort_for_supported_levels():
    llm = build_llm(reasoning_effort="high")
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        with patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = "ok"
            await llm.call(messages=[{"role": "user", "content": "hello"}], max_retries=0)
        sent_payload = mock_post.call_args.kwargs["json"]

    assert sent_payload["reasoning"] == {"summary": "detailed", "effort": "high"}


@pytest.mark.asyncio
async def test_codex_call_sends_reasoning_effort_for_low():
    llm = build_llm(reasoning_effort="low")
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        with patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = "ok"
            await llm.call(messages=[{"role": "user", "content": "hello"}], max_retries=0)
        sent_payload = mock_post.call_args.kwargs["json"]

    assert sent_payload["reasoning"] == {"summary": "concise", "effort": "low"}


@pytest.mark.asyncio
async def test_codex_tool_call_sends_reasoning_effort_for_supported_levels():
    llm = build_llm(reasoning_effort="medium")
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = response
        with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = ("ok", [])
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "recall the memory"}],
                tools=TOOLS,
                max_retries=0,
            )
        sent_payload = mock_post.call_args.kwargs["json"]

    assert sent_payload["reasoning"] == {"summary": "auto", "effort": "medium"}
