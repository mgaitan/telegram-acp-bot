"""Tests for MCP Markdown publishing tools."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from telegram.error import TelegramError

from telegram_acp_bot.mcp.tools import publishing as publishing_module
from telegram_acp_bot.mcp.tools.publishing import (
    PublishedPage,
    PublishError,
    format_published_message,
    is_root_inline_node,
    render_markdown_as_telegraph_nodes,
    resolve_author_name,
    sanitize_telegraph_node,
    telegraph_api_request,
    wrap_root_inline_nodes,
)
from tests.mcp.support import STATE_FILE_ENV, TOKEN_ENV, mcp_channel, save_session_chat_map

TELEGRAPH_REQUEST_COUNT = 2


def test_render_markdown_as_telegraph_nodes_preserves_basic_structure():
    nodes = render_markdown_as_telegraph_nodes(
        "# Heading\n\nParagraph with **bold**, `code`, and [link](https://example.com).\n\n- one\n- two\n"
    )
    paragraph = cast(dict[str, object], nodes[1])
    paragraph_children = cast(list[object], paragraph["children"])

    assert nodes[0] == {"tag": "h3", "children": ["Heading"]}
    assert paragraph["tag"] == "p"
    assert {"tag": "strong", "children": ["bold"]} in paragraph_children
    assert {"tag": "code", "children": ["code"]} in paragraph_children
    assert {"tag": "a", "attrs": {"href": "https://example.com"}, "children": ["link"]} in paragraph_children
    assert nodes[2] == {
        "tag": "ul",
        "children": [
            {"tag": "li", "children": [{"tag": "p", "children": ["one"]}]},
            {"tag": "li", "children": [{"tag": "p", "children": ["two"]}]},
        ],
    }


def test_render_markdown_as_telegraph_nodes_sanitizes_unsafe_urls():
    nodes = render_markdown_as_telegraph_nodes(
        "[safe](https://example.com) [bad](javascript:alert('x')) ![ok](https://example.com/x.png) ![nope](file:///tmp/x.png)"
    )
    paragraph = cast(dict[str, object], nodes[0])
    paragraph_children = cast(list[object], paragraph["children"])

    assert paragraph_children == [
        {"tag": "a", "attrs": {"href": "https://example.com"}, "children": ["safe"]},
        {"tag": "a", "children": ["bad"]},
        {"tag": "img", "attrs": {"src": "https://example.com/x.png"}},
    ]


def test_render_markdown_as_telegraph_nodes_converts_strikethrough():
    nodes = render_markdown_as_telegraph_nodes("~~gone~~")

    assert nodes == [{"tag": "p", "children": [{"tag": "s", "children": ["gone"]}]}]


def test_sanitize_telegraph_node_drops_invalid_tag_payloads():
    invalid_node: dict[str, object] = {"tag": 1}

    assert sanitize_telegraph_node(invalid_node) == []


def test_sanitize_telegraph_node_flattens_unsupported_blocks():
    assert sanitize_telegraph_node({"tag": "section", "children": ["hello"]}) == ["hello"]


def test_sanitize_telegraph_node_handles_missing_or_invalid_attrs():
    assert sanitize_telegraph_node({"tag": "a", "attrs": None, "children": ["hello"]}) == [
        {"tag": "a", "children": ["hello"]}
    ]
    assert sanitize_telegraph_node({"tag": "a", "attrs": {"href": 42}, "children": ["hello"]}) == [
        {"tag": "a", "children": ["hello"]}
    ]


def test_publish_error_factories_return_meaningful_messages():
    assert str(PublishError.content_too_large()) == "published content exceeds the Telegraph 64 KB content limit"
    assert str(PublishError.malformed_response()) == "Telegraph request failed: malformed response payload"


def test_render_markdown_as_telegraph_nodes_wraps_inline_root_nodes():
    wrapped = wrap_root_inline_nodes(["hello", {"tag": "strong", "children": ["world"]}])

    assert wrapped == [{"tag": "p", "children": ["hello", {"tag": "strong", "children": ["world"]}]}]


def test_wrap_root_inline_nodes_flushes_before_block_nodes():
    wrapped = wrap_root_inline_nodes(["hello", {"tag": "p", "children": ["world"]}])

    assert wrapped == [
        {"tag": "p", "children": ["hello"]},
        {"tag": "p", "children": ["world"]},
    ]


def test_format_published_message_includes_optional_summary():
    message = format_published_message(
        title="Build report",
        url="https://telegra.ph/build-report",
        summary="Shared for easier reading.",
    )

    assert message == (
        "Published long-form page: Build report\nhttps://telegra.ph/build-report\n\nShared for easier reading."
    )


def test_resolve_author_name_strips_or_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(publishing_module.PUBLISH_AUTHOR_NAME_ENV, "Configured Bot")

    assert resolve_author_name("  Alice  ") == "Alice"
    assert resolve_author_name("   ") == publishing_module.DEFAULT_PUBLISH_AUTHOR_NAME
    assert resolve_author_name(None) == "Configured Bot"


def test_render_markdown_as_telegraph_nodes_keeps_safe_mailto_links():
    nodes = render_markdown_as_telegraph_nodes("[mail](mailto:bot@example.com)")

    assert nodes == [
        {
            "tag": "p",
            "children": [{"tag": "a", "attrs": {"href": "mailto:bot@example.com"}, "children": ["mail"]}],
        }
    ]


@pytest.mark.asyncio
async def test_publish_markdown_rejects_when_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.delenv(mcp_channel.ALLOW_PUBLISH_ENV, raising=False)

    result = await mcp_channel.telegram_publish_markdown(
        session_id="s1",
        title="Build report",
        markdown="# Hello",
    )

    assert result["ok"] is False
    assert "disabled by default" in str(result["error"])


@pytest.mark.asyncio
async def test_publish_markdown_rejects_blank_inputs(tmp_path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PUBLISH_ENV, "1")

    missing_title = await mcp_channel.telegram_publish_markdown(
        session_id="s1",
        title="   ",
        markdown="# Hello",
    )
    missing_markdown = await mcp_channel.telegram_publish_markdown(
        session_id="s1",
        title="Build report",
        markdown="   ",
    )

    assert missing_title["error"] == "title must not be empty"
    assert missing_markdown["error"] == "markdown must not be empty"


@pytest.mark.asyncio
async def test_publish_markdown_reports_context_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(mcp_channel.ALLOW_PUBLISH_ENV, "1")
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.delenv(STATE_FILE_ENV, raising=False)

    result = await mcp_channel.telegram_publish_markdown(
        session_id="s1",
        title="Build report",
        markdown="# Hello",
    )

    assert result["ok"] is False
    assert result["error"] == f"missing {TOKEN_ENV}"


@pytest.mark.asyncio
async def test_publish_markdown_sends_link_to_chat(tmp_path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PUBLISH_ENV, "1")
    bot = mocker.AsyncMock()
    bot.send_message.return_value = SimpleNamespace(message_id=77)
    mocker.patch("telegram_acp_bot.mcp.tools.publishing.Bot", return_value=bot)
    mocker.patch(
        "telegram_acp_bot.mcp.tools.publishing.publish_markdown_page",
        return_value=PublishedPage(
            path="Build-report-04-11",
            title="Build report",
            url="https://telegra.ph/Build-report-04-11",
        ),
    )

    result = await mcp_channel.telegram_publish_markdown(
        session_id="s1",
        title="Build report",
        markdown="# Hello",
        summary="Shared for easier reading.",
    )

    assert result == {
        "ok": True,
        "session_id": "s1",
        "chat_id": 123,
        "message_id": 77,
        "provider": "telegraph",
        "title": "Build report",
        "path": "Build-report-04-11",
        "url": "https://telegra.ph/Build-report-04-11",
    }
    bot.send_message.assert_awaited_once_with(
        chat_id=123,
        text=(
            "Published long-form page: Build report\n"
            "https://telegra.ph/Build-report-04-11\n\n"
            "Shared for easier reading."
        ),
        disable_web_page_preview=False,
    )


@pytest.mark.asyncio
async def test_publish_markdown_reports_delivery_errors(tmp_path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PUBLISH_ENV, "1")
    bot = mocker.AsyncMock()
    bot.send_message.side_effect = TelegramError("telegram is down")
    mocker.patch("telegram_acp_bot.mcp.tools.publishing.Bot", return_value=bot)
    mocker.patch(
        "telegram_acp_bot.mcp.tools.publishing.publish_markdown_page",
        return_value=PublishedPage(
            path="Build-report-04-11",
            title="Build report",
            url="https://telegra.ph/Build-report-04-11",
        ),
    )

    result = await mcp_channel.telegram_publish_markdown(
        session_id="s1",
        title="Build report",
        markdown="# Hello",
    )

    assert result == {
        "ok": False,
        "error": (
            "published page but failed to send Telegram message: telegram is down. "
            "Published URL: https://telegra.ph/Build-report-04-11"
        ),
        "url": "https://telegra.ph/Build-report-04-11",
    }


@pytest.mark.asyncio
async def test_publish_markdown_reports_provider_errors(tmp_path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PUBLISH_ENV, "1")
    mocker.patch(
        "telegram_acp_bot.mcp.tools.publishing.publish_markdown_page",
        side_effect=PublishError("boom"),
    )

    result = await mcp_channel.telegram_publish_markdown(
        session_id="s1",
        title="Build report",
        markdown="# Hello",
    )

    assert result["ok"] is False
    assert result["error"] == "boom"


@pytest.mark.asyncio
async def test_publish_markdown_page_rejects_content_that_is_too_large():
    huge_markdown = "a" * (publishing_module.TELEGRAPH_CONTENT_LIMIT_BYTES + 1)

    with pytest.raises(RuntimeError, match="64 KB"):
        await publishing_module.publish_markdown_page(
            title="Huge report",
            markdown=huge_markdown,
            author_name="Bot",
            author_url=None,
        )


@pytest.mark.asyncio
async def test_publish_markdown_page_calls_telegraph_api(mocker):
    account_response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"ok": True, "result": {"access_token": "token-123"}},
    )
    page_response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "ok": True,
            "result": {
                "path": "Build-report-04-11",
                "title": "Build report",
                "url": "https://telegra.ph/Build-report-04-11",
            },
        },
    )
    client = mocker.AsyncMock()
    client.post.side_effect = [account_response, page_response]
    client_cm = mocker.AsyncMock()
    client_cm.__aenter__.return_value = client
    client_cm.__aexit__.return_value = False
    mocker.patch("telegram_acp_bot.mcp.tools.publishing.httpx.AsyncClient", return_value=client_cm)

    page = await publishing_module.publish_markdown_page(
        title="Build report",
        markdown="# Hello",
        author_name="Bot",
        author_url="https://example.com",
    )

    assert page == PublishedPage(
        path="Build-report-04-11",
        title="Build report",
        url="https://telegra.ph/Build-report-04-11",
    )
    assert client.post.await_count == TELEGRAPH_REQUEST_COUNT


@pytest.mark.asyncio
async def test_publish_markdown_page_reports_api_errors(mocker):
    client = mocker.AsyncMock()
    client.post.return_value = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"ok": False, "error": "CONTENT_TOO_BIG"},
    )
    client_cm = mocker.AsyncMock()
    client_cm.__aenter__.return_value = client
    client_cm.__aexit__.return_value = False
    mocker.patch("telegram_acp_bot.mcp.tools.publishing.httpx.AsyncClient", return_value=client_cm)

    with pytest.raises(RuntimeError, match="CONTENT_TOO_BIG"):
        await publishing_module.publish_markdown_page(
            title="Build report",
            markdown="# Hello",
            author_name="Bot",
            author_url=None,
        )


@pytest.mark.asyncio
async def test_telegraph_api_request_wraps_http_errors(mocker):
    client = mocker.AsyncMock()
    client.post.side_effect = httpx.HTTPStatusError(
        "boom",
        request=mocker.Mock(),
        response=mocker.Mock(),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await telegraph_api_request(
            client,
            method="createPage",
            payload={"title": "Build report"},
        )


@pytest.mark.asyncio
async def test_telegraph_api_request_rejects_malformed_result(mocker):
    client = mocker.AsyncMock()
    client.post.return_value = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"ok": True, "result": []},
    )

    with pytest.raises(RuntimeError, match="malformed response payload"):
        await telegraph_api_request(
            client,
            method="createPage",
            payload={"title": "Build report"},
        )


@pytest.mark.asyncio
async def test_telegraph_api_request_rejects_non_json_payload(mocker):
    client = mocker.AsyncMock()
    client.post.return_value = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: (_ for _ in ()).throw(ValueError("no json")),
    )

    with pytest.raises(RuntimeError, match="malformed response payload"):
        await telegraph_api_request(
            client,
            method="createPage",
            payload={"title": "Build report"},
        )


@pytest.mark.asyncio
async def test_telegraph_api_request_rejects_non_object_payload(mocker):
    client = mocker.AsyncMock()
    client.post.return_value = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: ["not", "an", "object"],
    )

    with pytest.raises(RuntimeError, match="malformed response payload"):
        await telegraph_api_request(
            client,
            method="createPage",
            payload={"title": "Build report"},
        )


def test_is_root_inline_node_handles_non_wrappable_values():
    assert is_root_inline_node("  ") is False
    assert is_root_inline_node({"tag": "p", "children": ["hello"]}) is False
