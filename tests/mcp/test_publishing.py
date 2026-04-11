"""Tests for MCP Markdown publishing tools."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from telegram_acp_bot.mcp.tools.publishing import PublishError
from tests.mcp.support import STATE_FILE_ENV, TOKEN_ENV, mcp_channel, save_session_chat_map

TELEGRAPH_REQUEST_COUNT = 2


def test_render_markdown_as_telegraph_nodes_preserves_basic_structure():
    nodes = mcp_channel._render_markdown_as_telegraph_nodes(
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
            {"tag": "li", "children": ["one"]},
            {"tag": "li", "children": ["two"]},
        ],
    }


def test_publish_error_factories_return_meaningful_messages():
    assert str(PublishError.content_too_large()) == "published content exceeds the Telegraph 64 KB content limit"
    assert str(PublishError.malformed_response()) == "Telegraph request failed: malformed response payload"


def test_html_renderer_flattens_unsupported_tags_and_preserves_void_nodes():
    nodes = mcp_channel.TelegraphHTMLRenderer().render(
        "<section>hello<br/><img src='https://example.com/x.png'/></section>"
    )

    assert nodes == [
        {
            "tag": "p",
            "children": [
                "hello",
                {"tag": "br"},
                {"tag": "img", "attrs": {"src": "https://example.com/x.png"}},
            ],
        }
    ]


def test_html_renderer_ignores_unsupported_self_closing_tags():
    nodes = mcp_channel.TelegraphHTMLRenderer().render("<custom/>")

    assert nodes == []


def test_html_renderer_ignores_unmatched_end_tags():
    renderer = mcp_channel.TelegraphHTMLRenderer()

    renderer.handle_endtag("p")

    assert renderer.render("") == []


def test_render_markdown_as_telegraph_nodes_wraps_inline_root_nodes():
    wrapped = mcp_channel._wrap_root_inline_nodes(["hello", {"tag": "strong", "children": ["world"]}])

    assert wrapped == [{"tag": "p", "children": ["hello", {"tag": "strong", "children": ["world"]}]}]


def test_wrap_root_inline_nodes_flushes_before_block_nodes():
    wrapped = mcp_channel._wrap_root_inline_nodes(["hello", {"tag": "p", "children": ["world"]}])

    assert wrapped == [
        {"tag": "p", "children": ["hello"]},
        {"tag": "p", "children": ["world"]},
    ]


def test_format_published_message_includes_optional_summary():
    message = mcp_channel._format_published_message(
        title="Build report",
        url="https://telegra.ph/build-report",
        summary="Shared for easier reading.",
    )

    assert message == (
        "Published long-form page: Build report\nhttps://telegra.ph/build-report\n\nShared for easier reading."
    )


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
        return_value=mcp_channel._PublishedPage(
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
    huge_markdown = "a" * (mcp_channel.TELEGRAPH_CONTENT_LIMIT_BYTES + 1)

    with pytest.raises(RuntimeError, match="64 KB"):
        await mcp_channel._publish_markdown_page(
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

    page = await mcp_channel._publish_markdown_page(
        title="Build report",
        markdown="# Hello",
        author_name="Bot",
        author_url="https://example.com",
    )

    assert page == mcp_channel._PublishedPage(
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
        await mcp_channel._publish_markdown_page(
            title="Build report",
            markdown="# Hello",
            author_name="Bot",
            author_url=None,
        )


@pytest.mark.asyncio
async def test_telegraph_api_request_wraps_http_errors(mocker):
    client = mocker.AsyncMock()
    client.post.side_effect = mcp_channel.httpx.HTTPStatusError(
        "boom",
        request=mocker.Mock(),
        response=mocker.Mock(),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await mcp_channel.telegraph_api_request(
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
        await mcp_channel.telegraph_api_request(
            client,
            method="createPage",
            payload={"title": "Build report"},
        )


def test_normalize_telegraph_tag_rejects_unknown_values():
    assert mcp_channel._normalize_telegraph_tag("h1") == "h3"
    assert mcp_channel._normalize_telegraph_tag("section") is None


def test_is_root_inline_node_handles_non_wrappable_values():
    assert mcp_channel.is_root_inline_node("  ") is False
    assert mcp_channel.is_root_inline_node({"tag": "p", "children": ["hello"]}) is False
