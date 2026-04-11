"""Long-form Markdown publishing tool for the internal Telegram MCP server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import cast

import httpx
from markdown_it import MarkdownIt
from mcp.server.fastmcp import FastMCP
from telegram import Bot

from telegram_acp_bot.mcp.context import resolve_request_context

ALLOW_PUBLISH_ENV = "ACP_TELEGRAM_CHANNEL_ALLOW_PUBLISH"
PUBLISH_SHORT_NAME_ENV = "ACP_TELEGRAM_CHANNEL_PUBLISH_SHORT_NAME"
PUBLISH_AUTHOR_NAME_ENV = "ACP_TELEGRAM_CHANNEL_PUBLISH_AUTHOR_NAME"
PUBLISH_AUTHOR_URL_ENV = "ACP_TELEGRAM_CHANNEL_PUBLISH_AUTHOR_URL"
DEFAULT_PUBLISH_SHORT_NAME = "telegram-acp-bot"
DEFAULT_PUBLISH_AUTHOR_NAME = "telegram-acp-bot"
TELEGRAPH_API_BASE_URL = "https://api.telegra.ph"
TELEGRAPH_CONTENT_LIMIT_BYTES = 64 * 1024

ALLOWED_TELEGRAPH_TAGS = {
    "a",
    "aside",
    "blockquote",
    "br",
    "code",
    "em",
    "h3",
    "h4",
    "hr",
    "img",
    "li",
    "ol",
    "p",
    "pre",
    "s",
    "strong",
    "u",
    "ul",
}
ALLOWED_TELEGRAPH_ATTRS = {"href", "src"}
INLINE_TELEGRAPH_TAGS = {"a", "br", "code", "em", "img", "s", "strong", "u"}
TAG_ALIASES = {
    "b": "strong",
    "del": "s",
    "h1": "h3",
    "h2": "h3",
    "h5": "h4",
    "h6": "h4",
    "i": "em",
    "strike": "s",
}

type TelegraphNode = str | dict[str, object]


@dataclass(frozen=True, slots=True)
class PublishedPage:
    path: str
    title: str
    url: str


class PublishError(RuntimeError):
    """Raised when external page publication fails."""

    @classmethod
    def content_too_large(cls) -> PublishError:
        return cls("published content exceeds the Telegraph 64 KB content limit")

    @classmethod
    def malformed_response(cls) -> PublishError:
        return cls("Telegraph request failed: malformed response payload")

    @classmethod
    def request_failed(cls, detail: str) -> PublishError:
        return cls(f"Telegraph request failed: {detail}")


class TelegraphHTMLRenderer(HTMLParser):
    """Convert sanitized Markdown HTML into Telegraph Node arrays."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._root: list[TelegraphNode] = []
        self._children_stack: list[list[TelegraphNode]] = [self._root]
        self._frame_opens_node: list[bool] = []

    def render(self, html: str) -> list[TelegraphNode]:
        self.feed(html)
        self.close()
        return strip_blank_text_nodes(wrap_root_inline_nodes(self._root))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = normalize_telegraph_tag(tag)
        if normalized is None:
            self._children_stack.append(self._children_stack[-1])
            self._frame_opens_node.append(False)
            return

        children: list[TelegraphNode] = []
        node: dict[str, object] = {"tag": normalized, "children": children}
        filtered_attrs = {key: value for key, value in attrs if key in ALLOWED_TELEGRAPH_ATTRS and value is not None}
        if filtered_attrs:
            node["attrs"] = filtered_attrs
        self._children_stack[-1].append(node)
        self._children_stack.append(children)
        self._frame_opens_node.append(True)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = normalize_telegraph_tag(tag)
        if normalized is None:
            return
        node: dict[str, object] = {"tag": normalized}
        filtered_attrs = {key: value for key, value in attrs if key in ALLOWED_TELEGRAPH_ATTRS and value is not None}
        if filtered_attrs:
            node["attrs"] = filtered_attrs
        self._children_stack[-1].append(node)

    def handle_endtag(self, tag: str) -> None:
        if not self._frame_opens_node:
            return
        self._frame_opens_node.pop()
        self._children_stack.pop()

    def handle_data(self, data: str) -> None:
        if data:
            self._children_stack[-1].append(data)


def register_publishing_tools(mcp: FastMCP) -> None:
    """Register long-form publishing tools on the provided MCP server."""

    mcp.tool(
        name="telegram_publish_markdown",
        description=(
            "Publish long Markdown content to an external Telegraph page and post the resulting URL "
            "back to the current Telegram chat. This is disabled by default and must be explicitly enabled."
        ),
    )(telegram_publish_markdown)


async def telegram_publish_markdown(
    title: str,
    markdown: str,
    session_id: str | None = None,
    summary: str | None = None,
    author_name: str | None = None,
) -> dict[str, object]:
    """Publish long Markdown content and send the resulting page URL back to Telegram."""

    def fail(error: str) -> dict[str, object]:
        return {"ok": False, "error": error}

    if not allow_page_publishing():
        return fail(f"page publishing is disabled by default. Set {ALLOW_PUBLISH_ENV}=1 to enable it explicitly.")

    if not title.strip():
        return fail("title must not be empty")
    if not markdown.strip():
        return fail("markdown must not be empty")

    context = resolve_request_context(session_id=session_id)
    if isinstance(context, str):
        return fail(context)

    resolved_author_name = author_name or os.getenv(PUBLISH_AUTHOR_NAME_ENV, DEFAULT_PUBLISH_AUTHOR_NAME).strip()
    resolved_author_url = os.getenv(PUBLISH_AUTHOR_URL_ENV, "").strip() or None

    try:
        page = await publish_markdown_page(
            title=title,
            markdown=markdown,
            author_name=resolved_author_name,
            author_url=resolved_author_url,
        )
    except PublishError as exc:
        return fail(str(exc))

    bot = Bot(token=context.token)
    sent = await bot.send_message(
        chat_id=context.chat_id,
        text=format_published_message(title=title, url=page.url, summary=summary),
        disable_web_page_preview=False,
    )
    return {
        "ok": True,
        "session_id": context.session_id,
        "chat_id": context.chat_id,
        "message_id": sent.message_id,
        "provider": "telegraph",
        "title": page.title,
        "path": page.path,
        "url": page.url,
    }


async def publish_markdown_page(
    *,
    title: str,
    markdown: str,
    author_name: str,
    author_url: str | None,
) -> PublishedPage:
    """Publish Markdown content as a Telegraph page."""

    short_name = os.getenv(PUBLISH_SHORT_NAME_ENV, DEFAULT_PUBLISH_SHORT_NAME).strip() or DEFAULT_PUBLISH_SHORT_NAME
    telegraph_nodes = render_markdown_as_telegraph_nodes(markdown)
    encoded_content = json.dumps(telegraph_nodes, ensure_ascii=False, separators=(",", ":"))
    if len(encoded_content.encode("utf-8")) > TELEGRAPH_CONTENT_LIMIT_BYTES:
        raise PublishError.content_too_large()

    async with httpx.AsyncClient(timeout=15.0) as client:
        account = await telegraph_api_request(
            client,
            method="createAccount",
            payload={
                "short_name": short_name,
                "author_name": author_name,
                "author_url": author_url or "",
            },
        )
        page = await telegraph_api_request(
            client,
            method="createPage",
            payload={
                "access_token": cast(str, account["access_token"]),
                "title": title,
                "author_name": author_name,
                "author_url": author_url or "",
                "content": encoded_content,
                "return_content": "false",
            },
        )

    return PublishedPage(
        path=cast(str, page["path"]),
        title=cast(str, page["title"]),
        url=cast(str, page["url"]),
    )


def render_markdown_as_telegraph_nodes(markdown: str) -> list[TelegraphNode]:
    """Render a Markdown string into a Telegraph-compatible node array."""

    parser = MarkdownIt("commonmark", {"html": False})
    html = parser.render(markdown)
    return TelegraphHTMLRenderer().render(html)


def format_published_message(*, title: str, url: str, summary: str | None) -> str:
    """Format the Telegram confirmation message for a published page."""

    lines = [f"Published long-form page: {title}", url]
    if summary and summary.strip():
        lines.extend(("", summary.strip()))
    return "\n".join(lines)


async def telegraph_api_request(
    client: httpx.AsyncClient,
    *,
    method: str,
    payload: dict[str, str],
) -> dict[str, object]:
    """Perform one Telegraph API request and return the `result` payload."""

    try:
        response = await client.post(f"{TELEGRAPH_API_BASE_URL}/{method}", data=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise PublishError.request_failed(str(exc)) from exc

    decoded = response.json()
    if not decoded.get("ok"):
        error = str(decoded.get("error") or "unknown Telegraph error")
        raise PublishError.request_failed(error)
    result = decoded.get("result")
    if not isinstance(result, dict):
        raise PublishError.malformed_response()
    return cast(dict[str, object], result)


def normalize_telegraph_tag(tag: str) -> str | None:
    """Map HTML tags into the subset accepted by Telegraph."""

    normalized = TAG_ALIASES.get(tag, tag)
    if normalized in ALLOWED_TELEGRAPH_TAGS:
        return normalized
    return None


def wrap_root_inline_nodes(nodes: list[TelegraphNode]) -> list[TelegraphNode]:
    """Wrap stray inline nodes so the document remains Telegraph-friendly."""

    wrapped: list[TelegraphNode] = []
    paragraph_children: list[TelegraphNode] = []

    for node in nodes:
        if isinstance(node, str) and not node.strip():
            continue
        if is_root_inline_node(node):
            paragraph_children.append(node)
            continue
        if paragraph_children:
            wrapped.append({"tag": "p", "children": paragraph_children})
            paragraph_children = []
        wrapped.append(node)

    if paragraph_children:
        wrapped.append({"tag": "p", "children": paragraph_children})
    return wrapped


def strip_blank_text_nodes(nodes: list[TelegraphNode]) -> list[TelegraphNode]:
    """Remove blank text nodes produced by HTML formatting whitespace."""

    cleaned: list[TelegraphNode] = []
    for node in nodes:
        if isinstance(node, str):
            if node.strip():
                cleaned.append(node)
            continue
        children = node.get("children")
        if isinstance(children, list):
            cleaned_node = {**node, "children": strip_blank_text_nodes(cast(list[TelegraphNode], children))}
            cleaned.append(cleaned_node)
            continue
        cleaned.append(node)
    return cleaned


def is_root_inline_node(node: TelegraphNode) -> bool:
    """Return whether a root node should be wrapped into a paragraph."""

    if isinstance(node, str):
        return bool(node.strip())
    tag = node.get("tag")
    return isinstance(tag, str) and tag in INLINE_TELEGRAPH_TAGS


def allow_page_publishing() -> bool:
    """Return whether external long-form publishing is enabled."""

    value = os.getenv(ALLOW_PUBLISH_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}
