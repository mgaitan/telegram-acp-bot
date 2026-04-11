"""Long-form Markdown publishing tool for the internal Telegram MCP server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlsplit

import httpx
from mcp.server.fastmcp import FastMCP
from md_to_telegraph import md_to_telegraph
from telegram import Bot
from telegram.error import TelegramError

from telegram_acp_bot.mcp.context import RequestContext, resolve_request_context

ALLOW_PUBLISH_ENV = "ACP_TELEGRAM_CHANNEL_ALLOW_PUBLISH"
PUBLISH_SHORT_NAME_ENV = "ACP_TELEGRAM_CHANNEL_PUBLISH_SHORT_NAME"
PUBLISH_AUTHOR_NAME_ENV = "ACP_TELEGRAM_CHANNEL_PUBLISH_AUTHOR_NAME"
PUBLISH_AUTHOR_URL_ENV = "ACP_TELEGRAM_CHANNEL_PUBLISH_AUTHOR_URL"
DEFAULT_PUBLISH_SHORT_NAME = "telegram-acp-bot"
DEFAULT_PUBLISH_AUTHOR_NAME = "telegram-acp-bot"
TELEGRAPH_API_BASE_URL = "https://api.telegra.ph"
TELEGRAPH_CONTENT_LIMIT_BYTES = 64 * 1024

ALLOWED_LINK_SCHEMES = {"http", "https", "mailto"}
ALLOWED_IMAGE_SCHEMES = {"http", "https"}
ALLOWED_TELEGRAPH_TAGS = {
    "a",
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
    "ul",
}
INLINE_TELEGRAPH_TAGS = {"a", "br", "code", "em", "img", "s", "strong"}
TAG_ALIASES = {"del": "s"}

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

    @classmethod
    def delivery_failed(cls, *, detail: str, url: str) -> PublishError:
        return cls(f"published page but failed to send Telegram message: {detail}. Published URL: {url}")


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

    def fail(error: str, **extra: object) -> dict[str, object]:
        return {"ok": False, "error": error, **extra}

    invalid_request = validate_publish_request(title=title, markdown=markdown, session_id=session_id)
    if invalid_request is not None:
        return fail(invalid_request)

    context = cast(RequestContext, resolve_request_context(session_id=session_id))

    resolved_author_name = resolve_author_name(author_name)
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
    try:
        sent = await bot.send_message(
            chat_id=context.chat_id,
            text=format_published_message(title=title, url=page.url, summary=summary),
            disable_web_page_preview=False,
        )
    except TelegramError as exc:
        return fail(str(PublishError.delivery_failed(detail=str(exc), url=page.url)), url=page.url)

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

    return wrap_root_inline_nodes(sanitize_telegraph_nodes(md_to_telegraph(markdown)))


def sanitize_telegraph_nodes(nodes: list[TelegraphNode]) -> list[TelegraphNode]:
    """Normalize tag names and discard unsafe Telegraph attributes."""

    sanitized: list[TelegraphNode] = []
    for node in nodes:
        sanitized.extend(sanitize_telegraph_node(node))
    return sanitized


def sanitize_telegraph_node(node: TelegraphNode) -> list[TelegraphNode]:
    """Normalize one Telegraph node into the subset this client allows."""

    if isinstance(node, str):
        return [node] if node.strip() else []

    tag = node.get("tag")
    if not isinstance(tag, str):
        return []
    normalized_tag = TAG_ALIASES.get(tag, tag)
    children = sanitize_telegraph_nodes(cast(list[TelegraphNode], node.get("children", [])))

    if normalized_tag == "img":
        return build_image_node(node)
    if normalized_tag not in ALLOWED_TELEGRAPH_TAGS:
        return children

    attrs = sanitize_attrs(node=node, tag=normalized_tag)
    sanitized: dict[str, object] = {"tag": normalized_tag}
    if children:
        sanitized["children"] = children
    if attrs:
        sanitized["attrs"] = attrs
    return [sanitized]


def sanitize_attrs(*, node: dict[str, object], tag: str) -> dict[str, str]:
    """Return only the Telegraph attributes this client allows for a node."""

    if tag != "a":
        return {}
    href = extract_allowed_url(node, attr_name="href", allowed_schemes=ALLOWED_LINK_SCHEMES)
    if href is None:
        return {}
    return {"href": href}


def extract_allowed_url(
    node: dict[str, object],
    *,
    attr_name: str,
    allowed_schemes: set[str],
) -> str | None:
    """Extract and validate one URL attribute from a Telegraph node."""

    attrs = node.get("attrs")
    if not isinstance(attrs, dict):
        return None
    attrs_map = cast(dict[str, object], attrs)
    value = attrs_map.get(attr_name)
    if not isinstance(value, str):
        return None
    return value if is_allowed_url(value, allowed_schemes=allowed_schemes) else None


def build_image_node(node: dict[str, object]) -> list[TelegraphNode]:
    """Return a sanitized Telegraph image node when the source URL is allowed."""

    src = extract_allowed_url(node, attr_name="src", allowed_schemes=ALLOWED_IMAGE_SCHEMES)
    if src is None:
        return []
    return [{"tag": "img", "attrs": {"src": src}}]


def is_allowed_url(value: str, *, allowed_schemes: set[str]) -> bool:
    """Return whether a URL uses an allowed scheme and has the required components."""

    parsed = urlsplit(value)
    if parsed.scheme not in allowed_schemes:
        return False
    if parsed.scheme == "mailto":
        return bool(parsed.path)
    return bool(parsed.netloc)


def wrap_root_inline_nodes(nodes: list[TelegraphNode]) -> list[TelegraphNode]:
    """Wrap stray inline nodes so the document remains Telegraph-friendly."""

    wrapped: list[TelegraphNode] = []
    paragraph_children: list[TelegraphNode] = []

    for node in nodes:
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


def is_root_inline_node(node: TelegraphNode) -> bool:
    """Return whether a root node should be wrapped into a paragraph."""

    if isinstance(node, str):
        return bool(node.strip())
    tag = node.get("tag")
    return isinstance(tag, str) and tag in INLINE_TELEGRAPH_TAGS


def resolve_author_name(author_name: str | None) -> str:
    """Resolve the Telegraph author name from tool input or environment defaults."""

    candidate = author_name or os.getenv(PUBLISH_AUTHOR_NAME_ENV, DEFAULT_PUBLISH_AUTHOR_NAME)
    normalized = candidate.strip()
    return normalized or DEFAULT_PUBLISH_AUTHOR_NAME


def validate_publish_request(*, title: str, markdown: str, session_id: str | None) -> str | None:
    """Validate publish inputs and return an error message when the request is invalid."""

    if not allow_page_publishing():
        return f"page publishing is disabled by default. Set {ALLOW_PUBLISH_ENV}=1 to enable it explicitly."
    if not title.strip():
        return "title must not be empty"
    if not markdown.strip():
        return "markdown must not be empty"

    context = resolve_request_context(session_id=session_id)
    return context if isinstance(context, str) else None


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

    try:
        decoded = response.json()
    except ValueError as exc:
        raise PublishError.malformed_response() from exc
    if not isinstance(decoded, dict):
        raise PublishError.malformed_response()
    if not decoded.get("ok"):
        error = str(decoded.get("error") or "unknown Telegraph error")
        raise PublishError.request_failed(error)
    result = decoded.get("result")
    if not isinstance(result, dict):
        raise PublishError.malformed_response()
    return cast(dict[str, object], result)


def allow_page_publishing() -> bool:
    """Return whether external long-form publishing is enabled."""

    value = os.getenv(ALLOW_PUBLISH_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}
