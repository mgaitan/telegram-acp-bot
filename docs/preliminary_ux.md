# Preliminary UX and implementation plan

This document captures an initial UX proposal for a Telegram-first ACP client.

## Goals

- Make Telegram feel like a natural remote console for coding agents.
- Keep chat interaction simple: users can type at any time and get streamed progress.
- Preserve a clean separation between Telegram transport concerns and ACP protocol concerns.
- Stay agent-agnostic (Codex, Claude Code, others) while supporting Codex-specific workflows.

## Product principles

- Direct interaction: normal text messages are prompts.
- Fast feedback: show progress incrementally, not only final answers.
- Safe by default: explicit permission prompts for risky tool calls.
- Recoverable sessions: users can stop, resume, inspect, and reset state.
- Operational clarity: users can always see session status and current workspace.

## Agreed decisions

- Session scope: one active ACP session per Telegram chat.
- Command style: slash commands only for control actions; no extra command prefixes.
- Shell usage: no explicit shell command in bot UX. Users ask in natural language and the agent decides whether to execute tools based on ACP permissions.
- Permissions: configurable policy (strict/manual by default, with optional auto-approve profiles).

## External references and extracted patterns

- ACP protocol and ecosystem: <https://agentclientprotocol.com/>
- ACP Python SDK (official): <https://agentclientprotocol.github.io/python-sdk/>
- ACP Python SDK examples: <https://github.com/agentclientprotocol/python-sdk/tree/main/examples>
- Telegram ACP bot reference (`tb3`): <https://github.com/niquola/tb3>
- Slack ACP bot reference (`juan`): <https://github.com/DiscreteTom/juan>
- Codex ACP adapter (`codex-acp`): <https://github.com/zed-industries/codex-acp>

Patterns worth adopting:

- One chat/thread maps to one ACP session and working directory (`tb3`, `juan`).
- Distinct control commands plus plain message forwarding (`tb3`, `juan`).
- Support file/image input and output (`juan`, `codex-acp`).
- Stream session updates (message chunks, tool calls, plan updates) from ACP SDK examples.
- Handle permission flows explicitly (`juan`, `codex-acp`, ACP SDK).

## UX model

### Conversation model

- A Telegram chat (or topic/thread when available) is bound to one active ACP session.
- Plain text messages are forwarded as `prompt` content blocks.
- Bot responses stream incrementally and are grouped as one logical assistant turn.
- Images sent by the user are forwarded as ACP image content blocks.

### Session model

- Session starts with `/new` (agent + workspace).
- Session can be paused (`/stop`) and resumed (`/resume`).
- Session can be fully reset (`/clear`).
- Users can inspect current session state (`/session`).

### Runtime visibility

- Show concise status line while running: `busy`, current tool, elapsed time.
- Expose `plan` updates as structured bullets.
- Expose tool call progress and final outputs.
- Large outputs and diffs are sent as files when they exceed Telegram text limits.

### Permission flow

- If ACP requests permissions, bot asks user with inline keyboard choices:
  - `Approve once`
  - `Approve for session`
  - `Reject`
- Time out pending requests if no answer after configurable interval.
- Keep audit note in chat for each permission decision.

## Proposed command set (v1)

### Core session commands

- `/start` show quick help and current state.
- `/help` show command reference.
- `/new <agent> [path]` create new session in workspace path.
- `/resume` resume last stopped session in this chat.
- `/stop` stop active agent process but preserve session metadata when possible.
- `/clear` remove active session state and start fresh.
- `/session` show active session details.
- `/cancel` cancel current running prompt.

### Agent configuration commands

- `/agents` list configured agents.
- `/mode [value]` show or set mode.
- `/model [value]` show or set model.
- `/cwd [path]` show or set working directory for the session.

### Inspection and file commands

- `/diff [git diff args]` show repo diff from current workspace.
- `/read <path>` read file and send excerpt or attachment.
- `/ls [path]` list directory safely.

### UX utility commands

- `/follow on|off` toggle verbose tool/progress updates.
- `/compact` ask agent to summarize and reduce context footprint.
- `/review [extra instructions]` ask agent for code review of current changes.

## Telegram interaction behaviors

- Any non-command text message is treated as prompt input.
- Photo/document attachments are accepted and included in prompt context.
- Replies to bot messages keep context in the same session.
- When users ask for operational actions (for example, run tests or create a commit), the bot forwards intent as prompt text and surfaces resulting tool outputs from ACP updates.
- Optional reaction/status markers:
  - processing: hourglass
  - done: check
  - error: cross

## Architecture split

## Layer 1: Telegram adapter (`telegram_acp_bot.telegram`)

Responsibilities:

- Receive Telegram updates and normalize user intents.
- Parse commands and attachments.
- Render outbound messages, files, keyboards, and status updates.
- Apply Telegram-specific constraints (rate limits, message length).

Should not know ACP internals beyond typed application-facing interfaces.

## Layer 2: ACP application service (`telegram_acp_bot.acp_app`)

Responsibilities:

- Session lifecycle (`initialize`, `new_session`, `prompt`, `cancel`).
- Agent process/connection management (stdio transport).
- Mapping ACP session updates into neutral app events.
- Permission request lifecycle and decision propagation.

Should not depend on Telegram-specific types.

## Layer 3: domain/state (`telegram_acp_bot.core`)

Responsibilities:

- Session registry (chat_id -> session state).
- Persistence strategy (in-memory first; optional file/db later).
- Policy decisions (default agent, allowed users, auto-approve rules).

## Async model

Recommended baseline:

- `python-telegram-bot` in async mode.
- ACP SDK async client/session APIs.
- One task group per active session for:
  - prompt execution,
  - streaming updates,
  - permission handling,
  - cancellation.

Design constraints:

- Prevent concurrent prompts in same session unless explicitly enabled.
- Avoid blocking handlers; all long actions must run in background tasks.
- Ensure graceful shutdown closes ACP subprocesses.

## Open UX decisions to iterate

- Diff presentation: inline snippets first, full patch as attachment by default?
- File reads: strict safe roots only (workspace) or configurable allowlist?
- Permission timeout and fallback behavior (auto-reject vs keep pending)?

## Implementation backlog (phased)

## Phase 0: project bootstrap

- Add runtime dependencies:
  - `python-telegram-bot`
  - `agent-client-protocol`
- Add config model for Telegram token, allowed users, agent definitions, and permission profiles.
- Add basic app entrypoint and logging setup.

## Phase 1: minimal end-to-end chat

- Implement Telegram polling bot with `/start`, `/help`, `/new`, `/session`.
- Spawn ACP agent process and run `initialize` + `new_session`.
- Forward plain text to `prompt` and stream text chunks back.
- Add tests for command parsing and session mapping.

## Phase 2: control and safety

- Add `/cancel`, `/stop`, `/clear`, `/resume`.
- Add busy-state guard per session.
- Implement permission prompt workflow with inline keyboard.
- Add tests for cancellation and permission handling.

## Phase 3: files and richer outputs

- Support incoming Telegram images as ACP image content blocks.
- Add `/read`, `/diff`, and large-output file attachments.
- Render plan updates and tool-call progress in Telegram-friendly format.
- Capture and present tool execution output when the agent performs commands requested in natural language.
- Add tests for output formatting and attachment thresholds.

## Phase 4: codex-focused polish (still generic core)

- Provide preconfigured Codex agent profile using `codex-acp`.
- Add `/review` and `/compact` convenience commands.
- Add docs with deployment examples (local machine, tmux/systemd).

## Non-goals for initial release

- Multi-tenant SaaS behavior.
- Complex RBAC beyond allowlist.
- Full historical replay UI.
- Advanced MCP server management UI inside Telegram.

## GitHub tracking

- Tracker issue: <https://github.com/mgaitan/telegram-acp-bot/issues/11>
- Foundation:
  - <https://github.com/mgaitan/telegram-acp-bot/issues/1>
  - <https://github.com/mgaitan/telegram-acp-bot/issues/2>
  - <https://github.com/mgaitan/telegram-acp-bot/issues/3>
  - <https://github.com/mgaitan/telegram-acp-bot/issues/4>
- Core UX:
  - <https://github.com/mgaitan/telegram-acp-bot/issues/5>
  - <https://github.com/mgaitan/telegram-acp-bot/issues/6>
  - <https://github.com/mgaitan/telegram-acp-bot/issues/7>
  - <https://github.com/mgaitan/telegram-acp-bot/issues/8>
- Codex polish:
  - <https://github.com/mgaitan/telegram-acp-bot/issues/9>
- Quality and docs:
  - <https://github.com/mgaitan/telegram-acp-bot/issues/10>
  - <https://github.com/mgaitan/telegram-acp-bot/issues/12>
