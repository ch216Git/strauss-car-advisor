"""Manual end-to-end smoke chat: the real agent loop, tools and engine, with real Gemini.

Uses real Gemini quota; it is not an automated test and never runs by itself.
Run from backend/:  python -m scripts.agent_chat      (type "exit" to quit)

Prints only the advisor's replies. With AGENT_DEBUG=1 it also prints, to stderr, timings (each
provider call, local tools per round, whole turn), tool names with ok / error code, the stop reason
and the conversation status - never keys, prompts, tool arguments or results, employee values or
provider internals.
"""

import os
import sys
import time
from collections.abc import Callable

from app.agent.agent import TurnResult, TurnStop, run_turn
from app.agent.llm import LLMProvider, ProviderError, Role
from app.agent.providers.gemini import GeminiProvider, GeminiSettings
from app.agent.state import ConversationState, conversation_status
from app.agent.tools import ToolContext
from app.env_file import load_env_files

DEBUG_VARIABLE = "AGENT_DEBUG"
EXIT_WORDS = {"exit", "quit", "יציאה"}
INTERNAL_ERROR_REPLY = "מצטער/ת, הייתה תקלה פנימית. אפשר לנסות שוב."


def chat(
    provider: LLMProvider,
    ctx: ToolContext,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    debug: Callable[[str], None] | None = None,
) -> ConversationState:
    """Read employee messages until "exit"; the conversation state is kept between turns."""
    state = ConversationState("cli")
    while True:
        try:
            text = read("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.casefold() in EXIT_WORDS:
            break
        if not text:
            continue
        try:
            turn = run_turn(state, text, provider, ctx)
        except Exception as e:  # never show a traceback: exception text may contain employee values
            write(INTERNAL_ERROR_REPLY)
            if debug:
                debug(f"[debug] internal error: {type(e).__name__}")
            continue
        if debug:
            for line in debug_lines(state, turn, ctx):
                debug(line)
        state = turn.state
        write(turn.reply)
    return state


def debug_lines(before: ConversationState, turn: TurnResult, ctx: ToolContext) -> list[str]:
    """Per round: provider time, local tool time, tool names with ok / error code. Then the turn summary.

    Timings, names, codes and status only - no arguments, results, employee values or prompts.
    """
    new_messages = turn.state.messages[len(before.messages):]
    calls = [m for m in new_messages if m.tool_calls]
    results = [m for m in new_messages if m.role == Role.TOOL]
    final_label = {
        TurnStop.MAX_ROUNDS: "tool calls not executed (round limit)",
        TurnStop.PROVIDER_ERROR: "provider error",
    }.get(turn.stop, "final reply")

    lines = []
    for number, timing in enumerate(turn.timings, start=1):
        line = f"[debug] round {number}: provider {timing.provider_seconds:.2f}s | "
        if timing.tool_seconds is None:
            line += final_label
        else:
            index = number - 1
            outcomes = [
                f"{call.name}={_outcome(result.content)}"
                for call, result in zip(calls[index].tool_calls, results[index].tool_results)
            ]
            line += f"tools {timing.tool_seconds:.2f}s: {', '.join(outcomes)}"
        lines.append(line)
    provider_total = sum(t.provider_seconds for t in turn.timings)
    tools_total = sum(t.tool_seconds or 0.0 for t in turn.timings)
    lines.append(
        f"[debug] turn: stop={turn.stop.value} tool_rounds={turn.tool_rounds} provider={provider_total:.2f}s "
        f"tools={tools_total:.2f}s total={turn.total_seconds:.2f}s "
        f"status={conversation_status(turn.state, ctx.rules).value}"
    )
    return lines


def _outcome(content) -> str:
    return "ok" if content.get("ok") else str(content.get("error", {}).get("code", "error"))


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
    load_env_files()
    try:
        provider = GeminiProvider(GeminiSettings.from_env())
    except ProviderError as e:
        print(f"Cannot start: {e}", file=sys.stderr)
        return 1
    debug = (lambda line: print(line, file=sys.stderr)) if os.environ.get(DEBUG_VARIABLE) == "1" else None
    started = time.perf_counter()
    ctx = ToolContext.load()  # rules, catalogue, CSV and car search index
    if debug:
        debug(f"[debug] startup: data loaded in {time.perf_counter() - started:.2f}s")
    chat(provider, ctx, debug=debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
