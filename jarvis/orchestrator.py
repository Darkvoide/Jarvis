"""JARVIS Orchestrator — the core agentic loop.

Manages the multi-turn conversation, calls Ollama with registered tools,
executes tool calls, feeds results back, and produces the final response.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import ollama

from jarvis.config import get_config
from jarvis.context import JarvisContext
from jarvis.system_prompt import build_system_prompt

logger = logging.getLogger(__name__)


class Orchestrator:
    """Drives the JARVIS agentic loop.

    Usage::

        ctx = JarvisContext()
        orch = Orchestrator(ctx, tools=[speak, search_web, ...])
        response = orch.handle("Open Chrome")

    The orchestrator keeps calling the LLM until it produces a plain
    text response (no more tool calls), implementing the standard
    "agentic loop" pattern.
    """

    MAX_TOOL_ROUNDS = 8  # guard against infinite loops

    def __init__(
        self,
        context: JarvisContext,
        tools: list[Callable],
    ) -> None:
        """Initialise the orchestrator.

        Args:
            context: Shared JarvisContext instance.
            tools: List of Python callables to register as tools.
                   Each must have a docstring and type hints — Ollama
                   auto-generates the JSON schema from these.
        """
        self.context = context
        self.tools = tools
        self.tool_map: dict[str, Callable] = {fn.__name__: fn for fn in tools}

        cfg = get_config()
        self.ollama_cfg = cfg.ollama

        # Inject the system prompt once
        tool_names = list(self.tool_map.keys())
        system_content = build_system_prompt(tool_names)
        context.append_raw({"role": "system", "content": system_content})

        logger.info(
            "Orchestrator ready — model=%s, tools=%s",
            self.ollama_cfg.model,
            ", ".join(tool_names),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def handle(self, user_input: str) -> str:
        """Process one user turn and return JARVIS's final reply.

        Args:
            user_input: Raw text from any input modality (voice transcript,
                        terminal text, or gesture-derived intent string).

        Returns:
            The plain-text response from JARVIS.
        """
        self._spoke_this_turn = False  # reset per-turn speech tracker
        self.context.add_message("user", user_input)
        logger.debug("User → %s", user_input)

        reply = self._agentic_loop()
        self.context.add_message("assistant", reply)
        logger.debug("JARVIS → %s", reply)
        return reply

    # ── Internal ───────────────────────────────────────────────────────────────

    def _agentic_loop(self) -> str:
        """Run the model → tool → model loop until a text reply is produced.

        Returns:
            Final text response from the model.
        """
        for round_num in range(self.MAX_TOOL_ROUNDS):
            messages = self.context.get_messages()

            try:
                response = ollama.chat(
                    model=self.ollama_cfg.model,
                    messages=messages,
                    tools=self.tools,
                    options={
                        "num_predict": 180,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                )
            except Exception as exc:
                logger.error("Ollama error: %s", exc)
                return f"I ran into a problem reaching the LLM: {exc}"

            msg = response.message

            # ── No tool calls → final answer ───────────────────────────────
            if not msg.tool_calls:
                return msg.content or "(no response)"

            # ── Append the assistant's tool-call turn ─────────────────────
            self.context.append_raw(msg)
            logger.debug("Round %d: %d tool call(s)", round_num + 1, len(msg.tool_calls))

            # ── Execute each tool call and append results ──────────────────
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args: dict[str, Any] = tool_call.function.arguments or {}

                result = self._execute_tool(fn_name, fn_args)

                # Track if the LLM explicitly called speak() so main.py won't double-speak
                if fn_name == "speak":
                    self._spoke_this_turn = True

                self.context.append_raw(
                    {
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        # Exceeded max rounds — give up gracefully
        logger.warning("Max tool rounds (%d) exceeded.", self.MAX_TOOL_ROUNDS)
        return "I wasn't able to complete that in a reasonable number of steps. Please try rephrasing."


    def _execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Look up and call a tool by name.

        Args:
            name: Tool function name.
            args: Keyword arguments for the tool.

        Returns:
            Tool's return value, or an error dict on failure.
        """
        fn = self.tool_map.get(name)
        if fn is None:
            logger.warning("Unknown tool requested: %s", name)
            return {"error": f"Tool '{name}' is not registered."}

        logger.info("Calling tool: %s(%s)", name, args)
        try:
            result = fn(**args)
            logger.debug("Tool %s result: %s", name, result)
            return result
        except TypeError as exc:
            logger.error("Tool %s argument error: %s", name, exc)
            return {"error": f"Invalid arguments for '{name}': {exc}"}
        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc)
            return {"error": f"Tool '{name}' raised an exception: {exc}"}
