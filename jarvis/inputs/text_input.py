"""Text input — simple async terminal readline loop."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class TextListener:
    """Reads lines from stdin and fires a callback for each non-empty line.

    Args:
        on_text: Callback called with the stripped input string.
    """

    def __init__(self, on_text: Callable[[str], None]) -> None:
        self.on_text = on_text

    async def run_async(self) -> None:
        """Run the text input loop (async). Exits on EOF or 'exit'/'quit'."""
        loop = asyncio.get_event_loop()
        logger.info("Text input ready. Type your message and press Enter.")
        while True:
            try:
                line: str = await loop.run_in_executor(None, input, "")
                line = line.strip()
                if not line:
                    continue
                if line.lower() in {"exit", "quit", "bye"}:
                    logger.info("Text input: exit signal received.")
                    break
                self.on_text(line)
            except (EOFError, KeyboardInterrupt):
                break

    def run(self) -> None:
        """Blocking version — runs the async loop synchronously."""
        asyncio.run(self.run_async())
