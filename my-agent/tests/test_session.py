"""Tests for Markdown-backed persistent advisory sessions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

import session


class _FakeAgent:
    """Return a fixed assistant response without a provider call."""

    def invoke(self, _input: object, *, config: object) -> dict[str, object]:
        """Return one assistant message."""
        return {"messages": [AIMessage(content="Saved response")]}


class PersistentSessionTests(unittest.TestCase):
    """Verify append and new-thread compaction behavior."""

    def test_turn_is_appended_to_markdown(self) -> None:
        """A completed user and assistant turn should be persisted."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory_path = root / "conversation.md"
            thread_path = root / "current_thread.md"
            archive_path = root / "conversation_archive.md"
            with (
                patch.object(session, "CONVERSATION_PATH", memory_path),
                patch.object(session, "CURRENT_THREAD_PATH", thread_path),
                patch.object(session, "CONVERSATION_ARCHIVE_PATH", archive_path),
                patch.object(
                    session, "create_advisory_agent", return_value=_FakeAgent()
                ),
            ):
                advisor = session.PersistentAdvisorSession(
                    "openai:test", thread_id="one"
                )
                advisor.ask("My question")

            content = thread_path.read_text(encoding="utf-8")
            self.assertIn("### User", content)
            self.assertIn("My question", content)
            self.assertIn("Saved response", content)
            self.assertNotIn("My question", memory_path.read_text(encoding="utf-8"))
            self.assertIn("My question", archive_path.read_text(encoding="utf-8"))

    def test_new_thread_replaces_transcript_with_summary(self) -> None:
        """A new session should compact and clear the previous transcript."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory_path = root / "conversation.md"
            thread_path = root / "current_thread.md"
            archive_path = root / "conversation_archive.md"
            memory_path.write_text("Previous summary", encoding="utf-8")
            thread_path.write_text("### User\n\nOld raw message\n", encoding="utf-8")
            with (
                patch.object(session, "CONVERSATION_PATH", memory_path),
                patch.object(session, "CURRENT_THREAD_PATH", thread_path),
                patch.object(session, "CONVERSATION_ARCHIVE_PATH", archive_path),
                patch.object(
                    session, "create_advisory_agent", return_value=_FakeAgent()
                ),
                patch.object(
                    session.PersistentAdvisorSession,
                    "_summarize",
                    return_value="Compact summary",
                ),
            ):
                session.PersistentAdvisorSession("openai:test", thread_id="two")

            content = memory_path.read_text(encoding="utf-8")
            self.assertIn("Compact summary", content)
            self.assertNotIn("Old raw message", content)
            thread = thread_path.read_text(encoding="utf-8")
            self.assertIn("Thread ID: `two`", thread)
            self.assertNotIn("Old raw message", thread)

    def test_empty_thread_preserves_summary_without_duplicate_heading(self) -> None:
        """Restarting an empty thread should preserve one summary section."""
        existing = session._render_memory("Existing summary")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory_path = root / "conversation.md"
            thread_path = root / "current_thread.md"
            archive_path = root / "conversation_archive.md"
            memory_path.write_text(existing, encoding="utf-8")
            with (
                patch.object(session, "CONVERSATION_PATH", memory_path),
                patch.object(session, "CURRENT_THREAD_PATH", thread_path),
                patch.object(session, "CONVERSATION_ARCHIVE_PATH", archive_path),
                patch.object(
                    session, "create_advisory_agent", return_value=_FakeAgent()
                ),
            ):
                session.PersistentAdvisorSession("openai:test", thread_id="two")

            content = memory_path.read_text(encoding="utf-8")
            self.assertEqual(content.count("## Summary of Earlier Threads"), 1)
            self.assertIn("Existing summary", content)


if __name__ == "__main__":
    unittest.main()
