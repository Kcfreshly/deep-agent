"""Offline tests for the Canadian advisory agent configuration."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage, ToolMessage

import advisor


class AdvisorConfigurationTests(unittest.TestCase):
    """Verify the agent architecture without calling external APIs."""

    def test_factory_configures_specialists_and_isolated_memory(self) -> None:
        """The factory should wire four specialists and expose only memory on disk."""
        with patch.object(
            advisor, "create_deep_agent", return_value=object()
        ) as create:
            advisor.create_advisory_agent("openai:test-model")

        options = create.call_args.kwargs
        subagents = options["subagents"]
        self.assertEqual(
            {subagent["name"] for subagent in subagents},
            {
                "admission-search-specialist",
                "admission-strategy-specialist",
                "canadian-immigration-specialist",
                "labour-settlement-specialist",
            },
        )
        backend = options["backend"]
        self.assertIsInstance(backend, CompositeBackend)
        self.assertIsInstance(backend.default, StateBackend)
        memory_backend = backend.routes["/memory/"]
        self.assertIsInstance(memory_backend, FilesystemBackend)
        self.assertEqual(memory_backend.cwd, advisor.MEMORY_ROOT.resolve())

    def test_all_memory_templates_exist(self) -> None:
        """Every configured semantic memory source should exist on disk."""
        self.assertTrue(
            all(
                (advisor.MEMORY_ROOT / filename).is_file()
                for filename in advisor.MEMORY_FILES
            )
        )

    def test_factory_configures_openai_hosted_search(self) -> None:
        """Only specialists should receive medium-context hosted web search."""
        with patch.object(
            advisor, "create_deep_agent", return_value=object()
        ) as create:
            advisor.create_advisory_agent("openai:test-model")

        options = create.call_args.kwargs
        self.assertEqual(options["tools"], [])
        self.assertTrue(
            all(
                subagent["tools"] == [advisor.OPENAI_WEB_SEARCH_TOOL]
                for subagent in options["subagents"]
            )
        )
        self.assertEqual(
            advisor.OPENAI_WEB_SEARCH_TOOL["search_context_size"], "medium"
        )

    def test_factory_configures_search_tool_for_deepseek(self) -> None:
        """DeepSeek specialists should receive search when a search key exists."""
        with (
            patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "test-key", "OPENAI_API_KEY": "test-key"},
            ),
            patch.object(advisor, "resolve_advisory_model", return_value=object()),
            patch.object(advisor, "create_deep_agent", return_value=object()) as create,
        ):
            advisor.create_advisory_agent("deepseek:deepseek-v4-flash")

        subagents = create.call_args.kwargs["subagents"]
        self.assertTrue(
            all(
                subagent["tools"] == [advisor.internet_search] for subagent in subagents
            )
        )

    def test_pdf_middleware_is_attached_to_coordinator_and_specialists(self) -> None:
        """All agents that can read files should get the PDF-only fallback."""
        with patch.object(
            advisor, "create_deep_agent", return_value=object()
        ) as create:
            advisor.create_advisory_agent("openai:test-model")

        options = create.call_args.kwargs
        self.assertTrue(
            any(
                isinstance(item, advisor.OpenAIPdfReadMiddleware)
                for item in options["middleware"]
            )
        )
        for subagent in options["subagents"]:
            self.assertTrue(
                any(
                    isinstance(item, advisor.OpenAIPdfReadMiddleware)
                    for item in subagent["middleware"]
                )
            )

    def test_pdf_read_is_extracted_by_openai_then_sent_as_text(self) -> None:
        """Only the PDF is sent to OpenAI; the downstream model gets extracted text."""

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps(
                    {"output_text": "Applicant name: Ada. Page 2."}
                ).encode()

        def fake_urlopen(request, timeout):
            self.assertEqual(request.full_url, advisor.OPENAI_RESPONSES_URL)
            self.assertEqual(timeout, 90)
            body = json.loads(request.data.decode())
            self.assertEqual(body["model"], "test-pdf-model")
            self.assertFalse(body["store"])
            file_part = body["input"][0]["content"][0]
            self.assertEqual(file_part["filename"], "transcript.pdf")
            self.assertTrue(
                file_part["file_data"].startswith("data:application/pdf;base64,")
            )
            return FakeResponse()

        request = ModelRequest(
            model=object(),  # type: ignore[arg-type]
            messages=[
                HumanMessage(content="Find my applicant name in the transcript."),
                ToolMessage(
                    content_blocks=[
                        {
                            "type": "file",
                            "mime_type": "application/pdf",
                            "base64": "JVBERi0=",
                        }
                    ],
                    name="read_file",
                    tool_call_id="read-pdf",
                    additional_kwargs={
                        "read_file_media_type": "application/pdf",
                        "read_file_path": "/memory/transcript.pdf",
                    },
                ),
            ],
        )
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "test-key", "OPENAI_PDF_MODEL": "test-pdf-model"},
            ),
            patch.object(advisor, "urlopen", side_effect=fake_urlopen) as openai_call,
        ):
            middleware = advisor.OpenAIPdfReadMiddleware()
            transformed = middleware.wrap_model_call(request, lambda updated: updated)

        openai_call.assert_called_once()
        pdf_message = transformed.messages[-1]
        self.assertIsInstance(pdf_message, ToolMessage)
        self.assertEqual(pdf_message.content_blocks[0]["type"], "text")
        self.assertIn("Applicant name: Ada", pdf_message.text)

    def test_non_pdf_messages_do_not_call_openai(self) -> None:
        """The PDF fallback leaves non-PDF tool results untouched."""
        request = ModelRequest(
            model=object(),  # type: ignore[arg-type]
            messages=[
                ToolMessage(
                    content="ordinary text", name="read_file", tool_call_id="read-text"
                )
            ],
        )
        with patch.object(advisor, "urlopen") as openai_call:
            transformed = advisor.OpenAIPdfReadMiddleware().wrap_model_call(
                request, lambda updated: updated
            )

        openai_call.assert_not_called()
        self.assertEqual(transformed.messages, request.messages)

    def test_pdf_without_openai_key_is_replaced_with_explanation(self) -> None:
        """Without a key, the PDF is not forwarded as an invalid DeepSeek file block."""
        request = ModelRequest(
            model=object(),  # type: ignore[arg-type]
            messages=[
                ToolMessage(
                    content_blocks=[
                        {
                            "type": "file",
                            "mime_type": "application/pdf",
                            "base64": "JVBERi0=",
                        }
                    ],
                    name="read_file",
                    tool_call_id="read-pdf",
                    additional_kwargs={
                        "read_file_media_type": "application/pdf",
                        "read_file_path": "/memory/transcript.pdf",
                    },
                )
            ],
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
            patch.object(advisor, "urlopen") as openai_call,
        ):
            transformed = advisor.OpenAIPdfReadMiddleware().wrap_model_call(
                request, lambda updated: updated
            )

        openai_call.assert_not_called()
        self.assertEqual(transformed.messages[0].content_blocks[0]["type"], "text")
        self.assertIn("set OPENAI_API_KEY", transformed.messages[0].text)

    def test_factory_runs_deepseek_without_search_key(self) -> None:
        """A DeepSeek-only setup should start, just without live web tools."""
        with (
            patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "test-key",
                    "OPENAI_API_KEY": "",
                    "TAVILY_API_KEY": "",
                },
            ),
            patch.object(advisor, "resolve_advisory_model", return_value=object()),
            patch.object(advisor, "create_deep_agent", return_value=object()) as create,
        ):
            advisor.create_advisory_agent("deepseek:deepseek-v4-flash")

        subagents = create.call_args.kwargs["subagents"]
        self.assertTrue(all(subagent["tools"] == [] for subagent in subagents))

    def test_openai_search_uses_small_configurable_model(self) -> None:
        """OpenAI-backed search should use the small default search model."""
        response = {
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {"type": "url", "url": "https://example.com/source"}
                        ]
                    },
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Search summary.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/source",
                                    "title": "Example",
                                }
                            ],
                        }
                    ],
                },
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 45)
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["model"], advisor.DEFAULT_OPENAI_SEARCH_MODEL)
            self.assertEqual(payload["tools"][0]["type"], "web_search")
            self.assertEqual(payload["tools"][0]["search_context_size"], "low")
            self.assertEqual(payload["tool_choice"], {"type": "web_search"})
            fake = FakeResponse()
            fake.read = lambda: json.dumps(response).encode("utf-8")
            return fake

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
            patch.object(advisor, "urlopen", side_effect=fake_urlopen),
        ):
            results = advisor.openai_web_search("latest DLI rules")

        self.assertEqual(results[0]["url"], "https://example.com/source")
        self.assertEqual(results[0]["content"], "Search summary.")

    def test_specialists_receive_only_relevant_memory(self) -> None:
        """Specialists should not receive the full semantic-memory collection."""
        with patch.object(
            advisor, "create_deep_agent", return_value=object()
        ) as create:
            advisor.create_advisory_agent("openai:test-model")

        for subagent in create.call_args.kwargs["subagents"]:
            middleware = subagent["middleware"][0]
            self.assertLess(len(middleware.sources), len(advisor.MEMORY_SOURCES))

    def test_factory_rejects_unsupported_model_provider(self) -> None:
        """Unsupported providers should fail with a clear configuration error."""
        with self.assertRaisesRegex(ValueError, "Supported model providers"):
            advisor.create_advisory_agent("anthropic:test-model")

    def test_deepseek_model_uses_compatible_non_thinking_mode(self) -> None:
        """The compatibility adapter should avoid reasoning replay requirements."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            model = advisor.resolve_advisory_model(
                "deepseek:deepseek-v4-flash", max_tokens=1_200
            )

        self.assertEqual(model.model_name, "deepseek-v4-flash")
        self.assertEqual(
            model.extra_body,
            {"thinking": {"type": "disabled"}, "max_tokens": 1_200},
        )

    def test_factory_enables_langsmith_tracing(self) -> None:
        """Advisory runs should override disabled tracing configuration."""
        tracing = {
            "LANGSMITH_TRACING": "false",
            "LANGSMITH_TRACING_V2": "false",
        }
        with (
            patch.dict(os.environ, tracing),
            patch.object(advisor, "create_deep_agent", return_value=object()),
        ):
            advisor.create_advisory_agent("openai:test-model")
            self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
            self.assertEqual(os.environ["LANGSMITH_TRACING_V2"], "true")

    def test_coordinator_tracks_and_advances_a_goal(self) -> None:
        """The coordinator should receive its goal ledger and next-step contract."""
        with patch.object(
            advisor, "create_deep_agent", return_value=object()
        ) as create:
            advisor.create_advisory_agent("openai:test-model")

        options = create.call_args.kwargs
        self.assertIn("/memory/goals.md", options["memory"])
        self.assertIn(
            "exactly one clearly labelled `Next step`", options["system_prompt"]
        )


if __name__ == "__main__":
    unittest.main()
