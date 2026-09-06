"""Offline tests for the Canadian advisory agent configuration."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

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
            all(subagent["tools"] == [advisor.internet_search] for subagent in subagents)
        )

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
        self.assertIn("exactly one clearly labelled `Next step`", options["system_prompt"])


if __name__ == "__main__":
    unittest.main()
