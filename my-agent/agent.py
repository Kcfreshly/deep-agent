"""Run a Deep Agent from the command line."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from advisor import DEFAULT_MODEL
from session import PersistentAdvisorSession


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Task for the agent (prompted interactively when omitted)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"LangChain provider:model identifier (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Create the Deep Agent and run one task."""
    args = parse_args()
    prompt = " ".join(args.prompt).strip() or input("Prompt: ").strip()
    if not prompt:
        msg = "A prompt is required."
        raise SystemExit(msg)
    session = PersistentAdvisorSession(args.model)
    print(f"Agent: {session.ask(prompt)}")


if __name__ == "__main__":
    main()
