"""Calling the Claude API safely: a fallback for strict tools, and errors in plain English.

Strict tools (``"strict": True``) make the API guarantee that the model's tool input matches
the schema, by compiling the schema into a grammar. A large schema can exceed the grammar's
size limit ("The compiled grammar is too large"), which happened to the client-form tool in
November 2026. Every tool's input is also checked in code (``profile.clean_draft``,
``spec.spec_from_draft``, ``tools.execute_tool``), so if the limit is hit the request is
retried once without strict mode instead of failing.
"""

import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

GRAMMAR_TOO_LARGE = "grammar is too large"


def without_strict(tools: list[dict]) -> list[dict]:
    """The same tools with strict mode off (inputs are then validated in code)."""
    return [{k: v for k, v in t.items() if k != "strict"} for t in tools]


def create_message(client: Any, *, tools: list[dict], **kwargs) -> tuple[Any, list[dict]]:
    """``client.messages.create`` with a one-time fallback when strict tools are too large.

    Returns the response and the tools actually used, so a multi-turn caller can keep sending
    the same tool definitions on later turns.
    """
    try:
        return client.messages.create(tools=tools, **kwargs), tools
    except anthropic.BadRequestError as e:
        strict = any(t.get("strict") for t in tools)
        if not strict or GRAMMAR_TOO_LARGE not in str(e).lower():
            raise
        logger.warning("Strict tool schemas too large for the API; retrying without strict mode")
        relaxed = without_strict(tools)
        return client.messages.create(tools=relaxed, **kwargs), relaxed


def friendly_error(error: Exception, task: str = "the AI request") -> str:
    """A message an adviser can act on, instead of a raw API error (details go to the log)."""
    logger.warning("%s failed: %r", task, error)
    if isinstance(error, ValueError):  # our own readable messages (e.g. no form was filled)
        return str(error)
    if isinstance(error, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return (
            f"{task.capitalize()} couldn't run: the AI service rejected the API key. The app's "
            "administrator needs to check it. You can still fill everything in by hand."
        )
    if isinstance(error, anthropic.RateLimitError):
        return f"The AI service is busy right now. Please try {task} again in a minute."
    if isinstance(error, anthropic.APITimeoutError | anthropic.APIConnectionError):
        return (
            f"Couldn't reach the AI service for {task}. Check the connection and try again; "
            "nothing was changed."
        )
    if isinstance(error, anthropic.InternalServerError) or (
        isinstance(error, anthropic.APIStatusError) and error.status_code >= 500
    ):
        return (
            f"The AI service is temporarily unavailable, so {task} didn't run. Please try "
            "again in a few minutes; nothing was changed."
        )
    if isinstance(error, anthropic.APIStatusError):
        return (
            f"The AI service couldn't process {task}, so nothing was changed. Try shorter or "
            "simpler wording, or fill the form by hand."
        )
    return f"Something went wrong with {task}, so nothing was changed. Please try again."
