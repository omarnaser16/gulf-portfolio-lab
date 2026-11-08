"""Calling the API safely: the strict-tool fallback and plain-English errors (no network)."""

import anthropic
import httpx2
import pytest

from gpl.agent.api import create_message, friendly_error, without_strict
from test_agent import response, text


def _error(cls, status: int, message: str):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message, response=httpx2.Response(status, request=req), body=None)


TOO_LARGE = "The compiled grammar is too large, which would cause performance issues."


class Client:
    """Fails with the given errors first, then answers; records the tools of each request."""

    def __init__(self, errors: list) -> None:
        self.errors, self.tools = list(errors), []
        self.messages = self

    def create(self, **kwargs):
        self.tools.append(kwargs["tools"])
        if self.errors:
            raise self.errors.pop(0)
        return response([text("ok")], "end_turn")


STRICT = [{"name": "t", "strict": True, "input_schema": {"type": "object"}}]


def test_too_large_strict_grammar_retries_once_without_strict() -> None:
    client = Client([_error(anthropic.BadRequestError, 400, TOO_LARGE)])
    reply, tools = create_message(client, tools=STRICT, model="m", messages=[])
    assert reply.stop_reason == "end_turn"
    assert client.tools[0][0]["strict"] is True and "strict" not in client.tools[1][0]
    assert tools == without_strict(STRICT)  # later turns keep the relaxed tools


def test_other_errors_are_not_retried() -> None:
    client = Client([_error(anthropic.BadRequestError, 400, "messages: field required")])
    with pytest.raises(anthropic.BadRequestError):
        create_message(client, tools=STRICT, model="m", messages=[])
    relaxed = Client([_error(anthropic.BadRequestError, 400, TOO_LARGE)])
    with pytest.raises(anthropic.BadRequestError):  # nothing strict to relax
        create_message(relaxed, tools=without_strict(STRICT), model="m", messages=[])
    assert len(client.tools) == 1 and len(relaxed.tools) == 1


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_error(anthropic.AuthenticationError, 401, "invalid x-api-key"), "rejected the API key"),
        (_error(anthropic.RateLimitError, 429, "rate limit"), "busy right now"),
        (_error(anthropic.InternalServerError, 500, "boom"), "temporarily unavailable"),
        (_error(anthropic.BadRequestError, 400, TOO_LARGE), "couldn't process reading your goals"),
        (ValueError("The model did not fill the form."), "The model did not fill the form."),
    ],
)
def test_friendly_errors_hide_the_raw_api_message(error, expected) -> None:
    message = friendly_error(error, "reading your goals")
    assert expected in message
    assert "invalid_request_error" not in message and "Error code" not in message
