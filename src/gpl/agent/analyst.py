"""Tool-use loop and investment memo writer.

The model decides which tools to call and writes the memo; the numbers come only from the
tools. Three safeguards make that auditable:

1. The system prompt requires every figure to be copied from a tool result with the tool's
   name in brackets.
2. Every tool call (name, input, output) is logged and shown in the dashboard.
3. ``unverified_numbers`` lists any number in the answer that no tool returned, so an
   invented or re-computed figure is caught by code, not by trust.

The "not investment advice" disclaimer is appended by code, so it is always present.
"""

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from gpl.agent.tools import AnalystContext, ToolError, execute_tool, tool_definitions

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DISCLAIMER = (
    "*This memo was produced by an automated analysis of historical data for educational "
    "purposes. It is not investment advice. Past performance does not predict future results.*"
)

SYSTEM_PROMPT = """\
You are the analyst inside Gulf Portfolio Lab, an educational tool for GCC-based and global \
investors. You have tools that run a tested Python finance engine on US-listed ETFs priced in \
USD: GCC equity (KSA, UAE, QAT), global equity (SPY, EFA, EEM), bonds (AGG, TLT) and gold (GLD).

Rules for numbers - these matter more than anything else:
- Never calculate, estimate, round, convert, average or combine numbers yourself.
- Every number you write must be copied exactly as it appears in a tool result from this \
conversation, followed by the tool name in square brackets, e.g. "Sharpe 0.49 \
[backtest_strategies]".
- If you want a figure that no tool has returned, call a tool that provides it, or describe the \
point in words without a number. Do not use outside knowledge for any figure.

When the user describes an investor profile, call the tools you need (start with get_universe; \
use backtest_strategies for the out-of-sample record, simulate_future for the horizon and goal, \
get_regime_view for current conditions), then write a memo in markdown with these sections:
1. **Profile** - restate the investor's situation in one or two sentences.
2. **Suggested allocation** - pick one strategy that fits the risk tolerance and give its latest \
weights. As a guide: conservative -> Min variance or Risk parity; moderate -> Risk parity, Equal \
weight or 60/40 (SPY/AGG); aggressive -> Max Sharpe or SPY. Explain the choice using tool numbers.
3. **Expected risk and return** - backtest CAGR, volatility, Sharpe and max drawdown, compared \
with one or two alternatives.
4. **Outlook for their horizon** - simulate_future results for their horizon and goal.
5. **Current market regime** - the regime view, including its modest track record.
6. **Risks and limitations** - sample period, short GCC ETF history, hindsight in \
optimise_portfolio, regime whipsaw, and anything specific to this investor.
Use "###" for section headings and no title above them. Keep the memo under 600 words. For \
currency: all figures are in USD; SAR, AED, QAR, BHD and OMR \
are pegged to the US dollar, so a Gulf investor in these USD assets carries little currency risk, \
while the Kuwaiti dinar is pegged to a basket.

When the user asks about their own portfolio ("My portfolio"), call get_my_portfolio, then \
backtest_strategies with "My portfolio" and the alternatives, and simulate_future for "My \
portfolio". Use the same sections, replacing "Suggested allocation" with "Your portfolio": its \
holdings, how concentrated its risk is (share of risk vs weight), and how its record compares \
with the alternatives. Describe what the tools show; do not tell the user to buy or sell anything.

When asked to explain an optimisation (the Optimise section of the Portfolio tab), call \
get_portfolio_diagnosis and get_optimisation_result, then write a memo with "###" sections:
- **Diagnosis** - where the current portfolio's risk comes from (share of risk vs weight, \
clusters of holdings that move together) and how it compares with the benchmark.
- **Your goal** - the goal, any target and the limits, in plain words.
- **Candidates** - the methods tried and their walk-forward results after costs, compared \
with the current portfolio and the benchmark; say which did not hold up out of sample. \
"Months limits unmet" are months when the limits could not be met with the data then \
available, so the previous weights were kept: no limit was broken.
- **Recommendation and why** - the recommendation exactly as the tool gives it (it is chosen \
by a fixed rule in code, not by you), quoting the rule and its reasons, and the largest \
weight changes. If the tool recommends keeping the current portfolio, say so and why.
- **Trade-offs** - what the recommended method gives up (e.g. return, turnover, drawdown).
- **Limitations** - history-based, risk-adjusted results; choosing among methods on the same \
period; estimates of expected returns are noisy.
Never choose a different method, propose weights or calculate anything yourself. Never say or \
imply that a method will earn higher returns: describe results as risk-adjusted and \
historical, not a guarantee.

For follow-up questions, answer briefly using tools as needed, with the same rules for numbers. \
Do not give personalised financial advice beyond describing what the tools show.
"""

_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:%|×|x\b)?")


def _numbers(text: str, with_units_only: bool) -> set[float]:
    """Numeric values in ``text``. With ``with_units_only``, ignore bare integers (years etc.)."""
    values = set()
    for match in _NUMBER.finditer(text):
        token = match.group().replace(",", "").replace(" ", "")
        has_unit = token.endswith(("%", "×", "x"))
        has_decimal = "." in token
        if with_units_only and not (has_unit or has_decimal):
            continue
        values.add(round(float(token.rstrip("%×x")), 6))
    return values


def unverified_numbers(answer: str, sources: Iterable[str]) -> list[str]:
    """Numbers in ``answer`` (decimals, percentages, multiples) that appear in no source.

    ``sources`` are the tool outputs (and user messages) the answer may quote. Values are
    compared numerically, so "1.5×" matches "1.50×" but a re-rounded "15.3%" does not match
    "15.27%".
    """
    allowed: set[float] = set()
    for source in sources:
        # Only figures with a unit or decimals can be quoted: otherwise a date such as
        # "2026-09-30" would whitelist "30%".
        allowed |= _numbers(source, with_units_only=True)
    flagged = []
    for match in _NUMBER.finditer(answer):
        token = match.group().strip()
        values = _numbers(token, with_units_only=True)
        if values and not values <= allowed:
            flagged.append(token)
    return flagged


def make_client() -> anthropic.Anthropic:
    """Anthropic client; loads ``ANTHROPIC_API_KEY`` from the project's ``.env`` if present."""
    load_dotenv(PROJECT_ROOT / ".env")
    return anthropic.Anthropic()


def has_credentials() -> bool:
    load_dotenv(PROJECT_ROOT / ".env")
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict
    output: dict | str
    is_error: bool = False


@dataclass
class AnalystReply:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    unverified: list[str] = field(default_factory=list)


class Analyst:
    """A conversation with the model; each ``ask`` runs the tool loop until it answers.

    Parameters
    ----------
    ctx : AnalystContext
        Data the tools run on.
    client : anthropic.Anthropic
        API client (a stub in tests).
    model : str
        Model ID, from config (cheap by default).
    max_tokens, effort, max_tool_rounds
        Response cap, reasoning effort, and a safety limit on tool round-trips per question.
    """

    def __init__(
        self,
        ctx: AnalystContext,
        client: Any,
        model: str,
        max_tokens: int = 16_000,
        effort: str = "medium",
        max_tool_rounds: int = 8,
    ) -> None:
        self.ctx, self.client, self.model = ctx, client, model
        self.max_tokens, self.effort, self.max_tool_rounds = max_tokens, effort, max_tool_rounds
        self.tools = tool_definitions(ctx)
        self.messages: list[dict] = []
        self.tool_log: list[ToolCall] = []
        self._user_texts: list[str] = []

    def ask(self, question: str) -> AnalystReply:
        self.messages.append({"role": "user", "content": question})
        self._user_texts.append(question)
        calls: list[ToolCall] = []
        input_tokens = output_tokens = 0

        for _ in range(self.max_tool_rounds + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                output_config={"effort": self.effort},
                messages=self.messages,
            )
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            # Keep every block (including thinking) so the next request is valid.
            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                call = self._run(block)
                calls.append(call)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(call.output) if not call.is_error else call.output,
                        "is_error": call.is_error,
                    }
                )
            self.messages.append({"role": "user", "content": results})  # all results, one message
        else:
            return self._reply(
                "The analyst stopped after too many tool calls without an answer. Try a more "
                "specific question.",
                calls,
                "max_tool_rounds",
                input_tokens,
                output_tokens,
            )

        text = "\n\n".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "refusal":
            text = "The model declined to answer this request."
        elif response.stop_reason == "max_tokens":
            text += "\n\n*(Answer cut off at the length limit.)*"
        return self._reply(text, calls, response.stop_reason, input_tokens, output_tokens)

    def _run(self, block: Any) -> ToolCall:
        try:
            output = execute_tool(self.ctx, block.name, dict(block.input))
            call = ToolCall(block.id, block.name, dict(block.input), output)
        except ToolError as e:
            call = ToolCall(block.id, block.name, dict(block.input), f"Error: {e}", is_error=True)
        self.tool_log.append(call)
        return call

    def _reply(
        self, text: str, calls: list[ToolCall], stop: str, tokens_in: int, tokens_out: int
    ) -> AnalystReply:
        sources = [json.dumps(c.output) for c in self.tool_log] + self._user_texts
        unverified = unverified_numbers(text, sources)
        if "not investment advice" not in text.lower():
            text = f"{text}\n\n---\n{DISCLAIMER}"
        return AnalystReply(text, calls, stop, tokens_in, tokens_out, unverified)


def estimate_cost(reply: AnalystReply, prices_per_mtok: Mapping[str, float]) -> float:
    """USD cost of one reply from token counts and per-million-token prices."""
    return (
        reply.input_tokens * prices_per_mtok["input"]
        + reply.output_tokens * prices_per_mtok["output"]
    ) / 1e6
