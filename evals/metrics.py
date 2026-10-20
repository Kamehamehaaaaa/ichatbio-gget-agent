"""
Metrics for the gget agent regression suite.

Deterministic (no LLM, always run):
  - ToolCorrectnessMetric (DeepEval built-in): the expected tools were called, in order.
  - ToolArgumentsMetric: the expected tools got the right arguments.
  - OutcomeMetric: ended with finish/abort as expected, no forbidden tools, no crash, no runaway loop.
  - AnswerContainsMetric: the final answer mentions the key facts.
  - GroundedNumbersMetric: numbers in the final answer appear in tool outputs (catches invented figures).

LLM-judged (only when EVAL_JUDGE_MODEL is set):
  - GEval "Answer quality" against each case's expected_answer.
"""

import os
import re

from deepeval.metrics import BaseMetric, GEval, ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

# ── Judge models ──────────────────────────────────────────────────────────────


class NoJudge(DeepEvalBaseLLM):
    """Placeholder so DeepEval doesn't try to build an OpenAI client for deterministic metrics."""

    def __init__(self):
        pass

    def load_model(self):
        return None

    def generate(self, *args, **kwargs) -> str:
        raise RuntimeError("No judge model configured (set EVAL_JUDGE_MODEL).")

    async def a_generate(self, *args, **kwargs) -> str:
        return self.generate()

    def get_model_name(self) -> str:
        return "no-judge"


class OpenAICompatibleJudge(DeepEvalBaseLLM):
    """Any OpenAI-compatible chat endpoint (e.g. UF NaviGator) as the DeepEval judge."""

    def __init__(self, model: str, base_url: str | None, api_key: str | None):
        from openai import AsyncOpenAI, OpenAI

        self.model_name = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def load_model(self):
        return self.client

    def _messages(self, prompt: str):
        return [
            {"role": "system", "content": "You are an evaluation judge. When asked for JSON, reply with JSON only."},
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str, *args, **kwargs) -> str:
        r = self.client.chat.completions.create(model=self.model_name, messages=self._messages(prompt), temperature=0)
        return r.choices[0].message.content or ""

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        r = await self.aclient.chat.completions.create(
            model=self.model_name, messages=self._messages(prompt), temperature=0
        )
        return r.choices[0].message.content or ""

    def get_model_name(self) -> str:
        return self.model_name


def get_judge() -> DeepEvalBaseLLM | None:
    """Judge from EVAL_JUDGE_MODEL / EVAL_JUDGE_BASE_URL / EVAL_JUDGE_API_KEY, or None if unset."""
    model = os.getenv("EVAL_JUDGE_MODEL")
    if not model:
        return None
    return OpenAICompatibleJudge(model, os.getenv("EVAL_JUDGE_BASE_URL"), os.getenv("EVAL_JUDGE_API_KEY"))


# ── Argument matching ─────────────────────────────────────────────────────────


def arg_matches(expected, actual) -> bool:
    """
    Expected-value rules (from cases.yaml):
      "$present"           key present and non-empty
      "$absent"            key missing, None or empty
      "text"               case-insensitive substring of the actual value (lists/dicts are stringified)
      true / 5 / 1.5       exact equality
      [a, b]               every item matches somewhere in the actual value
      {any_of: [a, b]}     at least one alternative matches
      {key: rule, ...}     nested dict: every key matches (e.g. settings, filters)
    """
    if expected == "$present":
        return actual not in (None, "", [], {})
    if expected == "$absent":
        return actual in (None, "", [], {})
    if isinstance(expected, dict) and set(expected) == {"any_of"}:
        return any(arg_matches(alt, actual) for alt in expected["any_of"])
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(arg_matches(v, actual.get(k)) for k, v in expected.items())
    if isinstance(expected, list):
        return all(arg_matches(item, actual) for item in expected)
    if isinstance(expected, bool) or isinstance(expected, (int, float)):
        if isinstance(actual, str):
            return actual.strip().lower() == str(expected).lower()
        return actual == expected
    if isinstance(expected, str):
        return actual is not None and expected.lower() in str(actual).lower()
    return actual == expected


# ── Custom deterministic metrics ──────────────────────────────────────────────


class _DeterministicMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.include_reason = True
        self.async_mode = False
        self.strict_mode = False
        self.evaluation_model = None

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        self.success = self.score is not None and self.score >= self.threshold
        return self.success


class ToolArgumentsMetric(_DeterministicMetric):
    """Each expected tool with `args` was called at least once with matching arguments."""

    def __init__(self, expected: list[dict], threshold: float = 1.0):
        super().__init__(threshold)
        self.expected = [e for e in expected if e.get("args")]

    @property
    def __name__(self):
        return "Tool arguments"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        if not self.expected:
            self.score, self.reason = 1.0, "No argument constraints for this case."
            return self.score
        called = test_case.tools_called or []
        failures = []
        for exp in self.expected:
            calls = [c for c in called if c.name == exp["name"]]
            if not calls:
                failures.append(f"{exp['name']} was never called")
            elif not any(all(arg_matches(v, (c.input_parameters or {}).get(k)) for k, v in exp["args"].items()) for c in calls):
                got = "; ".join(str(c.input_parameters) for c in calls)[:400]
                failures.append(f"{exp['name']} args {exp['args']} not matched (got {got})")
        self.score = 1 - len(failures) / len(self.expected)
        self.reason = "All argument constraints met." if not failures else " | ".join(failures)
        return self.score


class OutcomeMetric(_DeterministicMetric):
    """Ended the expected way, avoided forbidden tools, no crash, no runaway tool loop."""

    def __init__(self, expected_outcome: str, forbidden: list[str], max_calls: int, run):
        super().__init__(1.0)
        self.expected_outcome, self.forbidden, self.max_calls, self.run = expected_outcome, forbidden, max_calls, run

    @property
    def __name__(self):
        return "Outcome"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        problems = []
        if self.run.error:
            problems.append(f"agent error: {self.run.error}")
        outcome = self.run.outcome
        if self.expected_outcome != "either" and outcome != self.expected_outcome:
            problems.append(f"ended with '{outcome}', expected '{self.expected_outcome}'")
        if self.expected_outcome == "either" and outcome not in ("finish", "abort"):
            problems.append(f"ended with '{outcome}', expected finish or abort")
        used = {c.name for c in self.run.calls}
        bad = sorted(used & set(self.forbidden))
        if bad:
            problems.append(f"called forbidden tool(s): {', '.join(bad)}")
        n = len(self.run.work_calls)
        if n > self.max_calls:
            problems.append(f"{n} tool calls (max {self.max_calls}); possible loop")
        self.score = 0.0 if problems else 1.0
        self.reason = "; ".join(problems) or f"Ended with '{outcome}' after {n} tool call(s)."
        return self.score


class AnswerContainsMetric(_DeterministicMetric):
    """Final answer mentions each required fact (an item may be a list of acceptable alternatives)."""

    def __init__(self, required: list, threshold: float = 1.0):
        super().__init__(threshold)
        self.required = required

    @property
    def __name__(self):
        return "Answer contains"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        if not self.required:
            self.score, self.reason = 1.0, "No required facts for this case."
            return self.score
        answer = (test_case.actual_output or "").lower()
        missing = []
        for item in self.required:
            options = item if isinstance(item, list) else [item]
            if not any(str(o).lower() in answer for o in options):
                missing.append(" / ".join(map(str, options)))
        self.score = 1 - len(missing) / len(self.required)
        self.reason = "All required facts present." if not missing else f"Missing: {'; '.join(missing)}"
        return self.score


# In the ANSWER, only standalone numbers count: skips digits inside IDs like ENSG00000130234, NM_000546.6, hg38
_NUMBER = re.compile(r"(?<![\w.\-/])\d[\d,]*(?:\.\d+)?(?![\w])")
# In TOOL OUTPUTS, any digit run counts, so "chromosome 11" in an answer is grounded by "chr11" in a tool output
_ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str, pattern: re.Pattern = _NUMBER) -> list[float]:
    out = []
    for m in pattern.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


class GroundedNumbersMetric(_DeterministicMetric):
    """
    Share of numbers in the final answer that also appear in a tool output (or the request).
    Ignores small integers (<= 10) such as "top 5". Flags invented percentages, lengths and counts.
    """

    def __init__(self, run, threshold: float = 0.8):
        super().__init__(threshold)
        self.run = run

    @property
    def __name__(self):
        return "Grounded numbers"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        answer_numbers = [n for n in _numbers(test_case.actual_output) if not (n.is_integer() and n <= 10)]
        if not answer_numbers:
            self.score, self.reason = 1.0, "No numbers to check."
            return self.score
        source = _numbers(self.run.tool_outputs_text(), _ANY_NUMBER) + _numbers(self.run.request, _ANY_NUMBER)

        def grounded(x: float) -> bool:
            # exact, rounded (87 vs 87.04), or same value as a % of 1 (0.87 vs 87)
            return any(abs(x - s) <= 0.51 or (s and abs(x - s) / abs(s) <= 0.005) or abs(x - s * 100) <= 0.51 for s in source)

        ungrounded = [x for x in answer_numbers if not grounded(x)]
        self.score = 1 - len(ungrounded) / len(answer_numbers)
        shown = ", ".join(f"{x:g}" for x in ungrounded[:10])
        self.reason = (
            f"All {len(answer_numbers)} numbers found in tool outputs."
            if not ungrounded
            else f"{len(ungrounded)}/{len(answer_numbers)} numbers not found in any tool output: {shown}"
        )
        return self.score


# ── Assembling the metric set for one case ────────────────────────────────────


def build_metrics(case: dict, run, judge: DeepEvalBaseLLM | None) -> list[BaseMetric]:
    expected = case.get("expected_tools", [])
    metrics: list[BaseMetric] = []
    if expected:
        metrics.append(
            ToolCorrectnessMetric(
                threshold=1.0,
                should_consider_ordering=case.get("ordered", True),
                async_mode=False,
                model=judge or NoJudge(),
            )
        )
    metrics += [
        ToolArgumentsMetric(expected),
        OutcomeMetric(case.get("outcome", "finish"), case.get("forbidden_tools", []), case.get("max_tool_calls", 6), run),
        AnswerContainsMetric(case.get("answer_contains", [])),
        GroundedNumbersMetric(run, threshold=case.get("grounded_threshold", 0.8)),
    ]
    if judge and case.get("expected_answer"):
        metrics.append(
            GEval(
                name="Answer quality",
                criteria=(
                    "Judge whether the actual output correctly and honestly answers the input, given the "
                    "expected output as a reference. Reward stating the key facts from the expected output. "
                    "Penalize invented data, wrong organisms or numbers, claiming success after a tool error, "
                    "and missing the main conclusion. Do not penalize extra correct detail or different wording."
                ),
                evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
                model=judge,
                threshold=0.6,
                async_mode=False,
            )
        )
    return metrics
