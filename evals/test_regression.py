"""
DeepEval regression suite for the gget agent.

    uv run pytest evals                      # all cases
    uv run pytest evals -m "not slow"        # skip BLAST / NCBI Virus cases (minutes each)
    uv run pytest evals -k align             # one group by id
    uv run pytest evals --save-baseline      # record this run as the baseline to compare against

Needs the same env as the agent (LLM, credentials, network to NCBI/Ensembl/UCSC/Bgee).
Optional LLM judge for answer quality: EVAL_JUDGE_MODEL, EVAL_JUDGE_BASE_URL, EVAL_JUDGE_API_KEY.
"""

import asyncio
from pathlib import Path

import pytest
import yaml
from deepeval.test_case import LLMTestCase, ToolCall

from harness import run_agent
from metrics import build_metrics, get_judge

CASES = yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text())
JUDGE = get_judge()


def _params(case):
    marks = [pytest.mark.slow] if case.get("slow") else []
    return pytest.param(case, id=case["id"], marks=marks)


@pytest.mark.parametrize("case", [_params(c) for c in CASES])
def test_agent(case, results_log):
    run = asyncio.run(run_agent(case["query"], case.get("sequence"), timeout=case.get("timeout", 900)))

    test_case = LLMTestCase(
        input=run.request,
        actual_output=run.final_answer or f"[no reply; outcome={run.outcome}; error={run.error}]",
        expected_output=case.get("expected_answer"),
        tools_called=[ToolCall(name=c.name, input_parameters=c.args, output=c.output) for c in run.work_calls],
        expected_tools=[ToolCall(name=t["name"]) for t in case.get("expected_tools", [])],
        name=case["id"],
    )

    metrics = build_metrics(case, run, JUDGE)
    scores = {}
    for metric in metrics:
        try:
            metric.measure(test_case)
            ok = metric.is_successful()
        except Exception as e:  # e.g. judge unreachable: report, don't crash the suite
            metric.score, metric.reason, ok = 0.0, f"metric error: {e}", False
        scores[metric.__name__] = {"score": round(metric.score or 0, 3), "passed": bool(ok), "reason": metric.reason}

    passed = all(s["passed"] for s in scores.values())
    results_log.append(
        {
            "id": case["id"],
            "passed": passed,
            "seconds": run.seconds,
            "outcome": run.outcome,
            "tools": [
                {"call": f"{c.name}({str(c.args)[:300]})", "output": (c.output or "")[:500]} for c in run.calls
            ],
            "answer": run.final_answer[:1500],
            "metrics": scores,
        }
    )

    if not passed:
        failed = "\n".join(f"  ✗ {name} ({s['score']}): {s['reason']}" for name, s in scores.items() if not s["passed"])
        trace = "\n".join(f"  {i + 1}. {c.name}({c.args})" for i, c in enumerate(run.calls)) or "  (no tool calls)"
        tb = f"\n\nAgent traceback:\n{run.traceback}" if run.traceback else ""
        pytest.fail(
            f"\n{failed}\n\nTool calls:\n{trace}\n\nFinal answer:\n  {run.final_answer[:800] or '(none)'}{tb}",
            pytrace=False,
        )
