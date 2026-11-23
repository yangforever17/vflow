"""Use a (searched) workflow as a tool:  spec in, verified Verilog out.

    python -m vflowx.cli --spec my_spec.txt --top my_module -o my_module.v
    python -m vflowx.cli --spec my_spec.txt --workflow examples/af27_best --show-usage

The spec is plain natural language (port list + behaviour). The output is linted again before writing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from .evaluate import load_workflow
from .llm import CURRENT_USAGE, LLM, Usage
from .ops import Ops

DEFAULT_WF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples/af27_best")


async def run(spec: str, top: str, wf: str) -> tuple[str, dict, dict]:
    usage = Usage()
    CURRENT_USAGE.set(usage)
    llm = LLM()
    ops = Ops(llm, top)
    t0 = time.time()
    code = await load_workflow(wf)(spec, ops)
    lint = await ops.lint(code)
    stats = {**usage.as_dict(), "seconds": round(time.time() - t0, 1), "model": llm.model}
    return code, lint, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="spec text file ('-' for stdin)")
    ap.add_argument("--top", default="TopModule", help="top module name used in the spec")
    ap.add_argument("--workflow", default=os.environ.get("VFX_WORKFLOW", DEFAULT_WF))
    ap.add_argument("-o", "--out", help="write Verilog here (default: stdout)")
    ap.add_argument("--show-usage", action="store_true")
    ap.add_argument("--executor", help="executor model '<provider>/<model>' (default: `executor` in aflow.toml)")
    a = ap.parse_args()
    if a.executor:
        os.environ["VFX_EXECUTOR"] = a.executor
    spec = sys.stdin.read() if a.spec == "-" else open(a.spec).read()
    code, lint, stats = asyncio.run(run(spec, a.top, a.workflow))
    if a.out:
        open(a.out, "w").write(code)
    else:
        sys.stdout.write(code)
    status = "lint OK" if lint["ok"] else "LINT FAILED:\n" + lint["errors"]
    print(f"[vflowx] {status}", file=sys.stderr)
    if lint.get("warnings"):
        print(f"[vflowx] verilator warnings:\n{lint['warnings']}", file=sys.stderr)
    if a.show_usage:
        print("[vflowx] " + json.dumps(stats), file=sys.stderr)
    sys.exit(0 if lint["ok"] else 2)


if __name__ == "__main__":
    main()
