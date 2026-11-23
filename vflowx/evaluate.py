"""Run a workflow directory over a benchmark split and score it.

    python -m vflowx.evaluate --workflow workflows/io --split val --out runs/io_val

Workflow directory layout:  graph.py (defines `async def solve(spec, ops) -> str`, may use `P.<NAME>`)
                            prompt.py (string constants; injected into graph.py as module `P`)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
import time
import traceback
import uuid

from . import bench
from .llm import CURRENT_USAGE, LLM, Usage
from .ops import Ops

MU = float(os.environ.get("VFX_MU", "0.0015"))  # score penalty per 1k tokens per problem


def objective(pass_rate: float, avg_tokens: float, mu: float = MU) -> float:
    return pass_rate - mu * avg_tokens / 1000.0


def load_workflow(wf_dir: str):
    tag = uuid.uuid4().hex[:8]
    P = None
    if os.path.exists(f"{wf_dir}/prompt.py"):
        spec = importlib.util.spec_from_file_location(f"vfx_prompt_{tag}", f"{wf_dir}/prompt.py")
        P = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(P)
    spec = importlib.util.spec_from_file_location(f"vfx_graph_{tag}", f"{wf_dir}/graph.py")
    G = importlib.util.module_from_spec(spec)
    G.P = P
    spec.loader.exec_module(G)
    if not hasattr(G, "solve"):
        raise AttributeError(f"{wf_dir}/graph.py does not define solve(spec, ops)")
    return G.solve


def seed_base(pid: str, rep: int, crn_salt: int = 0) -> int:
    return int(hashlib.sha1(f"{pid}|{rep}|{crn_salt}".encode()).hexdigest()[:8], 16)


async def diagnose_candidates(prob, cands: list, final_pass: bool, cap: int = 16) -> dict:
    """Oracle credit assignment (FEEDBACK set only): score every distinct candidate the workflow produced.
    Tells the optimizer whether a failure is a generation miss (no correct candidate ever existed) or a
    selection loss (a correct candidate existed but the workflow returned something else)."""
    seen, uniq = set(), []
    for stage, c in cands:
        k = re.sub(r"\s+", " ", c or "").strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append((stage, c))
    uniq = uniq[:cap]
    res = await asyncio.gather(*(prob.score(c) for _, c in uniq), return_exceptions=True)
    ok = [(st, isinstance(r, dict) and bool(r.get("pass"))) for (st, _), r in zip(uniq, res)]
    d = {"n_cands": len(uniq), "n_pass": sum(o for _, o in ok),
         "pass_by_stage": {st: sum(o for s2, o in ok if s2 == st) for st in sorted({s for s, _ in ok})},
         "n_by_stage": {st: sum(1 for s2, _ in ok if s2 == st) for st in sorted({s for s, _ in ok})}}
    d["verdict"] = ("pass" if final_pass else "selection_loss" if d["n_pass"] else "generation_miss")
    return d


async def _one(prob, solve, llm, rep: int, timeout: float, crn: bool = True, crn_salt: int = 0,
               diagnose: bool = False) -> dict:
    usage = Usage(seed_base=seed_base(prob.pid, rep, crn_salt) if crn else None)
    CURRENT_USAGE.set(usage)
    t0 = time.time()
    err = None
    code = ""
    try:
        code = await asyncio.wait_for(solve(prob.spec, Ops(llm, prob.top)), timeout)
        if not isinstance(code, str):
            raise TypeError(f"solve returned {type(code).__name__}, expected str")
    except asyncio.TimeoutError:
        err = f"workflow timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001 - workflow bugs must not kill the run
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        tb = traceback.format_exc().splitlines()
        err += "\n" + "\n".join(tb[-6:])
    latency = time.time() - t0
    sc = await prob.score(code) if code and not err else {"pass": False, "status": "workflow_error", "log": err}
    # the final code is included: workflows may edit candidates deterministically (no ops.generate/repair call)
    diag = await diagnose_candidates(prob, usage.candidates + [("final", code)], bool(sc["pass"])) if diagnose else None
    return {"diag": diag, "pid": prob.pid, "rep": rep, "pass": bool(sc["pass"]), "status": sc["status"],
            "mismatches": sc.get("mismatches"), "score_log": (sc.get("log") or "")[:1200],
            "error": err, "latency": round(latency, 2), **usage.as_dict(), "code": code}


def summarize(results: list[dict], n_problems: int, repeats: int, extra: dict | None = None) -> dict:
    n = len(results)
    pass_rate = sum(r["pass"] for r in results) / n
    avg_tok = statistics.mean(r["total_tokens"] for r in results)
    lat = sorted(r["latency"] for r in results)
    status = {}
    for r in results:
        status[r["status"]] = status.get(r["status"], 0) + 1
    return {**(extra or {}), "n_problems": n_problems, "repeats": repeats,
            "pass_rate": round(pass_rate, 4), "avg_tokens": round(avg_tok, 1),
            "avg_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in results), 1),
            "avg_calls": round(statistics.mean(r["calls"] for r in results), 2),
            "avg_latency": round(statistics.mean(lat), 2), "p90_latency": lat[int(0.9 * (n - 1))],
            "score": round(objective(pass_rate, avg_tok), 4), "mu": MU, "status": status}


async def evaluate(wf_dir: str, split: str = "val", repeats: int = 1, out: str | None = None,
                   limit: int | None = None, timeout: float = 1200, llm: LLM | None = None,
                   pids: list[str] | None = None, quiet: bool = False, crn: bool = True,
                   crn_salt: int = 0, diagnose: bool = False) -> dict:
    llm = llm or LLM()
    solve = load_workflow(wf_dir)
    probs = bench.load(split)
    if pids:
        probs = [p for p in probs if p.pid in set(pids)]
    if limit:
        probs = probs[:limit]
    t0 = time.time()
    tasks = [asyncio.create_task(_one(p, solve, llm, r, timeout, crn, crn_salt, diagnose)) for r in range(repeats) for p in probs]
    results = []
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        results.append(await fut)
        if not quiet and (i % 10 == 0 or i == len(tasks)):
            pr = sum(r["pass"] for r in results) / len(results)
            print(f"  [{os.path.basename(wf_dir.rstrip('/'))}] {i}/{len(tasks)} pass={pr:.3f} "
                  f"elapsed={time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    results.sort(key=lambda r: (r["pid"], r["rep"]))
    summary = summarize(results, len(probs), repeats,
                        {"workflow": wf_dir, "split": split, "model": llm.model})
    summary["wall_seconds"] = round(time.time() - t0, 1)
    if out:
        os.makedirs(out, exist_ok=True)
        with open(f"{out}/results.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pids", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--executor", help="executor model '<provider>/<model>' (default: `executor` in aflow.toml)")
    a = ap.parse_args()
    if a.executor:
        os.environ["VFX_EXECUTOR"] = a.executor
    s = asyncio.run(evaluate(a.workflow, a.split, a.repeats, a.out, a.limit, a.timeout, pids=a.pids))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
