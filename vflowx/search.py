"""AFlow-style workflow search with a coding agent (codex or DeepSeek Harness) as the optimizer.

    python -m vflowx.search --run runs/s1 --init workflows/io --rounds 12 --width 2

Loop (per iteration, `width` children in parallel):
  1. pick a parent among the top-k rounds (AFlow mixed softmax: 0.3 uniform + 0.7 softmax(0.2*score*100))
  2. copy parent -> rounds/round_N, write CONTEXT.md (scores, failure cases on VAL, experience log)
  3. the optimizer agent (`codex exec`, or DeepSeek Harness `dsh --profile headless`) edits graph.py /
     prompt.py in that dir and writes modification.md
  4. static leak check + smoke test (one automatic optimizer fix attempt on crash)
  5. evaluate on the VAL split; record experience (parent -> child delta)
Stops after `patience` iterations without a new best. Test split is never touched here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import shutil
import sys
import time

from . import bench, config, harness
from .evaluate import MU, evaluate, load_workflow
from .llm import LLM

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG_DIR)
FORBIDDEN = re.compile(r"verilog-eval|_ref\.sv|_test\.sv|RTLLM|testbench\.v|open\(|subprocess|os\.system|"
                       r"importlib|__import__|requests|urllib|socket", re.I)


FOCI = [
    "STRUCTURE — change the control flow / verification loop (how candidates are produced, checked, repaired, "
    "selected, when to stop or escalate). A pure prompt-wording change does NOT satisfy this focus.",
    "PROMPTS & SAMPLING — change instructions, add a spec-analysis/decomposition step, temperatures, "
    "number of samples, or where `think=True` is used.",
    "EFFICIENCY — keep the pass rate but cut tokens and latency (e.g. early exit when cheap checks agree, "
    "adaptive sampling, escalate only on disagreement/failure).",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


class Search:
    def __init__(self, run: str, init: str, width: int, repeats: int, topk: int, think_optimizer: str,
                 val_split: str = "val", optimizer: str | None = None, harness_name: str | None = None):
        self.val = val_split
        self.run = os.path.abspath(run)
        self.rounds_dir = f"{self.run}/rounds"
        os.makedirs(self.rounds_dir, exist_ok=True)
        self.init = init  # comma-separated list of initial workflow dirs
        self.width, self.repeats, self.topk = width, repeats, topk
        self.effort = think_optimizer
        # the optimizer model, driven by a coding-agent harness (codex or dsh); both from aflow.toml
        self.harness = harness.make(harness_name or config.harness_name(),
                                    config.optimizer_spec() if optimizer is None else optimizer,
                                    think_optimizer, self.run)
        self.llm = LLM()
        if self.llm.kind != "vllm":
            self.share_executor_key()
        self.exp_file = f"{self.run}/experience.jsonl"
        self.board_file = f"{self.run}/leaderboard.json"
        self.board: list[dict] = json.load(open(self.board_file)) if os.path.exists(self.board_file) else []
        self.lock = asyncio.Lock()

    def share_executor_key(self):
        """Let the optimizer's own smoke tests reach a hosted executor API: codex and dsh strip env vars named
        *KEY*/*TOKEN*/*SECRET* from the shells they spawn, so the key goes to a 0600 file whose path
        (VFX_CRED_FILE) survives. See llm._read_cred_file."""
        path = f"{self.run}/.executor_credential"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(self.llm.api_key)
        os.environ["VFX_CRED_FILE"] = path

    # ------------------------------------------------------------------ bookkeeping
    def next_round_id(self) -> int:
        ids = [int(d.split("_")[1]) for d in os.listdir(self.rounds_dir) if d.startswith("round_")]
        return max(ids, default=0) + 1

    def best(self) -> dict | None:
        ok = [b for b in self.board if b.get("score") is not None]
        return max(ok, key=lambda b: b["score"]) if ok else None

    def experience(self) -> list[dict]:
        if not os.path.exists(self.exp_file):
            return []
        return [json.loads(l) for l in open(self.exp_file)]

    def select_parent(self) -> dict:
        ok = sorted([b for b in self.board if b.get("score") is not None], key=lambda b: -b["score"])
        cand = ok[: self.topk]
        if ok and not any(b["round"] == 1 for b in cand):
            r1 = [b for b in ok if b["round"] == 1]
            cand += r1
        s = [b["score"] * 100 for b in cand]
        m = max(s)
        w = [math.exp(0.2 * (x - m)) for x in s]
        p = [0.3 / len(cand) + 0.7 * wi / sum(w) for wi in w]
        return random.choices(cand, weights=p, k=1)[0]

    # ------------------------------------------------------------------ context for codex
    def failure_digest(self, rdir: str, k: int = 6) -> str:
        rows = [json.loads(l) for l in open(f"{rdir}/eval/results.jsonl")]
        fails = [r for r in rows if not r["pass"]]
        random.shuffle(fails)
        specs = {p.pid: p.spec for p in bench.load(self.val)}
        out = []
        for r in fails[:k]:
            hint = "\n".join(l for l in r["score_log"].splitlines()
                             if re.search(r"hint|mismatch|error|fail|expected", l, re.I))[:600]
            out.append(f"### {r['pid']}  status={r['status']} mismatches={r['mismatches']} tokens={r['total_tokens']}\n"
                       f"Spec (truncated):\n```\n{specs.get(r['pid'], '')[:1400]}\n```\n"
                       f"Workflow output (truncated):\n```verilog\n{(r['code'] or '')[:1400]}\n```\n"
                       + (f"Workflow error:\n```\n{r['error'][:800]}\n```\n" if r["error"] else "")
                       + (f"Grader hints:\n```\n{hint}\n```\n" if hint else ""))
        hist = {}
        for r in rows:
            hist[r["status"]] = hist.get(r["status"], 0) + 1
        return f"Status histogram on val: {hist}\n\n" + "\n".join(out)

    def executor_text(self) -> str:
        """How the executor LLM is described to the optimizer (the vLLM wording is the one used for af27)."""
        if self.llm.kind == "vllm":
            return (f"Executor LLM: `{self.llm.model}` served by vLLM (cheap to call in parallel; `think=True` enables its\n"
                    "reasoning mode, which usually costs 5-20x more completion tokens).")
        cost = ", which usually costs several times more completion tokens"
        think = (f"`think=True` switches to `{self.llm.think_model}`{cost}" if self.llm.think_model else
                 f"`think=True` enables its reasoning mode{cost}" if re.match(r"^(gpt-5|o\d)", self.llm.model) else
                 "`think=True` has no effect for this model")
        return (f"Executor LLM: `{self.llm.model}` via the {self.llm.provider} API (paid per token, rate-limited; "
                f"{think}).")

    def write_context(self, child: str, parent: dict, cid: int):
        best = self.best()
        exp = self.experience()
        exp_txt = "\n".join(
            f"- round {e['parent']}→{e['round']}: {'IMPROVED' if e['succeed'] else 'no gain'} "
            f"(score {e['before']['score']:.3f}→{e['after']['score']:.3f}, pass {e['before']['pass_rate']:.3f}→"
            f"{e['after']['pass_rate']:.3f}, tokens {e['before']['avg_tokens']:.0f}→{e['after']['avg_tokens']:.0f}): "
            f"{e['modification'][:300]}" for e in exp) or "(none yet)"
        board_txt = "\n".join(f"- round {b['round']}: score={b['score']} pass={b['pass_rate']} tokens={b['avg_tokens']}"
                              f" latency={b['avg_latency']}s" for b in sorted(self.board, key=lambda b: b['round'])
                              if b.get("score") is not None)
        ctx = f"""# Task: improve a Verilog-generation agentic workflow (round {cid})

You are the optimizer in an AFlow-style workflow search. The files `graph.py` and `prompt.py` in this
directory are a copy of round {parent['round']} (the parent). Make ONE focused, well-motivated modification
that you expect to raise the objective, then write `modification.md` (2-5 lines: what you changed and why).

**Assigned focus for this round:** {FOCI[cid % len(FOCI)]}
Other optimizers work in parallel on other foci, so stay within yours.
Note: one evaluation = val split × {self.repeats} runs; run-to-run noise is several points (small split), so prefer
changes with a clear mechanism over small wording tweaks.

## Objective (higher is better)
score = pass_rate − {MU} × avg_total_tokens_per_problem / 1000
pass_rate = functional pass@1 on a hidden reference testbench on the validation split `{self.val}` of
{bench.DESCRIPTIONS[bench.bench_of(self.val)]}.
{self.executor_text()}
Parent round {parent['round']}: score={parent['score']} pass={parent['pass_rate']} tokens={parent['avg_tokens']}
Best so far: round {best['round']} score={best['score']} pass={best['pass_rate']} tokens={best['avg_tokens']}

## Workflow contract
- `graph.py` must define `async def solve(spec: str, ops) -> str` returning the final Verilog text.
- `prompt.py` holds string constants; graph.py accesses them as the injected global `P` (e.g. `P.GEN`).
- `ops` API: read `{PKG_DIR}/ops.py` (generate / repair / review / ask / lint / interface / traces / vote /
  diff_traces / self_test / extract_code). You may also write plain Python logic (loops, branching,
  asyncio.gather for parallel calls). Imports allowed: asyncio, re, json, math, random, collections, itertools.
- FORBIDDEN (automatic rejection): reading files, subprocess, network, importlib, referencing benchmark paths.
  The workflow only gets the natural-language spec. No reference solutions or testbenches exist at run time.
- Must finish in < 15 min per problem. Keep it robust: a crash = failed problem.

## What happened so far
Leaderboard:
{board_txt}

Experience log (parent→child, do NOT repeat ideas that gave no gain unless you change them substantially):
{exp_txt}

## Failure cases of the parent on the validation split
{self.failure_digest(f"{self.rounds_dir}/round_{parent['round']}")}

## How to check your change
From `{ROOT}` you may run a smoke test on at most 3 problems (do not run more, compute is shared):
`cd {ROOT} && .venv/bin/python -m vflowx.evaluate --workflow {child} --split {self.val} --limit 3`
Think about failure *patterns* (not the individual problems) — the result is scored on unseen problems.
Edit only files inside `{child}`.
"""
        open(f"{child}/CONTEXT.md", "w").write(ctx)

    # ------------------------------------------------------------------ optimizer agent
    async def codex(self, child: str, message: str, tag: str) -> int:
        """Run the optimizer agent (codex or dsh, see harness.py) once in `child`. Its transcript goes to
        `child/<harness>_<tag>.log`. Returns the exit code (124 = timed out)."""
        cmd, env, cwd = self.harness.command(child, message)
        with open(f"{child}/{self.harness.name}_{tag}.log", "w") as f:
            p = await asyncio.create_subprocess_exec(*cmd, stdout=f, stderr=asyncio.subprocess.STDOUT,
                                                     stdin=asyncio.subprocess.DEVNULL, cwd=cwd, env=env)
            try:
                return await asyncio.wait_for(p.wait(), 1800)
            except asyncio.TimeoutError:
                p.kill()
                await p.wait()
                return 124

    def static_check(self, child: str) -> str | None:
        for fn in ("graph.py", "prompt.py"):
            path = f"{child}/{fn}"
            if not os.path.exists(path):
                return f"{fn} missing" if fn == "graph.py" else None
            src = open(path).read()
            code_only = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
            if fn == "graph.py" and (m := FORBIDDEN.search(code_only)):
                return f"forbidden construct in {fn}: {m.group(0)!r}"
        try:
            load_workflow(child)
        except Exception as e:  # noqa: BLE001
            return f"import error: {e!r}"
        return None

    async def smoke(self, child: str) -> str | None:
        s = await evaluate(child, self.val, 1, None, limit=2, timeout=900, llm=self.llm, quiet=True)
        if s["status"].get("workflow_error"):
            return "workflow crashed on smoke test"
        return None

    # ------------------------------------------------------------------ one child
    async def expand(self) -> dict | None:
        async with self.lock:
            parent = self.select_parent()
            cid = self.next_round_id()
            child = f"{self.rounds_dir}/round_{cid}"
            os.makedirs(child)
            for fn in ("graph.py", "prompt.py"):
                src = f"{self.rounds_dir}/round_{parent['round']}/{fn}"
                if os.path.exists(src):
                    shutil.copy(src, child)
            self.write_context(child, parent, cid)
        log(f"round {cid}: parent={parent['round']} ({parent['score']}) -> {self.harness.name}")
        t0 = time.time()
        await self.codex(child, "Read CONTEXT.md in the current directory and carry out the task it describes.", "opt")
        problem = self.static_check(child) or await self.smoke(child)
        if problem:
            log(f"round {cid}: {problem} -> {self.harness.name} fix attempt")
            await self.codex(child, f"Your workflow in this directory is broken: {problem}. "
                                    "Fix it (see CONTEXT.md for the contract) without changing the idea.", "fix")
            problem = self.static_check(child) or await self.smoke(child)
        if problem:
            log(f"round {cid}: rejected ({problem})")
            entry = {"round": cid, "parent": parent["round"], "score": None, "rejected": problem}
            async with self.lock:
                self.board.append(entry)
                self._save()
            return None
        opt_min = (time.time() - t0) / 60
        s = await evaluate(child, self.val, self.repeats, f"{child}/eval", llm=self.llm, quiet=True)
        mod = open(f"{child}/modification.md").read().strip() if os.path.exists(f"{child}/modification.md") else "?"
        entry = {"round": cid, "parent": parent["round"], **{k: s[k] for k in
                 ("score", "pass_rate", "avg_tokens", "avg_latency", "avg_calls", "status")},
                 "optimizer_minutes": round(opt_min, 1), "modification": mod[:500]}
        async with self.lock:
            self.board.append(entry)
            exp = {"round": cid, "parent": parent["round"], "modification": mod[:600],
                   "before": {k: parent[k] for k in ("score", "pass_rate", "avg_tokens")},
                   "after": {k: s[k] for k in ("score", "pass_rate", "avg_tokens")},
                   "succeed": s["score"] > parent["score"]}
            with open(self.exp_file, "a") as f:
                f.write(json.dumps(exp) + "\n")
            self._save()
        log(f"round {cid}: score={s['score']} pass={s['pass_rate']} tokens={s['avg_tokens']} | {mod[:120]!r}")
        return entry

    def _save(self):
        json.dump(self.board, open(self.board_file, "w"), indent=1)

    # ------------------------------------------------------------------ main loop
    async def run_search(self, rounds: int, patience: int):
        await self.llm.resolve_model()
        if not self.board:
            for i, init in enumerate(self.init.split(","), 1):
                r = f"{self.rounds_dir}/round_{i}"
                shutil.copytree(os.path.abspath(init), r, dirs_exist_ok=True)
                s = await evaluate(r, self.val, self.repeats, f"{r}/eval", llm=self.llm, quiet=True)
                self.board.append({"round": i, "parent": None, **{k: s[k] for k in
                                   ("score", "pass_rate", "avg_tokens", "avg_latency", "avg_calls", "status")},
                                   "modification": f"initial workflow ({os.path.basename(init.rstrip('/'))})"})
                self._save()
                log(f"round {i} (init {init}): score={s['score']} pass={s['pass_rate']} tokens={s['avg_tokens']}")
        stale = 0
        while self.next_round_id() <= rounds:
            before = self.best()["score"]
            n = min(self.width, rounds - self.next_round_id() + 1)
            await asyncio.gather(*(self.expand() for _ in range(n)))
            stale = 0 if self.best()["score"] > before else stale + 1
            log(f"best so far: round {self.best()['round']} score={self.best()['score']} (stale={stale})")
            if stale >= patience:
                log("early stop: no improvement")
                break

    async def finalize(self, k: int = 3, repeats: int = 3):
        """Re-evaluate the top-k rounds on val with more repeats to de-noise the final pick."""
        repeats = max(repeats, self.repeats + 1)
        ok = sorted([b for b in self.board if b.get("score") is not None], key=lambda b: -b["score"])[:k]
        res = []
        for b in ok:
            d = f"{self.rounds_dir}/round_{b['round']}"
            s = await evaluate(d, self.val, repeats, f"{d}/eval_final", llm=self.llm, quiet=True)
            res.append({"round": b["round"], **{x: s[x] for x in ("score", "pass_rate", "avg_tokens", "avg_latency")}})
            log(f"final val x{repeats}: round {b['round']} score={s['score']} pass={s['pass_rate']}")
        best = max(res, key=lambda r: r["score"])
        json.dump({"candidates": res, "selected": best}, open(f"{self.run}/final.json", "w"), indent=1)
        shutil.copytree(f"{self.rounds_dir}/round_{best['round']}", f"{self.run}/best", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("eval*", "codex_*.log", "dsh_*.log", "CONTEXT.md"))
        log(f"selected round {best['round']} -> {self.run}/best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", default=f"{ROOT}/workflows/io")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--executor", help="executor model '<provider>/<model>' (default: `executor` in aflow.toml)")
    ap.add_argument("--optimizer", help="optimizer model '<provider>/<model>' ('<model>' or '': the harness's own "
                                        "login; default: `optimizer` in aflow.toml)")
    ap.add_argument("--harness", choices=sorted(harness.HARNESSES),
                    help="coding agent that drives the optimizer (default: `harness` in aflow.toml)")
    ap.add_argument("--effort", default="high", help="optimizer reasoning effort (low/medium/high/xhigh)")
    ap.add_argument("--val-split", default="val", help="val | rtllm_val")
    ap.add_argument("--finalize-only", action="store_true")
    a = ap.parse_args()
    if a.executor:
        os.environ["VFX_EXECUTOR"] = a.executor
    s = Search(a.run, a.init, a.width, a.repeats, a.topk, a.effort, a.val_split, a.optimizer, a.harness)

    async def go():
        await s.llm.resolve_model()
        if not a.finalize_only:
            await s.run_search(a.rounds, a.patience)
        await s.finalize()

    asyncio.run(go())


if __name__ == "__main__":
    main()
