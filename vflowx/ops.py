"""Operator library available to every workflow as `ops`.

A workflow is `async def solve(spec: str, ops) -> str` returning the final Verilog source.
All operators are async (except `extract_code` / `vote`). Use `asyncio.gather` for parallelism.

Rules for workflows (enforced by the evaluator, not by code):
  * The reference solution / reference testbench is NEVER available. Self-checking must come from
    lint, LLM-written tests (`self_test`), or cross-checking candidates (`traces` + `vote`).
  * Every LLM token is counted and penalised in the objective. Prefer cheap checks (lint, traces)
    before expensive ones (more samples, thinking mode).
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter

from . import eda, formal
from .llm import CURRENT_USAGE, LLM

SYSTEM = ("You are a senior digital design engineer. You write correct, synthesizable Verilog-2005 / "
          "SystemVerilog-2012 that compiles under Icarus Verilog (-g2012).")

FORMAT = ("Return the COMPLETE implementation of module `{top}` (keep the exact module name and port names "
          "from the specification) in a single ```verilog code block. Do not include a testbench.")

_BLOCK_RE = re.compile(r"```[ \t]*(?:verilog|systemverilog|sv|v)?[ \t]*\n(.*?)```", re.S | re.I)


class Ops:
    def __init__(self, llm: LLM, top: str = "TopModule"):
        self.llm = llm
        self.top = top

    # ------------------------------------------------------------------ helpers
    def extract_code(self, text: str) -> str:
        """Pull the Verilog for `self.top` out of an LLM reply."""
        blocks = _BLOCK_RE.findall(text)
        for b in reversed(blocks):
            if re.search(rf"\bmodule\s+{re.escape(self.top)}\b", b):
                return b.strip() + "\n"
        for b in reversed(blocks):
            if "module" in b:
                return b.strip() + "\n"
        m = re.search(rf"(module\s+{re.escape(self.top)}\b.*endmodule)", text, re.S)
        return (m.group(1) if m else text).strip() + "\n"

    # ------------------------------------------------------------------ LLM operators
    async def ask(self, prompt: str, *, temperature: float = 0.2, think: bool = False,
                  max_tokens: int | None = None) -> str:
        """Free-form LLM call (analysis, planning, extracting a truth table, ...). Returns text."""
        return await self.llm.ask(prompt, system=SYSTEM, temperature=temperature, think=think, max_tokens=max_tokens)

    async def generate(self, spec: str, instruction: str = "", *, n: int = 1, temperature: float = 0.2,
                       think: bool = False, context: str = "") -> list[str]:
        """Generate `n` candidate implementations of the spec. `instruction` is prepended guidance,
        `context` is extra material (e.g. an analysis from `ask`). Returns list of Verilog strings."""
        parts = [instruction.strip()] if instruction else []
        parts.append("## Specification\n" + spec.strip())
        if context:
            parts.append("## Notes\n" + context.strip())
        parts.append(FORMAT.format(top=self.top))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
        outs = await self.llm.chat(msgs, n=n, temperature=temperature if n == 1 else max(temperature, 0.5), think=think)
        return _record([self.extract_code(o) for o in outs], "generate")

    async def repair(self, spec: str, code: str, feedback: str, instruction: str = "", *, n: int = 1,
                     temperature: float = 0.2, think: bool = False) -> list[str]:
        """Revise `code` given `feedback` (compiler errors, lint warnings, failing self-tests,
        disagreement traces, review comments...). Returns list of revised Verilog strings."""
        prompt = (f"{instruction.strip()}\n\n" if instruction else "") + (
            f"## Specification\n{spec.strip()}\n\n## Current implementation\n```verilog\n{code.strip()}\n```\n\n"
            f"## Feedback\n{feedback.strip()}\n\nFix every problem the feedback reveals while keeping correct "
            f"behaviour. If the feedback is wrong given the specification, keep the code as is.\n"
            + FORMAT.format(top=self.top))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        outs = await self.llm.chat(msgs, n=n, temperature=temperature, think=think)
        return _record([self.extract_code(o) for o in outs], "repair")

    async def review(self, spec: str, code: str, *, temperature: float = 0.0, think: bool = False) -> str:
        """LLM code review against the spec. Returns text; starts with 'LGTM' if no issue found."""
        return await self.ask(
            f"## Specification\n{spec.strip()}\n\n## Implementation\n```verilog\n{code.strip()}\n```\n\n"
            "Check the implementation against the specification line by line (reset polarity and sync/async, "
            "clock edge, bit widths, signedness, state encoding, off-by-one, default/latch cases). "
            "If it is fully correct answer exactly `LGTM`. Otherwise list each concrete bug and the fix.",
            temperature=temperature, think=think)

    # ------------------------------------------------------------------ EDA operators
    async def lint(self, code: str) -> dict:
        """iverilog compile + verilator lint. -> {ok: bool, errors: str, warnings: str}"""
        return await eda.lint(code, self.top)

    async def interface(self, codes: list[str]) -> list | None:
        """Majority port list (yosys) across candidate codes. -> list[Port] or None."""
        res = await asyncio.gather(*(eda.interface(c, self.top) for c in codes))
        sigs = [tuple((p.name, p.direction, p.width) for p in r) for r in res if r]
        if not sigs:
            return None
        best = Counter(sigs).most_common(1)[0][0]
        return [eda.Port(*t) for t in best]

    async def traces(self, codes: list[str], *, cycles: int = 200, seed: int = 1, ports=None,
                     vectors: list[dict] | None = None) -> list[str | None]:
        """Simulate every candidate under the SAME stimulus (random, or explicit `vectors` from
        `directed_vectors`); returns one output trace per candidate (None if it fails to compile/run).
        Trace lines: `@i IN <inputs> OUT <all outputs packed, pre-edge> | <each output, post-edge>`.
        Equal traces => equivalent on this stimulus. Use with `vote` for reference-free differential testing."""
        ports = ports or await self.interface(codes)
        if not ports:
            return [None] * len(codes)
        return list(await asyncio.gather(*(eda.trace(c, ports, cycles, seed, self._sim_top(c), vectors)
                                           for c in codes)))

    def _sim_top(self, code: str) -> str:
        """Module the stimulus should instantiate: the spec's top if the code defines it, else TopModule
        (keeps workflows that rename to TopModule for simulation working)."""
        if re.search(rf"\bmodule\s+{re.escape(self.top)}\b", code):
            return self.top
        return "TopModule"

    def vote(self, codes: list[str], traces: list[str | None]) -> dict:
        """Group candidates by identical output trace. -> {best: code, best_idx, clusters: [[idx..], ..]
        (largest first), agreement: fraction of valid candidates in the largest cluster}."""
        groups: dict[str, list[int]] = {}
        for i, t in enumerate(traces):
            if t is not None:
                groups.setdefault(_outputs_only(t), []).append(i)
        clusters = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
        if not clusters:
            return {"best": codes[0] if codes else "", "best_idx": 0, "clusters": [], "agreement": 0.0}
        valid = sum(len(g) for g in clusters)
        return {"best": codes[clusters[0][0]], "best_idx": clusters[0][0], "clusters": clusters,
                "agreement": len(clusters[0]) / valid}

    def diff_traces(self, trace_a: str, trace_b: str, max_lines: int = 8) -> str:
        """First differing steps between two traces (for repair feedback: 'on these inputs another
        implementation produced different outputs')."""
        out = []
        for la, lb in zip(trace_a.splitlines(), trace_b.splitlines()):
            if _outputs_only(la) != _outputs_only(lb):
                out.append(f"this: {la}\nother: {lb}")
                if len(out) >= max_lines:
                    break
        return "\n".join(out)

    # ------------------------------------------------------------------ formal / adjudication operators
    async def equiv(self, code_a: str, code_b: str, *, ports=None, steps: int = 20) -> dict:
        """Bounded formal equivalence (yosys miter + SAT, reset asserted at step 1, FFs zero-initialised).
        -> {equal: True | False | None, vectors?, trace_a?, trace_b?}. equal=False comes with a
        simulation-confirmed counterexample (`trace_a`/`trace_b` show both outputs on the same inputs).
        None = inconclusive (unsupported construct / timeout). Typically 1-20 s per pair."""
        return await formal.equiv(code_a, code_b, self.top, ports, steps)

    async def refine(self, codes: list[str], traces: list[str | None], *, ports=None, max_checks: int = 6) -> dict:
        """Split trace-vote clusters that are not really equivalent: every member of a multi-member
        cluster is checked against the cluster representative with `equiv`. Returns the same shape as
        `vote` plus `formal`: list of {a, b, equal} checks performed. Clusters are ordered largest first."""
        v = self.vote(codes, traces)
        ports = ports or await self.interface(codes)
        checks, new_clusters, budget = [], [], max_checks
        queue = [list(c) for c in v["clusters"]]
        while queue:  # members split off a cluster are re-checked among themselves (not assumed equal)
            cl = queue.pop(0)
            rep, rest = cl[0], cl[1:]
            if not rest or budget <= 0 or not ports:
                new_clusters.append(cl)
                continue
            todo = rest[:budget]
            res = await asyncio.gather(*(self.equiv(codes[rep], codes[j], ports=ports) for j in todo))
            budget -= len(todo)
            same, split = [rep], []
            for j, r in zip(todo, res):
                checks.append({"a": rep, "b": j, "equal": r["equal"]})
                (split if r["equal"] is False else same).append(j)
            same += rest[len(todo):]  # unchecked (budget exhausted): keep, as trace-equal
            new_clusters.append(same)
            if split:
                queue.append(split)
        new_clusters.sort(key=lambda g: (-len(g), g[0]))
        if not new_clusters:
            return {**v, "formal": checks}
        valid = sum(len(g) for g in new_clusters)
        return {"best": codes[new_clusters[0][0]], "best_idx": new_clusters[0][0], "clusters": new_clusters,
                "agreement": len(new_clusters[0]) / valid, "formal": checks}

    async def adjudicate(self, spec: str, code_a: str, code_b: str, *, ports=None, n: int = 3,
                         think: bool = True, temperature: float = 0.6) -> dict:
        """Decide between two behaviourally different candidates using the SPEC only. Finds a
        distinguishing input sequence (formal counterexample, else random-trace difference), shows both
        output sequences on it and asks the LLM `n` times which one the spec requires.
        -> {winner: 'A'|'B'|'neither'|'same'|'unknown', votes: {...}, evidence: str}"""
        ports = ports or await self.interface([code_a, code_b])
        if not ports:
            return {"winner": "unknown", "votes": {}, "evidence": "no interface"}
        e = await self.equiv(code_a, code_b, ports=ports)
        if e["equal"] is False:
            ta, tb = e["trace_a"], e["trace_b"]
        else:  # bounded proof may be too shallow: a random-trace difference always wins over "equal"
            ta, tb = await self.traces([code_a, code_b], ports=ports)
            if ta is None or tb is None or _outputs_only(ta) == _outputs_only(tb):
                same = e["equal"] is True
                return {"winner": "same" if same else "unknown", "votes": {},
                        "evidence": "bounded-equivalent" if same else "no distinguishing input found"}
        evidence = _side_by_side(ta, tb, ports)
        q = (f"## Specification\n{spec.strip()}\n\n## Two implementations disagree on this input sequence\n"
             f"{evidence}\n\nUsing ONLY the specification, decide which output sequence is correct. Simulate the "
             "spec step by step on the first differing steps (mind reset, clock edge, and whether outputs are "
             "registered or combinational). Finish with exactly one line: `VERDICT: A`, `VERDICT: B` or "
             "`VERDICT: NEITHER`.")
        outs = await self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                                   n=n, temperature=temperature, think=think)
        votes = Counter()
        for o in outs:
            m = re.findall(r"VERDICT:\s*(A|B|NEITHER)", o, re.I)
            votes[m[-1].upper() if m else "UNKNOWN"] += 1
        top, cnt = votes.most_common(1)[0]
        winner = {"A": "A", "B": "B", "NEITHER": "neither"}.get(top, "unknown") if cnt * 2 > n else "unknown"
        return {"winner": winner, "votes": dict(votes), "evidence": evidence}

    async def directed_vectors(self, spec: str, *, ports=None, codes: list[str] | None = None, steps: int = 40,
                               think: bool = False) -> list[dict] | None:
        """LLM proposes a directed input sequence (corner cases: reset mid-operation, overflow, every FSM
        transition, back-to-back requests...) as JSON, WITHOUT expected outputs. Feed it to
        `traces(codes, vectors=...)` to compare candidates on targeted stimulus. -> list of {port: int} or None."""
        ports = ports or (await self.interface(codes) if codes else None)
        if not ports:
            return None
        _, _, clk, _, _ = eda.classify_ports(ports)
        ins = [f"{p.name}[{p.width}]" for p in ports if p.direction == "input" and p is not clk]
        reply = await self.ask(
            f"## Specification\n{spec.strip()}\n\nInputs (name[width]): {', '.join(ins)}"
            f"{' (clock is generated automatically, one step = one clock cycle)' if clk else ''}.\n"
            f"Write an input sequence of about {steps} steps that exercises the corner cases of this spec. "
            "Answer with only a JSON list of objects mapping input names to integer values; an omitted input "
            "keeps its previous value. Start with reset asserted if there is a reset.", think=think)
        m = re.search(r"\[.*\]", reply, re.S)
        if not m:
            return None
        try:
            vecs = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        names = {p.name for p in ports}
        out = [{k: int(v) for k, v in d.items() if k in names and isinstance(v, (int, bool))}
               for d in vecs if isinstance(d, dict)]
        return out[:200] or None

    async def self_test(self, spec: str, code: str, *, temperature: float = 0.2, think: bool = False) -> dict:
        """LLM writes a self-checking testbench FROM THE SPEC ONLY (expected values derived by the LLM),
        then it is run against `code`. -> {ok, failures: str, tb: str}. The testbench itself may be wrong;
        treat failures as evidence, not truth."""
        ports = await self.interface([code])
        port_txt = ", ".join(f"{p.direction} [{p.width - 1}:0] {p.name}" for p in ports) if ports else "(see spec)"
        tb_reply = await self.ask(
            f"## Specification\n{spec.strip()}\n\nPorts of `{self.top}`: {port_txt}\n\n"
            f"Write a self-checking SystemVerilog testbench `module vfx_tb;` that instantiates `{self.top} dut(...)`, "
            "drives directed corner cases plus some random cases, computes the expected outputs independently "
            "from the specification (do not copy any implementation), and for every mismatch prints "
            "`SELFTEST FAIL: <time> <inputs> expected=<..> got=<..>`. At the end print `SELFTEST DONE errors=<n>` "
            "and call $finish. Return only one ```verilog block.",
            temperature=temperature, think=think)
        tb = _BLOCK_RE.findall(tb_reply)
        tb = tb[-1] if tb else tb_reply
        r = await eda.simulate(code, tb)
        if r["stage"] == "compile":
            return {"ok": None, "failures": "testbench did not compile:\n" + r["log"][:1500], "tb": tb}
        fails = [l for l in r["log"].splitlines() if "SELFTEST FAIL" in l]
        done = re.search(r"SELFTEST DONE errors=(\d+)", r["log"])
        ok = done is not None and int(done.group(1)) == 0 and not fails
        return {"ok": ok, "failures": "\n".join(fails[:15]) if fails else ("" if done else r["log"][-800:]), "tb": tb}


def _record(codes: list[str], stage: str) -> list[str]:
    """Remember produced candidates (evaluator-side diagnostics only; invisible to the workflow)."""
    u = CURRENT_USAGE.get()
    if u is not None:
        u.candidates.extend((stage, c) for c in codes)
    return codes


def _outputs_only(t: str) -> str:
    return "\n".join(l.split(" OUT ", 1)[-1] for l in t.splitlines())


def _side_by_side(ta: str, tb: str, ports, context: int = 12) -> str:
    """Compact table of the steps around the first divergence (both implementations' outputs)."""
    _, out_ports, _, rsts, data = eda.classify_ports(ports)
    ins = [p.name for p in data + rsts]  # exactly the order eda._stim_tb prints them (clock excluded)
    outs = [p.name for p in out_ports]
    la, lb = ta.splitlines(), tb.splitlines()
    first = next((i for i, (x, y) in enumerate(zip(la, lb)) if _outputs_only(x) != _outputs_only(y)), 0)
    lo, hi = max(0, first - context), min(len(la), first + 4)
    rows = [f"columns: step | inputs ({' '.join(ins)}) | "
            f"outputs as [pre-edge | post-edge] (hex): pre-edge = ONE packed word {{{', '.join(outs)}}} "
            f"sampled just before the clock edge; post-edge = {' '.join(outs)} after the edge",
            f"(first difference at step {first})"]
    for x, y in zip(la[lo:hi], lb[lo:hi]):
        step_in = x.split(" OUT ", 1)[0]
        rows.append(f"{step_in}  A: {_outputs_only(x)}   B: {_outputs_only(y)}" +
                    ("   <-- differs" if _outputs_only(x) != _outputs_only(y) else ""))
    return "\n".join(rows)
