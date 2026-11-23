"""Bounded equivalence checking between two candidate implementations (no reference involved).

`equiv(a, b, ...)` builds a yosys miter of the two designs and runs a bounded SAT proof:
  * equal=True   : no output difference within `steps` clock cycles after reset (bounded proof)
  * equal=False  : a counterexample input sequence was found AND confirmed by iverilog replay
  * equal=None   : inconclusive (yosys could not parse/prove, timeout, or replay did not confirm)

Sequential handling: all FFs start at 0, async resets are converted to sync (`async2sync`), and every
detected reset input is asserted in the first step; outputs are compared from step 2 on.
"""
from __future__ import annotations

import os
import re

from . import eda

_MOD_RE = re.compile(r"\bmodule\s+(\w+)")


def _prefix_modules(code: str, prefix: str) -> tuple[str, dict]:
    names = sorted(set(_MOD_RE.findall(code)), key=len, reverse=True)
    mapping = {n: f"{prefix}{n}" for n in names}
    for n, m in mapping.items():
        code = re.sub(rf"(?<![\w$`]){re.escape(n)}(?!\w)", m, code)
    return code, mapping


def _parse_cex(out: str, steps: int) -> list[dict]:
    """Parse yosys `sat -show-inputs` rows: `<time> \\in_<port> <dec> <hex> <bin>` (last column = binary)."""
    vec = [dict() for _ in range(steps)]
    for line in out.splitlines():
        tok = line.split()
        if len(tok) < 3 or not tok[0].isdigit() or not tok[1].startswith("\\in_"):
            continue
        t, sig, binv = int(tok[0]), tok[1][1:], tok[-1]
        if t < 1 or t > steps or not re.fullmatch(r"[01xXzZ-]+", binv):
            continue
        bits = re.sub(r"[^01]", "0", binv)
        vec[t - 1][sig[3:]] = int(bits, 2) if bits else 0
    return vec


async def equiv(code_a: str, code_b: str, top: str, ports: list | None = None, steps: int = 20,
                timeout: float = 30) -> dict:
    ports = ports or await eda.interface(code_a, top)
    if not ports:
        return {"equal": None, "reason": "no interface"}
    a, ma = _prefix_modules(code_a, "vfxA_")
    b, mb = _prefix_modules(code_b, "vfxB_")
    if top not in ma or top not in mb:
        return {"equal": None, "reason": f"top `{top}` not defined in both"}
    _, _, clk, rsts, _ = eda.classify_ports(ports)
    if clk and re.search(rf"negedge\s+{re.escape(clk.name)}\b", code_a + code_b):
        return {"equal": None, "reason": "negedge clock (single-edge SAT model would be unsound)"}
    sets = []
    for r in rsts:
        sets.append(f"-set-at 1 in_{r.name} {0 if eda._is_active_low(r.name) else 1}")
    skip = "-prove-skip 1" if (clk or rsts) else ""
    nsteps = steps if clk else 1
    script = (f"read_verilog -sv a.sv; read_verilog -sv b.sv; hierarchy; proc; flatten; async2sync; "
              f"opt_clean; miter -equiv -flatten -make_outputs {ma[top]} {mb[top]} vfx_miter; "
              f"hierarchy -top vfx_miter; "
              f"sat -verify -seq {nsteps} -prove trigger 0 {skip} -set-init-zero {' '.join(sets)} "
              f"-show-inputs -timeout {int(timeout * 0.8)} vfx_miter")
    with eda._Tmp() as d:
        open(f"{d}/a.sv", "w").write(a)
        open(f"{d}/b.sv", "w").write(b)
        open(f"{d}/run.ys", "w").write(script.replace("; ", "\n"))
        rc, out = await eda._run([eda.YOSYS, "-s", "run.ys"], d, timeout)
    if "SUCCESS!" in out and "FAIL" not in out:
        return {"equal": True, "steps": nsteps}
    if "model found: FAIL" not in out and "proof did fail" not in out:
        return {"equal": None, "reason": ("timeout" if rc == 124 else "yosys") , "log": eda._clip(out, 800)}
    vectors = _parse_cex(out, nsteps)
    if clk:  # keep running a few more steps after the divergence so the difference is visible
        vectors += [{} for _ in range(4)]
    ta = await eda.trace(code_a, ports, top=top, vectors=vectors)
    tb = await eda.trace(code_b, ports, top=top, vectors=vectors)
    if ta is None or tb is None or _outs(ta) == _outs(tb):
        return {"equal": None, "reason": "counterexample not confirmed by simulation", "vectors": vectors}
    return {"equal": False, "vectors": vectors, "trace_a": ta, "trace_b": tb}


def _outs(t: str) -> list[str]:
    return [l.split(" OUT ", 1)[-1] for l in t.splitlines()]


def first_divergence(trace_a: str, trace_b: str) -> int | None:
    for i, (x, y) in enumerate(zip(_outs(trace_a), _outs(trace_b))):
        if x != y:
            return i
    return None
