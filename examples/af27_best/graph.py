"""Compiler repair, a reasoning restart on exhaustion, and semantic review."""

import asyncio
import re


def fix_output_redeclarations(code, errors):
    """Merge simple ANSI output + reg declarations diagnosed by the compiler."""
    # Keep offsets intact while ignoring comments and strings. Avoid macros and
    # multiple modules, where a textual edit could target the wrong declaration.
    masked = re.sub(
        r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',
        lambda m: "".join("\n" if c == "\n" else " " for c in m[0]),
        code, flags=re.S,
    )
    if "`" in masked or len(re.findall(r"\bmodule\b", masked)) != 1:
        return code
    module = re.search(r"\bmodule\s+TopModule\s*\(([^;]*?)\)\s*;", masked)
    if not module:
        return code
    # Only declarations before any procedural or nested scope are eligible.
    body_start = module.end()
    boundary = re.search(r"\b(?:always\w*|initial|function|task|generate|begin|endmodule)\b",
                         masked[body_start:])
    body_end = body_start + boundary.start() if boundary else len(masked)
    width = r"(?:\[\s*\d+\s*:\s*\d+\s*\])?"
    port_re = r"(?:^|,)\s*(output\s+(" + width + r")\s*([A-Za-z_]\w*))\s*(?=,|$)"
    edits = []
    for port in re.finditer(port_re, module[1]):
        name = port[3]
        if "'" + name + "' has already been declared in this scope" not in errors:
            continue
        decl_re = r"\breg\s+(" + width + r")\s*" + re.escape(name) + r"\s*;"
        decls = list(re.finditer(decl_re, masked[body_start:body_end]))
        if len(decls) != 1:
            continue
        decl = decls[0]
        if re.sub(r"\s+", "", port[2]) != re.sub(r"\s+", "", decl[1]):
            continue
        offset = module.start(1) + port.start(1) + len("output")
        edits.append((offset, offset, " reg"))
        edits.append((body_start + decl.start(), body_start + decl.end(), ""))
    for start, end, replacement in sorted(edits, reverse=True):
        code = code[:start] + replacement + code[end:]
    return code


async def generate(spec, ops):
    """Produce candidate implementations (list of Verilog strings)."""
    return await ops.generate(spec, P.GEN, n=1, temperature=0.2)


async def repair(spec, cands, ops):
    """Repair compilation failures only, checking every revision before stopping."""
    code = cands[0]
    seen = {code}
    for attempt in range(3):
        try:
            report = await asyncio.wait_for(ops.lint(code), timeout=90)
            if report["ok"] or attempt == 2:
                break
            patched = fix_output_redeclarations(code, report["errors"])
            if patched != code:
                patched_report = await asyncio.wait_for(ops.lint(patched), timeout=90)
                if patched_report["ok"]:
                    return [patched]
            revisions = await asyncio.wait_for(
                ops.repair(spec, code, report["errors"], n=1, temperature=0.2),
                timeout=180,
            )
            if not revisions or not revisions[0].strip() or revisions[0] in seen:
                break
            code = revisions[0]
            seen.add(code)
        except Exception:
            # An unavailable checker or repair service must not discard the RTL.
            break
    return [code]


async def select(spec, cands, ops):
    """Rescue exhausted compiler repairs; otherwise review compilable RTL once."""
    code = cands[0]
    try:
        report = await asyncio.wait_for(ops.lint(code), timeout=90)
        if not report["ok"]:
            # Both ordinary compiler repairs have already had their opportunity.
            # Re-derive from the spec without anchoring on the broken candidate.
            restarts = await asyncio.wait_for(
                ops.generate(spec, P.GEN, n=1, temperature=0.2, think=True),
                timeout=180,
            )
            if restarts and restarts[0].strip() and restarts[0] != code:
                restart_report = await asyncio.wait_for(ops.lint(restarts[0]), timeout=90)
                if restart_report["ok"]:
                    return restarts[0]
            return code
        feedback = await asyncio.wait_for(
            ops.review(spec, code, temperature=0.0, think=True), timeout=180,
        )
        if not feedback.strip() or re.fullmatch(r"LGTM[.!]?", feedback.strip(), re.I):
            return code
        revisions = await asyncio.wait_for(
            ops.repair(spec, code, feedback, n=1, temperature=0.2), timeout=180,
        )
        if not revisions or not revisions[0].strip() or revisions[0] == code:
            return code
        revised_report = await asyncio.wait_for(ops.lint(revisions[0]), timeout=90)
        if revised_report["ok"]:
            return revisions[0]
    except Exception:
        # Optional rescue/review failures preserve the incumbent RTL.
        pass
    return code


async def solve(spec, ops):
    """Retain the latest candidate if the added review exhausts the time budget."""
    cands = []

    async def run():
        nonlocal cands
        cands = await generate(spec, ops)
        cands = await repair(spec, cands, ops)
        return await select(spec, cands, ops)

    try:
        return await asyncio.wait_for(run(), timeout=840)
    except Exception:
        return cands[0] if cands else ""
