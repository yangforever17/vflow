"""IO baseline written as stages: one direct generation, no repair, trivial selection.

Behaviour (and LLM call) is identical to workflows/io. The stage split only fixes WHERE changes go:
  generate -> island `generate`, repair -> island `repair`, select -> island `verify`, solve -> island `cascade`.
"""


async def generate(spec, ops):
    """Produce candidate implementations (list of Verilog strings)."""
    return await ops.generate(spec, P.GEN, n=1, temperature=0.2)


async def repair(spec, cands, ops):
    """Revise candidates using feedback. Baseline: no repair."""
    return cands


async def select(spec, cands, ops):
    """Choose the final implementation. Baseline: the first candidate."""
    return cands[0]


async def solve(spec, ops):
    """Control flow between the stages."""
    cands = await generate(spec, ops)
    cands = await repair(spec, cands, ops)
    return await select(spec, cands, ops)
