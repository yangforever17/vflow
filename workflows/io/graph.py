"""Round-1 / IO baseline: one direct generation, no checks."""


async def solve(spec, ops):
    return (await ops.generate(spec, P.GEN, n=1, temperature=0.2))[0]
