"""N2 (6): what the electrode array must provide (ADQW, ns = 5e11 cm^-2, 1.26 W, design pump energy, realistic
fringing fields, index saturation). Electrode pitch 2 um (gap 1 um, breakdown limit 94 kV/cm between neighbours),
4, 8, 16 um (gap 2 um, 188 kV/cm), all valid biases; pitch 4 um without the breakdown limit; pitch 4 um with 1 um gaps (limit 94 kV/cm);
number of distinct voltages at 4 um pitch (K evenly spaced fields over the valid range, nearest valid grid fields, or
the benchmark pair alone). Each case: greedy synthesis over mu with abrupt fields, then
realistic refinement (two sweeps) from the greedy pattern and from the benchmark on the electrode grid; truncated
one electrode beyond the optimum. Writes sat_electrodes.json (finished jobs in parts/electrodes)."""
import os, sys, json, re, time
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import common_n2 as c
import synth as s
from sat_main import on_grid

I = 4; P = 1.26; DX = 0.25; LMAX = 400.0
MAIN = {(r["design"], r["P"]): r for r in json.load(open(os.path.join(here, "sat_main_I4.json")))}


def bench(bank):
    r = MAIN[("ad", P)]
    Fa, Fb = [int(v) for v in re.findall(r"[+-]\d+", r["twofield"]["realistic"]["label"])]
    sa, sb = r["twofield"]["realistic"]["scales"]
    La, Lb = sa * bank.Lcoh[bank.index_of(Fa)], sb * bank.Lcoh[bank.index_of(Fb)]
    return (Fa, Fb), (La, Lb), r["twofield"]["realistic"]["eta"], r["twofield"]["realistic"]["L"]


def levels(bank, K, pair):
    if K == "pair":
        return sorted({bank.index_of(F) for F in pair})
    if K == "all":
        return list(range(len(bank.F)))
    return sorted({bank.nearest(F) for F in np.linspace(bank.F.min(), bank.F.max(), int(K))})


def job(args):
    pitch, K, limit = args[:3]
    t0 = time.time()
    bank = c.Bank("ad", I, 0.0)
    gap = args[3] if len(args) > 3 else (1.0 if pitch < 4 else 2.0)
    dfm = c.dfmax(gap) if limit else np.inf
    pair, Ls, eta_b, L_b = bench(bank)
    allowed = levels(bank, K, pair)
    g, _, _, _ = s.greedy_scan(bank, P, pitch, DX, allowed, LMAX, dfm=dfm)
    n = int(LMAX / sum(Ls)) + 1
    start_b = [bank.nearest(F) for F in on_grid(list(pair) * n, list(Ls) * n, pitch, L_b + c.D_UM)]
    start_b = s.respect(bank, [min(allowed, key=lambda i: abs(bank.F[i] - bank.F[j])) for j in start_b], allowed, dfm)
    best = None
    for k, p0 in (("greedy", g), ("benchmark", start_b)):
        p1, e1 = s.refine_realistic(bank, p0, P, pitch, DX, allowed, sweeps=2, gap_um=gap, dfm=dfm)
        if best is None or e1 > best[1]:
            best = (k, e1, p1)
    pf, ef, Lf = s.finalize(bank, best[2], P, pitch, DX, gap)
    out = dict(pitch=pitch, gap=gap, K=K, limit=limit, n_levels=len(allowed), eta=ef, L=Lf, start=best[0],
               pattern=[int(bank.F[i]) for i in pf], benchmark=eta_b)
    print(f"pitch {pitch:.0f} um gap {gap:.0f} levels {K} ({len(allowed)}) breakdown limit {limit}: {100*ef:.4f}% from {best[0]} "
          f"(benchmark {100*eta_b:.4f}%, x{ef/eta_b:.2f}) "
          f"| {time.time()-t0:.0f} s", flush=True)
    return out


if __name__ == "__main__":
    jobs = [(p, "all", True) for p in (2.0, 4.0, 8.0, 16.0)] + [(4.0, "all", False)]
    jobs += [(4.0, K, True) for K in ("pair", 3, 5, 9)]
    jobs += [(4.0, "all", True, 1.0)]                       # 4 um pitch with 1 um gaps (limit 94 kV/cm)
    import parts
    res = parts.run(job, jobs, "electrodes", int(os.environ.get("NPROC", "2")))
    json.dump(res, open(os.path.join(here, "sat_electrodes.json"), "w"), indent=1)
