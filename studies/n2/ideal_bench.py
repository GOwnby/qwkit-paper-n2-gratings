"""N2 (4b): the benchmarks under ideal (abrupt) fields, optimized exactly as the realistic ones in sat_main.py: every
pair (ADQW, 10 kV/cm grid within the breakdown limit) or sign reversal (TW), then the best five refined by scaling
both segment lengths 0.4-1.6, then truncated one segment beyond the optimum. ns from env N2_I (4 or 2). Writes
ideal_bench_I{I}.json (finished jobs in parts/ideal_I{I}); tables.py uses it for the ideal benchmark column."""
import os, sys, json, itertools, time
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import common_n2 as c
from sat_main import I, LMAX, SCALES, DFM, periodic, ev, stage


def job(args):
    design, P = args
    t0 = time.time()
    bank = c.Bank(design, I, 0.0)
    if design == "ad":
        grid = [F for F in bank.F if int(F) % 10 == 0]
        pairs = [(a, b) for a, b in itertools.permutations(grid, 2) if abs(a - b) <= DFM]
    else:
        grid = [F for F in bank.F if 30 <= F <= 95]
        pairs = [(F, -F) for F in grid if 2 * F <= DFM] + [(-F, F) for F in grid if 2 * F <= DFM]
    jk = f"{design}_{P}"
    rows = [tuple(r) for r in stage(jk, "ideal_pairs", lambda: sorted(((ev(bank, *periodic(bank, a, b, LMAX), P, realistic=False) + (a, b))
                                                                     for a, b in pairs), key=lambda t: -t[0]))]
    best = rows[0] + (1.0, 1.0)

    def scale(a, b):
        out = None
        for sa in SCALES:
            for sb in SCALES:
                r = ev(bank, *periodic(bank, a, b, LMAX, sa, sb), P, realistic=False)
                if out is None or r[0] > out[0]:
                    out = (r[0], r[1], a, b, float(sa), float(sb))
        return out
    for k, (_, _, a, b) in enumerate(rows[:5]):
        r = tuple(stage(jk, f"ideal_scale{k}", lambda: scale(a, b)))
        if r[0] > best[0]:
            best = r
    eta, L, a, b, sa, sb = best
    F, Ls = periodic(bank, a, b, LMAX, sa, sb)
    e = 0.0; k = len(F)
    for j, l in enumerate(Ls):
        e += l
        if e > L:
            k = min(len(F), j + 2); break
    eta_t, L_t = ev(bank, F[:k], Ls[:k], P, realistic=False)
    out = dict(design=design, I=I, P=P, eta=eta_t, L=L_t, label=f"({a:+.0f},{b:+.0f})", scales=[sa, sb], unscaled=rows[0][0],
               unscaled_label=f"({rows[0][2]:+.0f},{rows[0][3]:+.0f})")
    print(f"I {I} {design} P {P}: ideal benchmark {100*eta_t:.4f}% {out['label']} x{out['scales']} L {L_t:.0f} um "
          f"(unscaled {100*rows[0][0]:.4f}% {out['unscaled_label']}) | {time.time()-t0:.0f} s", flush=True)
    return out


if __name__ == "__main__":
    jobs = [(d, P) for d in ("ad", "tw") for P in ((1.26, 10.0, 0.01) if I == 4 else (1.26, 10.0))]
    import parts
    res = parts.run(job, jobs, f"ideal_I{I}", int(os.environ.get("NPROC", "2")))
    json.dump(res, open(os.path.join(here, f"ideal_bench_I{I}.json"), "w"), indent=1)
