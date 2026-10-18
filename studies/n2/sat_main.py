"""N2 (4): electrically programmed gratings in the saturated, pump-depleted regime of N1, against the periodic schemes,
all with the same waveguide, media (index saturation included) and electrode model. Pump at the 10 um design energy;
ADQW and TW; ns = 5e11 (I=4) or 1e11 cm^-2 (I=2, env N2_I); P = 0.01, 1.26 and 10 W (1.26 and 10 W at I=2).
Electrodes: pitch 4 um, gaps 2 um, neighbouring fields within the lateral-breakdown limit (188 kV/cm).
Benchmarks, searched with realistic (fringing) fields and also reported with ideal (abrupt) fields:
  uniform  - one bias everywhere, best bias and length;
  twofield - periodic two-field pattern (ADQW): all ordered pairs of valid fields on a 10 kV/cm grid within the
             breakdown limit, segments one own coherence length; the best five refined by scaling both segment
             lengths (0.4-1.6);
  sign     - TW sign reversal +-F, F = 30-95 kV/cm, both orders, segment lengths refined the same way.
Programmed: greedy synthesis with abrupt fields over the pump weight mu, coordinate-ascent refinement (abrupt
fields), then refinement with realistic fields started from that pattern and from the best benchmark laid out on
the electrode grid; the better is kept and truncated one electrode beyond its optimum length.
Writes sat_main_I{I}.json (finished jobs are kept in parts/main_I{I})."""
import os, sys, json, itertools, time
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import common_n2 as c
import synth as s

I = int(os.environ.get("N2_I", "4"))
PITCH = 4.0
GAP = 2.0
DFM = c.dfmax(GAP)
DX = 0.25 if I == 4 else 0.5
LMAX = 400.0 if I == 4 else 1500.0
SCALES = np.round(np.arange(0.4, 1.61, 0.1), 2)


def _plain(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def stage(key, name, fn):
    """Checkpoint one stage of a job in parts/main_I{I}/<key>.<name>.json, so a job interrupted by a restart resumes."""
    if key is None:
        return fn()
    d = os.path.join(here, "parts", f"main_I{I}"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{key}.{name}.json")
    if os.path.exists(p):
        return json.load(open(p))
    r = fn()
    json.dump(r, open(p + ".tmp", "w"), default=_plain); os.replace(p + ".tmp", p)
    return json.load(open(p))


def periodic(bank, Fa, Fb, L, sa=1.0, sb=1.0):
    la, lb = sa * bank.Lcoh[bank.index_of(Fa)], sb * bank.Lcoh[bank.index_of(Fb)]
    n = int(L / (la + lb)) + 1
    return [Fa, Fb] * n, [la, lb] * n


def ev(bank, F, Ls, P, realistic=True):
    idx = c.realistic_steps(bank, F, Ls, DX, gap_um=GAP) if realistic else c.ideal_steps(bank, F, Ls, DX)
    r = c.propagate(bank, idx, DX, P)
    return r["eta"], r["L_opt_um"]


def truncate_segments(bank, F, Ls, L, P):
    """Keep the segments up to one beyond the optimum length L and re-evaluate (realistic)."""
    e = np.cumsum(Ls); k = min(len(F), int(np.searchsorted(e, L)) + 2)
    eta, Lo = ev(bank, F[:k], Ls[:k], P)
    return F[:k], Ls[:k], eta, Lo


def benchmark(bank, pairs, P, key=None):
    def scan():
        rows = []
        for a, b in pairs:
            F, Ls = periodic(bank, a, b, LMAX)
            eta, L = ev(bank, F, Ls, P)
            rows.append((eta, L, a, b, 1.0, 1.0))
        rows.sort(key=lambda t: -t[0])
        return rows
    rows = [tuple(r) for r in stage(key, "pairs", scan)]
    best = rows[0]

    def scale(a, b):
        out = None
        for sa in SCALES:
            for sb in SCALES:
                F, Ls = periodic(bank, a, b, LMAX, sa, sb)
                eta, L = ev(bank, F, Ls, P)
                if out is None or eta > out[0]:
                    out = (eta, L, a, b, float(sa), float(sb))
        return out
    for k, (_, _, a, b, _, _) in enumerate(rows[:5]):
        r = tuple(stage(key, f"scale{k}", lambda: scale(a, b)))
        if r[0] > best[0]:
            best = r
    eta, L, a, b, sa, sb = best
    F, Ls = periodic(bank, a, b, LMAX, sa, sb)
    F, Ls, eta, L = truncate_segments(bank, F, Ls, L, P)
    ide = stage(key, "ideal", lambda: max(((lambda r: (r[0], r[1], x, y))(ev(bank, *periodic(bank, x, y, LMAX), P, realistic=False)) for x, y in pairs)))
    return dict(realistic=dict(eta=eta, L=L, label=f"({a:+.0f},{b:+.0f})", scales=[sa, sb], unscaled=rows[0][0],
                               unscaled_label=f"({rows[0][2]:+.0f},{rows[0][3]:+.0f})"),
                ideal=dict(eta=ide[0], L=ide[1], label=f"({ide[2]:+.0f},{ide[3]:+.0f})",
                           its_realistic=ev(bank, *periodic(bank, ide[2], ide[3], LMAX), P)[0]),
                _seg=(F, Ls))


def on_grid(F_seg, L_seg, pitch, L):
    """Discretize consecutive segments on the electrode grid (each electrode takes the field at its centre)."""
    e = np.cumsum(L_seg); n = max(int(L / pitch), 1)
    return [F_seg[min(int(np.searchsorted(e, (j + 0.5) * pitch)), len(F_seg) - 1)] for j in range(n)]


def job(args):
    design, P = args
    t0 = time.time()
    bank = c.Bank(design, I, 0.0)
    allowed = list(range(len(bank.F)))
    out = dict(design=design, I=I, P=P, pitch=PITCH, gap=GAP, dfmax=DFM, dx=DX, index_saturation=c.INDEX_SAT)
    un = max((ev(bank, [F], [LMAX], P) + (F,) for F in bank.F))
    out["uniform"] = dict(realistic=dict(eta=un[0], L=un[1], label=f"{un[2]:+.0f}"))
    if design == "ad":
        grid = [F for F in bank.F if int(F) % 10 == 0]
        pairs = [(a, b) for a, b in itertools.permutations(grid, 2) if abs(a - b) <= DFM]
        key = "twofield"
    else:
        grid = [F for F in bank.F if 30 <= F <= 95]
        pairs = [(F, -F) for F in grid if 2 * F <= DFM] + [(-F, F) for F in grid if 2 * F <= DFM]
        key = "sign"
    jk = f"{design}_{P}"
    out[key] = benchmark(bank, pairs, P, jk)
    Fb, Lsb = out[key].pop("_seg")
    # programmed
    pat, eta_g, mu, scan = stage(jk, "greedy", lambda: s.greedy_scan(bank, P, PITCH, DX, allowed, LMAX, dfm=DFM))
    pat_i, eta_i = stage(jk, "refine", lambda: s.refine(bank, pat, P, PITCH, DX, allowed, sweeps=2, dfm=DFM))
    starts = {"refined": pat_i,
              "benchmark": s.respect(bank, [bank.index_of(F) for F in on_grid(Fb, Lsb, PITCH, out[key]["realistic"]["L"] + c.D_UM)], allowed, DFM)}
    allowed_r = allowed if I == 4 else [i for i in allowed if int(bank.F[i]) % 10 == 0]
    res = {}
    for k, p0 in starts.items():
        e0, p1, e1 = stage(jk, f"real_{k}", lambda: (s.eval_realistic(bank, p0, P, PITCH, DX, GAP)["eta"],)
                           + tuple(s.refine_realistic(bank, p0, P, PITCH, DX, allowed_r, sweeps=2 if I == 4 else 1, gap_um=GAP, dfm=DFM)))
        res[k] = dict(eta_start=e0, eta=e1, pattern=p1)
    kbest = max(res, key=lambda k: res[k]["eta"])
    pf, ef, Lf = s.finalize(bank, res[kbest]["pattern"], P, PITCH, DX, GAP)
    rb = s.eval_realistic(bank, pf, P, PITCH, DX, GAP, record=True)
    out["programmed"] = dict(greedy=dict(eta=eta_g, mu=mu, scan=scan),
                             ideal=dict(eta=eta_i, pattern=[int(bank.F[i]) for i in pat_i], L=len(pat_i) * PITCH),
                             realistic=dict(eta=ef, start=kbest, L=Lf, pattern=[int(bank.F[i]) for i in pf],
                                            starts={k: dict(eta_start=v["eta_start"], eta=v["eta"]) for k, v in res.items()},
                                            x_um=rb["x_um"][::4].tolist(), P2=rb["P2"][::4].tolist(), P1=rb["P1"][::4].tolist()))
    best_other = max(out[key]["realistic"]["eta"], out["uniform"]["realistic"]["eta"])
    out["gain"] = ef / out[key]["realistic"]["eta"]; out["gain_vs_best_unprogrammed"] = ef / best_other
    print(f"I {I} {design} P {P}: uniform {100*un[0]:.4f}% | {key} realistic {100*out[key]['realistic']['eta']:.4f}% "
          f"{out[key]['realistic']['label']} x{out[key]['realistic']['scales']} (unscaled best {100*out[key]['realistic']['unscaled']:.4f}%), "
          f"ideal {100*out[key]['ideal']['eta']:.4f}% {out[key]['ideal']['label']} | programmed ideal {100*eta_i:.4f}% (greedy "
          f"{100*eta_g:.4f}% mu {mu}), realistic {100*ef:.4f}% from {kbest} (L {Lf:.0f} um) | x{out['gain']:.2f} vs {key}, "
          f"x{out['gain_vs_best_unprogrammed']:.2f} vs best unprogrammed | {time.time()-t0:.0f} s", flush=True)
    return out


if __name__ == "__main__":
    jobs = [(d, P) for d in ("ad", "tw") for P in ((1.26, 10.0, 0.01) if I == 4 else (1.26, 10.0))]
    import parts
    res = parts.run(job, jobs, f"main_I{I}", int(os.environ.get("NPROC", "2")))
    json.dump(res, open(os.path.join(here, f"sat_main_I{I}.json"), "w"), indent=1)
