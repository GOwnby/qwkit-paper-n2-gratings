"""N2 (5): tuning and acceptance at 1.26 W (ns = 5e11 cm^-2), realistic fringing fields, index saturation, pitch 4 um,
gaps 2 um and the lateral-breakdown limit throughout. For pump detunings dhw = -40 ... +10 meV from the 10 um design
(waveguide recomputed at each pump energy):
  fixed device      - the best design-energy benchmark of sat_main_I4.json (ADQW two-field or TW sign, with its
                      segment lengths); (a) its design fields kept (acceptance), (b) both fields re-chosen at each dhw
                      from all valid fields on a 10 kV/cm grid within the breakdown limit (retuned);
  uniform           - best single bias at each dhw;
  programmed, fixed - the design-energy programmed pattern of sat_main_I4.json, unchanged (acceptance);
  reprogrammed      - greedy synthesis at each dhw, refined with realistic fields from the greedy pattern, from the
                      retuned fixed device on the electrode grid and from the design pattern; the best is kept.
Writes sat_tuning.json (finished jobs in parts/tuning)."""
import os, sys, json, itertools, time, re
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import common_n2 as c
import synth as s
from sat_main import on_grid, ev, PITCH, GAP, DFM

I = 4; P = 1.26; DX = 0.25; LMAX = 400.0
DHW = [-0.040, -0.030, -0.020, -0.010, 0.0, 0.010]
MAIN = {(r["design"], r["P"]): r for r in json.load(open(os.path.join(here, "sat_main_I4.json")))}


def design_geometry(design):
    r = MAIN[(design, P)]
    key = "twofield" if design == "ad" else "sign"
    Fa, Fb = [int(v) for v in re.findall(r"[+-]\d+", r[key]["realistic"]["label"])]
    sa, sb = r[key]["realistic"]["scales"]
    b0 = c.Bank(design, I, 0.0)
    return (Fa, Fb), (sa * b0.Lcoh[b0.index_of(Fa)], sb * b0.Lcoh[b0.index_of(Fb)]), r["programmed"]["realistic"]["pattern"]


def fixed(bank, F2, L2):
    n = int(LMAX / sum(L2)) + 1
    return ev(bank, list(F2) * n, list(L2) * n, P)


def job(args):
    design, dhw = args
    t0 = time.time()
    bank = c.Bank(design, I, dhw)
    allowed = list(range(len(bank.F)))
    (Fa, Fb), (La, Lb), prog0 = design_geometry(design)
    out = dict(design=design, dhw=dhw)
    out["fixed_design_fields"] = fixed(bank, (Fa, Fb), (La, Lb))[0]
    grid = [F for F in bank.F if int(F) % 10 == 0]
    rt = max(((lambda r: (r[0], a, b, r[1]))(fixed(bank, (a, b), (La, Lb)))
              for a, b in itertools.permutations(grid, 2) if abs(a - b) <= DFM))
    out["fixed_retuned"] = dict(eta=rt[0], fields=[rt[1], rt[2]], L=rt[3])
    out["uniform"] = max(ev(bank, [F], [LMAX], P)[0] for F in bank.F)
    p_design = [bank.nearest(F) for F in prog0]
    out["programmed_fixed"] = s.eval_realistic(bank, p_design, P, PITCH, DX, GAP)["eta"]
    g, _, _, _ = s.greedy_scan(bank, P, PITCH, DX, allowed, LMAX, dfm=DFM)
    n = int(LMAX / (La + Lb)) + 1
    starts = {"greedy": g,
              "fixed_retuned": s.respect(bank, [bank.index_of(F) for F in on_grid([rt[1], rt[2]] * n, [La, Lb] * n, PITCH, rt[3] + c.D_UM)], allowed, DFM),
              "design": p_design}
    best = None
    for k, p0 in starts.items():
        p1, e1 = s.refine_realistic(bank, p0, P, PITCH, DX, allowed, sweeps=1, gap_um=GAP, dfm=DFM)
        if best is None or e1 > best[1]:
            best = (k, e1, p1)
    pf, ef, Lf = s.finalize(bank, best[2], P, PITCH, DX, GAP)
    out["reprogrammed"] = dict(eta=ef, start=best[0], L=Lf, pattern=[int(bank.F[i]) for i in pf])
    print(f"{design} dhw {1e3*dhw:+.0f} meV: fixed(design fields) {100*out['fixed_design_fields']:.4f}% retuned "
          f"{100*rt[0]:.4f}% ({rt[1]:+.0f},{rt[2]:+.0f}) | uniform {100*out['uniform']:.4f}% | programmed(fixed) "
          f"{100*out['programmed_fixed']:.4f}% reprogrammed {100*ef:.4f}% ({best[0]}) | {time.time()-t0:.0f} s", flush=True)
    return out


if __name__ == "__main__":
    jobs = [(d, x) for d in ("ad", "tw") for x in DHW]
    import parts
    res = parts.run(job, jobs, "tuning", int(os.environ.get("NPROC", "2")))
    json.dump(res, open(os.path.join(here, "sat_tuning.json"), "w"), indent=1)
