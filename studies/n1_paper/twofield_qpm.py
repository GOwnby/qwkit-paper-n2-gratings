"""N1 benchmark, fair version: periodic two-field modulation with BOTH segment fields optimized.
ADQW: segments at (F1, F2), every pair from F in {0, +-20, ..., +-120} kV/cm restricted to fields where the
single-period model is valid (no partially localized subband below E1 + 0.40 eV; see stark_offscan.py).
Triple well (TW): sign reversal between +F and -F, F = 30-90 kV/cm (electrode factor of the 55 kV/cm design).
Pump detunings 0, -10, -20, -30 meV (waveguide recomputed), P = 1.26 and 10 W, ns = 1e11 and 5e11 cm^-2,
LO-phonon relaxation. Writes twofield_qpm.json."""
import sys, os, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, scattering as sc, propagation as pr
import propagation_study as ps
import detuned_qpm as dq
here = os.path.dirname(os.path.abspath(__file__))
AD = [(64, "well"), (10, "barrier"), (29, "well")]
TW = [(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")]
FA = [0] + [s * f for f in range(20, 140, 20) for s in (1, -1)]
FT = list(range(30, 100, 10))
DHW = [0.0, -0.010, -0.020, -0.030]
PS = [1.26, 10.0]
NSI = [2, 4]
WIN = 0.40


def state(args):
    design, I, F = args
    ns = ps.T["states"][I]["ns"]
    L = AD if design == "ad" else TW
    r = sp.self_consistent(L, float(F), ns, maxit=400, nstates=10)
    z, psi, act = r["z"], r["psi"], r["active"]; dz = z[1] - z[0]
    sel = (z >= act[0] - 25) & (z <= act[1] + 25)
    w = np.sum(psi[sel] ** 2, axis=0) * dz
    inwin = (r["E"] - r["E"][0]) < WIN
    valid = not np.any(inwin & (w > 0.15) & (w < 0.8))
    E, psi2, Z, rho = sp.prepare(r)
    N = ns / ((act[1] - act[0] + 200) * 1e-8)
    Et = sp.depol_energies(z, psi2, E, rho, ns, eps_r=13.5)
    st = dict(E=E.tolist(), Z=Z.tolist(), Et=Et.tolist(), rho=rho.tolist(), N=N)
    return dict(design=design, I=I, F=F, valid=bool(valid), st=st, R=sc.rate_matrix(z, psi2, E).tolist())


def run(args):
    kind, I, a, b, dhw, P = args
    hw = ps.HW + dhw; base = dq.wg_at(dhw)
    if kind == "ad":
        m1 = pr.Medium(a["st"], hw, gamma=0.010, R=a["R"]); m2 = pr.Medium(b["st"], hw, gamma=0.010, R=b["R"])
        w1 = dq.wg_for(a["st"], hw, base); w2 = dq.wg_for(b["st"], hw, base)
        r = pr.propagate_segments([(m1, w1["Lcoh_um"], 1.0), (m2, w2["Lcoh_um"], 1.0)], base, P, L_max_um=6000.0)
        return dict(kind=kind, I=I, F1=a["F"], F2=b["F"], dhw=dhw, P=P, eta=float(r["eta"]), L=float(r["L_opt_um"]))
    m = pr.Medium(a["st"], hw, gamma=0.010, R=a["R"]); w = dq.wg_for(a["st"], hw, base)
    r = pr.propagate(m, w, P, "sign", eta_e=ps.ETA_E, L_max_um=6000.0)
    return dict(kind=kind, I=I, F1=a["F"], F2=-a["F"], dhw=dhw, P=P, eta=float(r["eta"]), L=float(r["L_opt_um"]))


if __name__ == "__main__":
    with Pool(2) as pool:
        sts = pool.map(state, [("ad", I, F) for I in NSI for F in FA] + [("tw", I, F) for I in NSI for F in FT])
    jobs = []
    for I in NSI:
        ad = [s for s in sts if s["design"] == "ad" and s["I"] == I and s["valid"]]
        print(f"ns_i {I}: valid ADQW fields {[s['F'] for s in ad]}", flush=True)
        for a, b in itertools.combinations(ad, 2):
            jobs += [("ad", I, a, b, d, P) for d in DHW for P in PS]
        for t in [s for s in sts if s["design"] == "tw" and s["I"] == I]:
            jobs += [("tw", I, t, None, d, P) for d in DHW for P in PS]
    print(len(jobs), "propagations", flush=True)
    with Pool(2) as pool:
        res = pool.map(run, jobs, chunksize=8)
    json.dump(dict(results=res, valid={f"{s['design']}_{s['I']}_{s['F']}": s["valid"] for s in sts}),
              open(os.path.join(here, "twofield_qpm.json"), "w"), indent=1)
    for I in NSI:
        for P in PS:
            for kind in ("ad", "tw"):
                rr = [r for r in res if r["I"] == I and r["P"] == P and r["kind"] == kind]
                line = []
                for d in DHW:
                    b = max([r for r in rr if r["dhw"] == d], key=lambda r: r["eta"])
                    line.append(f"{d*1e3:+.0f}: {b['eta']*100:.3f}% ({b['F1']:+d},{b['F2']:+d})")
                print(f"ns_i {I} P {P} {kind}: " + " | ".join(line), flush=True)
