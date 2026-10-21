"""N1 follow-up (LATER_STAGE item 10): the QPM-scheme comparison of Table 3 repeated with the pump red-detuned from
resonance, ns = 5e11 cm^-2 (env N1_NS_INDEX=4, default) or another doping of tables.json, LO-phonon relaxation. For each pump photon energy hw = HW + dhw the waveguide (modes,
confinement, overlap, background loss) is the NL-02 recomputation at that wavelength (studies/nl02/hw_scan_wg.json),
the coherence length includes each medium's own ISB index at hw and 2hw, and the realistic Stark 'off' state is
re-selected at hw (smallest |F| with |chi2(hw)| <= 10% of the 0 V value). Schemes: TW sign reversal (with the
electrode efficiency), ADQW realistic Stark on/off, ADQW idealized on/off, ADQW growth reversal, ADQW perfect PM.
Writes detuned_qpm.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, scattering as sc, propagation as pr
import propagation_study as ps
here = os.path.dirname(os.path.abspath(__file__))
AD = [(64, "well"), (10, "barrier"), (29, "well")]
RATES = json.load(open(os.path.join(here, "rates.json")))
WGS = json.load(open(os.path.join(here, "..", "nl02", "hw_scan_wg.json")))
I5 = int(os.environ.get("N1_NS_INDEX", 4)); NS = ps.T["states"][I5]["ns"]
DHW = [0.0, -0.010, -0.020, -0.025, -0.030, -0.035, -0.040]
PS = [1.26, 4.0, 10.0]
GAM = 0.010


def R_for(d):
    return [r for r in RATES if r["i"] == I5 and r["design"] == d][0]["R"]


def wg_at(dhw):
    k = min(WGS, key=lambda s: abs(float(s) - dhw))
    assert abs(float(k) - dhw) < 1e-4, k
    return dict(WGS[k])


def chi1_re(st, hw):
    E = np.array(st["Et"]); Z = np.array(st["Z"]); rho = np.array(st["rho"]); N = st["N"] * 1e6
    s = 0j
    for l in range(len(rho)):
        for m in range(l + 1, len(rho)):
            Em = E[m, l]
            s += (rho[l] - rho[m]) * (Z[l, m] * 1e-10) ** 2 * (1 / (Em - hw - 1j * GAM) + 1 / (Em + hw + 1j * GAM)) / pr.QE
    return (N * pr.QE ** 2 / pr.EPS0 * s).real


def wg_for(st, hw, base):
    w = dict(base)
    dn1 = w["G1"] * chi1_re(st, hw) / (2 * w["n1"]); dn2 = w["G2"] * chi1_re(st, 2 * hw) / (2 * w["n2"])
    dn = (w["n2"] + dn2) - (w["n1"] + dn1)
    w["Lcoh_um"] = w["lam1_um"] / (4 * dn)
    return w


def stark_state(F):
    o = sp.chi2_sc(AD, float(F), ps.HW, ns_cm2=NS)
    E, psi, Z, rho = sp.prepare(o["r"])
    st = dict(E=o["E"].tolist(), Z=o["Z"].tolist(), Et=o["Et"].tolist(), rho=o["rho"].tolist(), N=o["N"])
    return dict(F=F, st=st, R=sc.rate_matrix(o["r"]["z"], psi, E).tolist())


def chi_abs(st, hw):
    return abs(sp.chi2_depol(np.array(st["Et"]), np.array(st["Z"]), np.array(st["rho"]), hw, GAM, st["N"]))


def job(a):
    scheme, P, dhw, off = a
    hw = ps.HW + dhw; base = wg_at(dhw)
    if scheme == "tw_sign":
        st = ps.T["states"][I5]["tw"]; m = pr.Medium(st, hw, gamma=GAM, R=R_for("tw")); w = wg_for(st, hw, base)
        r = pr.propagate(m, w, P, "sign", eta_e=ps.ETA_E, L_max_um=6000.0)
    elif scheme == "ad_stark":
        st = ps.T["states"][I5]["ad"]
        m_on = pr.Medium(st, hw, gamma=GAM, R=R_for("ad")); m_off = pr.Medium(off["st"], hw, gamma=GAM, R=off["R"])
        w_on = wg_for(st, hw, base); w_off = wg_for(off["st"], hw, base)
        r = pr.propagate_segments([(m_on, w_on["Lcoh_um"], 1.0), (m_off, w_off["Lcoh_um"], 1.0)], base, P, L_max_um=6000.0)
    else:
        st = ps.T["states"][I5]["ad"]; m = pr.Medium(st, hw, gamma=GAM, R=R_for("ad")); w = wg_for(st, hw, base)
        sch = {"ad_onoff": "onoff", "ad_rev": "sign", "ad_pm": "perfect", "ad_none": "none"}[scheme]
        r = pr.propagate(m, w, P, sch, eta_e=1.0, L_max_um=6000.0)
    return dict(scheme=scheme, P=P, dhw=dhw, eta=float(r["eta"]), L_opt_um=float(r["L_opt_um"]),
                F_off=None if off is None else off["F"])


if __name__ == "__main__":
    Fs = [f for Fa in range(10, 260, 10) for f in (Fa, -Fa)]
    with Pool(2) as pool:
        stark = pool.map(stark_state, Fs)
    on = ps.T["states"][I5]["ad"]
    offs = {}
    for dhw in DHW:
        hw = ps.HW + dhw; c0 = chi_abs(on, hw)
        offs[dhw] = next(s for s in stark if chi_abs(s["st"], hw) <= 0.1 * c0)
        print(f"dhw {dhw*1e3:+.0f} meV: Stark off state at F = {offs[dhw]['F']} kV/cm", flush=True)
    jobs = [(s, P, d, offs[d] if s == "ad_stark" else None) for s in ["tw_sign", "ad_stark", "ad_onoff", "ad_rev", "ad_pm"]
            for P in PS for d in DHW]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=2)
    json.dump(dict(results=res, ns=NS, eta_e=ps.ETA_E, F_off={str(k): v["F"] for k, v in offs.items()}),
              open(os.path.join(here, "detuned_qpm.json" if I5 == 4 else f"detuned_qpm_ns{I5}.json"), "w"), indent=1)
    for P in PS:
        print(f"P = {P} W")
        for s in ["tw_sign", "ad_stark", "ad_onoff", "ad_rev", "ad_pm"]:
            rr = sorted([r for r in res if r["scheme"] == s and r["P"] == P], key=lambda r: -r["dhw"])
            print(f"  {s:9s} " + " ".join(f"{r['dhw']*1e3:+.0f}:{r['eta']*100:.3f}%" for r in rr), flush=True)
