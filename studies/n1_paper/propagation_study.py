"""Saturated, pump-depleted SHG in the N1 waveguide for all schemes, doping levels and pump powers.
Writes propagation.json. Inputs: tables.json (self-consistent media), waveguide.json, electrodes.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import propagation as pr
here = os.path.dirname(os.path.abspath(__file__))
HW = 1.23984 / 10
T = json.load(open(os.path.join(here, "tables.json")))
W = [w for w in json.load(open(os.path.join(here, "waveguide.json"))) if w["t_up"] == 2.0 and w["t_sp"] == 0.9][0]
ETA_E = [r for r in json.load(open(os.path.join(here, "electrodes.json")))["results"] if r["gap_um"] == 2.0][0]["rel_ideal"]
WG = dict(n1=W["n1"], n2=W["n2"], G1=W["G1"], G2=W["G2"], O_um=W["O_um"], abg1_cm=W["a1"], abg2_cm=W["a2"],
          t_stack_um=40 * 378e-4, w_um=15.0, Lcoh_um=W["Lcoh_um"], lam1_um=10.0)
PS = list(np.logspace(-2, np.log10(20), 12))

def chi1_re(st, hw):
    """Real linear ISB susceptibility (resonant + antiresonant), dimensionless."""
    E = np.array(st["Et"]); Z = np.array(st["Z"]); rho = np.array(st["rho"]); N = st["N"] * 1e6; g = 0.010
    s = 0j
    for l in range(len(rho)):
        for m in range(l + 1, len(rho)):
            Em = E[m, l]
            s += (rho[l] - rho[m]) * (Z[l, m] * 1e-10) ** 2 * (1 / (Em - hw - 1j * g) + 1 / (Em + hw + 1j * g)) / pr.QE
    return (N * pr.QE ** 2 / pr.EPS0 * s).real

def wg_for(st):
    """Waveguide dict with the coherence length corrected for the medium's own ISB index."""
    w = dict(WG)
    dn1 = WG["G1"] * chi1_re(st, HW) / (2 * WG["n1"]); dn2 = WG["G2"] * chi1_re(st, 2 * HW) / (2 * WG["n2"])
    dn = (WG["n2"] + dn2) - (WG["n1"] + dn1)
    w["Lcoh_um"] = 10.0 / (4 * dn); w["dn_isb"] = (dn1, dn2)
    return w

def job(args):
    ns_i, which, scheme, P = args
    stset = T["states"][ns_i]
    st = stset["tw"] if which == "tw" else stset["ad"]
    med = pr.Medium(st, HW, gamma=0.010, tau_ps=1.0)
    w = wg_for(st)
    r = pr.propagate(med, w, P, scheme, eta_e=ETA_E if scheme == "sign" else 1.0, L_max_um=5000.0)
    return dict(ns=stset["ns"], design=which, scheme=scheme, P=P, eta=float(r["eta"]), L_opt_um=float(r["L_opt_um"]),
                Lcoh_um=w["Lcoh_um"])

if __name__ == "__main__":
    jobs = []
    for i in range(len(T["states"])):
        for P in PS:
            jobs += [(i, "ad", "none", P), (i, "ad", "onoff", P), (i, "ad", "perfect", P), (i, "tw", "sign", P)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=4)
    json.dump(dict(results=res, wg=WG, eta_e=ETA_E), open(os.path.join(here, "propagation.json"), "w"))
    for ns in sorted(set(r["ns"] for r in res)):
        sub = [r for r in res if r["ns"] == ns]
        for key in [("ad", "none"), ("ad", "onoff"), ("ad", "perfect"), ("tw", "sign")]:
            rr = [r for r in sub if (r["design"], r["scheme"]) == key]
            b = max(rr, key=lambda r: r["eta"])
            print(f"ns {ns:.0e} {key[0]}-{key[1]:8s} best eta {b['eta']*100:7.3f}% at P={b['P']:.2f} W, L={b['L_opt_um']:.0f} um (Lcoh {b['Lcoh_um']:.1f})", flush=True)
