"""N2 (8): anatomy of the programmed patterns against the benchmarks (realistic fields, 5e11 cm^-2, design energy):
mean in-phase chi(2) over the device, its spread, pump power left at the optimum length, fraction of the length
with the field within 20 kV/cm of resonance (ADQW: |F| < 20). Writes anatomy.json."""
import os, sys, json, re
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import common_n2 as c
from figs_n2 import profile

M = {(r["design"], r["P"]): r for r in json.load(open(os.path.join(here, "sat_main_I4.json")))}
out = []
for d in ("ad", "tw"):
    for P in (1.26, 10.0):
        r = M[(d, P)]; key = "twofield" if d == "ad" else "sign"
        b = c.Bank(d, 4, 0.0)
        Fa, Fb = [int(v) for v in re.findall(r"[+-]\d+", r[key]["realistic"]["label"])]
        sa, sb = r[key]["realistic"]["scales"]
        la, lb = sa * b.Lcoh[b.index_of(Fa)], sb * b.Lcoh[b.index_of(Fb)]
        nb = int(np.ceil(r[key]["realistic"]["L"] / (la + lb))) + 1
        for name, F, L in (("programmed", r["programmed"]["realistic"]["pattern"], r["pitch"]), ("benchmark", [Fa, Fb] * nb, [la, lb] * nb)):
            rr, Fc, ip = profile(b, F, L, P, gap=r["gap"])
            n = int(round(rr["L_opt_um"] / 0.25))
            row = dict(design=d, P=P, case=name, L=rr["L_opt_um"], eta=rr["eta"], inphase_mean_nm=float(np.mean(ip[:n]) * 1e9),
                       P1_left=float(rr["P1"][n] / P), near_res=float(np.mean(np.abs(Fc[:n]) < 20)) if d == "ad" else None,
                       field_range=[float(Fc[:n].min()), float(Fc[:n].max())])
            out.append(row)
            print(d, P, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("design", "P", "case")}, flush=True)
json.dump(out, open(os.path.join(here, "anatomy.json"), "w"), indent=1)
