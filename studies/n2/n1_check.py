"""N2 (9): the N1 electrode-factor question in N1's own conventions (no index saturation, no breakdown limit, segment
lengths one coherence length, 2 um gaps): best periodic two-field ADQW pattern and TW sign reversal with realistic
fields, against N1's values (studies/n1_paper/twofield_qpm.json, design energy, ns = 5e11 cm^-2, eta_e-scaled
for the ADQW as in twofield_best; the TW values already include the electrode factor). Also N1's pair with ideal fields (reproduces N1 without the eta_e scaling) and the best pair re-chosen with realistic fields. Writes n1_check.json."""
import os, sys, json, itertools
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import common_n2 as c
c.INDEX_SAT = False
DX = 0.25; LMAX = 400.0
N1A = [0] + [sg * f for f in range(20, 140, 20) for sg in (1, -1)]
N1T = list(range(30, 100, 10))
A = json.load(open(os.path.join(here, "..", "n1_paper", "twofield_qpm.json")))["results"]
ETA_E = json.load(open(os.path.join(here, "..", "n1_paper", "twofield_best.json")))


def per(bank, a, b, real):
    la, lb = bank.Lcoh[bank.index_of(a)], bank.Lcoh[bank.index_of(b)]
    n = int(LMAX / (la + lb)) + 1
    F, L = [a, b] * n, [la, lb] * n
    idx = c.realistic_steps(bank, F, L, DX) if real else c.ideal_steps(bank, F, L, DX)
    return c.propagate(bank, idx, DX, P)["eta"]


out = []
for P in (1.26, 10.0):
    for d in ("ad", "tw"):
        bank = c.Bank(d, 4, 0.0)
        if d == "ad":
            val = [F for F in N1A if np.any(bank.F == F)]
            pairs = list(itertools.permutations(val, 2))
        else:
            pairs = [(F, -F) for F in N1T] + [(-F, F) for F in N1T]
        rr = sorted(((per(bank, a, b, True), a, b) for a, b in pairs), reverse=True)
        n1 = max([r for r in A if r["I"] == 4 and r["P"] == P and r["kind"] == d and r["dhw"] == 0.0], key=lambda r: r["eta"])
        n1_real = per(bank, n1["F1"], n1["F2"], True)
        row = dict(P=P, design=d, best_realistic=rr[0][0], best_pair=[rr[0][1], rr[0][2]], n1_eta=n1["eta"], n1_pair=[n1["F1"], n1["F2"]],
                   n1_pair_realistic=n1_real, n1_pair_ideal=per(bank, n1["F1"], n1["F2"], False))
        if d == "ad":
            b = [x for x in ETA_E if x["I"] == 4 and x["P"] == P][0]["ad_best"]
            row["n1_eta_scaled"] = b["eta_e_applied"] if b["dhw"] == 0.0 else None
        out.append(row)
        print(f"P {P} {d}: N1 best at design energy {100*n1['eta']:.4f}% {n1['F1']:+d},{n1['F2']:+d} (that pair with realistic fields "
              f"{100*n1_real:.4f}%, ideal fields {100*row['n1_pair_ideal']:.4f}%) | best pair with realistic fields {100*rr[0][0]:.4f}% ({rr[0][1]:+.0f},{rr[0][2]:+.0f})", flush=True)
json.dump(out, open(os.path.join(here, "n1_check.json"), "w"), indent=1)
