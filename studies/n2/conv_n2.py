"""N2 (7): numerical checks of the realistic evaluation for the programmed and benchmark patterns (ADQW and TW,
5e11 cm^-2, 1.26 W): step 0.25 vs 0.125 um, three vs five heights in the stack (Simpson weights), and the
field grid (nearest computed bias, 5 kV/cm) against nearest-of-10 kV/cm (an upper estimate of the rounding error),
and index saturation switched off (N1 convention).
Writes conv_n2.json."""
import os, sys, json, re
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import common_n2 as c
import synth as s

M = {(r["design"], r["P"]): r for r in json.load(open(os.path.join(here, "sat_main_I4.json")))}
out = []
for d in ("ad", "tw"):
    r = M[(d, 1.26)]
    key = "twofield" if d == "ad" else "sign"
    b = c.Bank(d, 4, 0.0)
    Fa, Fb = [int(v) for v in re.findall(r"[+-]\d+", r[key]["realistic"]["label"])]
    sa, sb = r[key]["realistic"]["scales"]
    la, lb = sa * b.Lcoh[b.index_of(Fa)], sb * b.Lcoh[b.index_of(Fb)]
    nb = int(np.ceil(r[key]["realistic"]["L"] / (la + lb))) + 1
    cases = {"programmed": (r["programmed"]["realistic"]["pattern"], r["pitch"]), "benchmark": ([Fa, Fb] * nb, [la, lb] * nb)}
    for name, (F, L) in cases.items():
        row = dict(design=d, case=name)
        for dx in (0.25, 0.125):
            row[f"dx{dx}"] = c.propagate(b, c.realistic_steps(b, F, L, dx, gap_um=r["gap"]), dx, 1.26)["eta"]
        y3, w3 = c.Y_UM, c.YW
        c.Y_UM = tuple(np.linspace(y3[0], y3[-1], 5)); c.YW = (1 / 12, 4 / 12, 2 / 12, 4 / 12, 1 / 12)
        b5 = c.Bank(d, 4, 0.0)
        row["5heights"] = c.propagate(b5, c.realistic_steps(b5, F, L, 0.25, gap_um=r["gap"]), 0.25, 1.26)["eta"]
        c.Y_UM, c.YW = y3, w3
        b10 = c.Bank(d, 4, 0.0)
        keep = [i for i in range(len(b10.F)) if int(b10.F[i]) % 10 == 0]
        b10.nearest = lambda Fv, bb=b10, kk=keep: kk[int(np.argmin(np.abs(bb.F[kk] - Fv)))]
        row["grid10"] = c.propagate(b10, c.realistic_steps(b10, F, L, 0.25, gap_um=r["gap"]), 0.25, 1.26)["eta"]
        c.INDEX_SAT = False
        bn = c.Bank(d, 4, 0.0)
        row["no_index_sat"] = c.propagate(bn, c.realistic_steps(bn, F, L, 0.25, gap_um=r["gap"]), 0.25, 1.26)["eta"]
        c.INDEX_SAT = True
        out.append(row)
        print(d, name, {k: (round(100 * v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("design", "case")}, flush=True)
json.dump(out, open(os.path.join(here, "conv_n2.json"), "w"), indent=1)
