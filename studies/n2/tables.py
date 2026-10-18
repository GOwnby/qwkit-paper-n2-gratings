"""N2 manuscript table (paper_n2/table_main.tex) and the numbers quoted in the text (numbers.log, numbers.json),
from sat_main_I4.json, sat_main_I2.json, sat_tuning.json, sat_electrodes.json and conv_n2.json."""
import os, json
here = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(here, f))) if os.path.exists(os.path.join(here, f)) else None
out = {}; lines = []; rows = []
for I, ns in ((4, r"$5\times10^{11}$"), (2, r"$10^{11}$")):
    d = J(f"sat_main_I{I}.json")
    if d is None:
        continue
    ib = {(x["design"], x["P"]): x for x in (J(f"ideal_bench_I{I}.json") or [])}
    first = True
    for design in ("ad", "tw"):
        for P in (0.01, 1.26, 10.0):
            r = [x for x in d if x["design"] == design and x["P"] == P]
            if not r:
                continue
            r = r[0]; k = "twofield" if design == "ad" else "sign"; pr = r["programmed"]
            if (design, P) in ib:   # ideal benchmark optimized like the realistic one (segment scaling, truncation)
                x = ib[(design, P)]
                r[k]["ideal"].update(eta=x["eta"], L=x["L"], label=x["label"], scales=x["scales"])
            g = pr["realistic"]["eta"] / r[k]["realistic"]["eta"]
            gb = r["gain_vs_best_unprogrammed"]
            rows.append(f"{ns if first else ''} & {'ADQW' if design == 'ad' else 'TW'} & {P:g} & "
                        f"{100*r[k]['ideal']['eta']:#.3g} ({r[k]['ideal']['L']:.0f}) & {100*r[k]['realistic']['eta']:#.3g} ({r[k]['realistic']['L']:.0f}) & "
                        f"{100*pr['ideal']['eta']:#.3g} ({pr['ideal']['L']:.0f}) & {100*pr['realistic']['eta']:#.3g} ({pr['realistic']['L']:.0f}) & "
                        f"{100*r['uniform']['realistic']['eta']:#.3g} & {gb:.2f} \\\\")
            first = False
            key = f"I{I}_{design}_{P:g}"
            out[key] = dict(bench_ideal=r[k]["ideal"]["eta"], bench_real=r[k]["realistic"]["eta"], bench_label=r[k]["realistic"]["label"],
                            prog_ideal=pr["ideal"]["eta"], prog_real=pr["realistic"]["eta"], uniform=r["uniform"]["realistic"]["eta"],
                            gain=g, gain_vs_best_unprogrammed=gb, gain_ideal=pr["ideal"]["eta"] / r[k]["ideal"]["eta"],
                            bench_scales=r[k]["realistic"]["scales"], bench_ideal_label=r[k]["ideal"]["label"],
                            bench_ideal_pair_realistic=r[k]["ideal"]["its_realistic"],
                            fringe_loss_bench=1 - r[k]["realistic"]["eta"] / r[k]["ideal"]["eta"],
                            fringe_loss_prog=1 - pr["realistic"]["eta"] / pr["ideal"]["eta"], start=pr["realistic"]["start"])
            lines.append(f"{key}: bench {r[k]['realistic']['label']} x{r[k]['realistic']['scales']} (ideal-best pair {r[k]['ideal']['label']} realistic {100*r[k]['ideal']['its_realistic']:.4f}%) ideal {100*r[k]['ideal']['eta']:.4f}% real {100*r[k]['realistic']['eta']:.4f}% | "
                         f"prog ideal {100*pr['ideal']['eta']:.4f}% real {100*pr['realistic']['eta']:.4f}% (start {pr['realistic']['start']}) | "
                         f"uniform {100*r['uniform']['realistic']['eta']:.4f}% | gain real {g:.2f} (vs best unprogrammed {gb:.2f}) ideal {out[key]['gain_ideal']:.2f} | "
                         f"fringe loss bench {100*out[key]['fringe_loss_bench']:.0f}% prog {100*out[key]['fringe_loss_prog']:.0f}%")
    rows.append(r"\midrule")
if rows and rows[-1] == r"\midrule":
    rows.pop()
open(os.path.join(here, "..", "..", "paper_n2", "table_main.tex"), "w").write("\n".join(rows) + "\n")
t = J("sat_tuning.json")
if t:
    for design in ("ad", "tw"):
        for x in sorted([x for x in t if x["design"] == design], key=lambda x: x["dhw"]):
            x["retuned_best"] = max(x["fixed_retuned"]["eta"], x["fixed_design_fields"])
            lines.append(f"tuning {design} {1e3*x['dhw']:+.0f} meV: fixed(design) {100*x['fixed_design_fields']:.4f}% retuned "
                         f"{100*x['fixed_retuned']['eta']:.4f}% {x['fixed_retuned']['fields']} (best of grid and design pair {100*x['retuned_best']:.4f}%) uniform {100*x['uniform']:.4f}% "
                         f"programmed(fixed) {100*x['programmed_fixed']:.4f}% reprogrammed {100*x['reprogrammed']['eta']:.4f}% "
                         f"| reprogrammed/retuned {x['reprogrammed']['eta']/x['retuned_best']:.2f} "
                         f"| reprogrammed/design pattern {x['reprogrammed']['eta']/x['programmed_fixed']:.2f}")
    out["tuning"] = t
e = J("sat_electrodes.json")
if e:
    for x in e:
        lines.append(f"electrodes pitch {x['pitch']:.0f} gap {x['gap']:.0f} limit {x['limit']} levels {x['K']} ({x['n_levels']}): "
                     f"{100*x['eta']:.4f}% L {x['L']:.0f} (benchmark {100*x['benchmark']:.4f}%, x{x['eta']/x['benchmark']:.2f})"
)
    out["electrodes"] = e
cv = J("conv_n2.json")
if cv:
    for x in cv:
        lines.append(f"convergence {x['design']} {x['case']}: " + ", ".join(f"{k} {100*v:.4f}%" for k, v in x.items() if k not in ("design", "case")))
open(os.path.join(here, "numbers.log"), "w").write("\n".join(lines) + "\n")
json.dump(out, open(os.path.join(here, "numbers.json"), "w"), indent=1)
print("\n".join(lines))
