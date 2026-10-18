"""N2 (3): tuning by reprogramming. For pump detunings -50 ... +15 meV from the 10 um design, the electrode pattern is
re-synthesized (phase following, grating.synthesize) on electrode arrays of pitch 1, 2, 4 and 8 um, with either all
valid biases of the medium or a binary pair (+F*, -F*); the device length is the optimum (largest small-signal
efficiency along x). Compared with: the fixed N1 bipolar grating (+-F* alternating, segment lengths of the design
detuning 0) evaluated at each detuning, and the uniform best single bias (no grating). Media: ADQW (64/10/29 A,
F = -140 ... 140 kV/cm) and TW (sign reversal, F = -95 ... 95 kV/cm), ns = 5e11 cm^-2. Writes tuning.json."""
import sys, os, json
import numpy as np
from multiprocessing import Pool
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import grating as g

S = json.load(open(os.path.join(here, "chimap.json")))
DHW = [round(x, 3) for x in np.arange(-0.050, 0.0151, 0.005)]
PITCH = (2.0, 4.0, 8.0)
LMAX = 1500.0                                           # um


def states(design):
    return [s for s in S if s["design"] == design and s["valid"]]


def best_eta(seg, pat, p):
    _, c = g.run(seg, pat, p)
    i = int(np.argmax(c))
    ls = np.broadcast_to(np.asarray(p, float), (len(pat),))
    return float(c[i]), float(np.cumsum(ls)[i])


def job(args):
    design, dhw = args
    st = states(design)
    seg = g.Segments(st, dhw)
    F = list(seg.F)
    out = dict(design=design, dhw=dhw)
    # fixed N1-type bipolar grating designed at dhw = 0: +-F* with each segment its own coherence length at 0
    seg0 = g.Segments(st, 0.0)
    Fs = 80 if design == "ad" else 55
    ip, im = F.index(Fs), F.index(-Fs)
    L0 = [np.pi / seg0.dk[ip] * 1e6, np.pi / seg0.dk[im] * 1e6]
    n = int(LMAX / np.mean(L0))
    pat = [ip if k % 2 == 0 else im for k in range(n)]
    out["fixed"] = best_eta(seg, pat, [L0[k % 2] for k in range(n)])
    # uniform single bias (no grating), best over biases
    out["uniform"] = max(best_eta(seg, [i] * int(LMAX / 1.0), 1.0) for i in range(len(F)))
    out["prog"] = {}
    for p in PITCH:
        n = int(LMAX / p)
        full, ef = g.refine(seg, g.synthesize(seg, n, p, range(len(F))), p, range(len(F)))
        binp, eb = g.refine(seg, g.synthesize(seg, n, p, [ip, im]), p, [ip, im])
        out["prog"][str(p)] = dict(all=(ef, len(full) * p), binary=(eb, len(binp) * p),
                                   biases_used=sorted(set(int(F[i]) for i in full)), pattern=[int(F[i]) for i in full])
    print(f"{design} dhw {1000 * dhw:+.0f} meV: fixed {out['fixed'][0]:.2e} uniform {out['uniform'][0]:.2e} | " +
          " ".join(f"p{p:.0f}: all {out['prog'][str(p)]['all'][0]:.2e} bin {out['prog'][str(p)]['binary'][0]:.2e}" for p in PITCH),
          flush=True)
    return out


if __name__ == "__main__":
    jobs = [(d, x) for d in ("ad", "tw") for x in DHW]
    with Pool(int(os.environ.get("NPROC", "2"))) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(here, "tuning.json"), "w"), indent=1)
