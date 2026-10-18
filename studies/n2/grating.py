"""N2 (2): small-signal (undepleted, unsaturated) second-harmonic transfer of an electrode-programmed waveguide.
The waveguide of N1 (modes, confinement, overlap and background loss recomputed at each pump energy,
studies/nl02/hw_scan_wg.json) carries an array of electrodes of pitch p; electrode j applies a field F_j, so the
segment under it has the chi(2), ISB absorption at w and 2w and ISB index of the state set at F_j (chimap.json).
Within a segment of length l with constant coefficients,
    a1(s) = a1(0) exp(-alpha1 s / 2),
    a2(l) = exp(-alpha2 l / 2) [a2(0) - i K chi a1(0)^2 exp(i theta0) int_0^l exp((alpha2/2 - alpha1 + i dk) s) ds],
    theta(l) = theta0 + dk l,   dk = (4 pi / lambda1) [(n2 + dn2(F)) - (n1 + dn1(F))],
with dn_j(F) = G_j Re chi1_ISB(F) / (2 n_j) as in N1. Efficiency eta = |a2(L)|^2 / P1^2 (1/W).
Pattern synthesis by phase following: electrodes are assigned in order, each taking the bias (from an allowed set)
whose contribution has the largest projection on the field already generated (the first takes the largest |chi|);
this maximizes |a2| at the end for one pump energy in the lossless limit up to the electrode discretization."""
import sys, os, json
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..")); sys.path.insert(0, os.path.join(here, "..", "n1_paper"))
from qwkit import propagation as pr, sp

HW0 = 1.23984 / 10
GAM = 0.010
WGS = json.load(open(os.path.join(here, "..", "nl02", "hw_scan_wg.json")))
_DH = np.array(sorted(float(k) for k in WGS))


def wg_at(dhw):
    """Waveguide quantities at pump detuning dhw (eV), linearly interpolated between the computed detunings."""
    keys = sorted(WGS, key=float)
    out = {}
    for q in ("n1", "n2", "G1", "G2", "O_um", "abg1_cm", "abg2_cm", "lam1_um"):
        out[q] = float(np.interp(dhw, _DH, [WGS[k][q] for k in keys]))
    out["lam1_um"] = 1.23984 / (HW0 + dhw)
    out["t_stack_um"] = 1.512; out["w_um"] = 15.0
    return out


def chi1_re(st, hw):
    E = np.array(st["Et"]); Z = np.array(st["Z"]); rho = np.array(st["rho"]); N = st["N"] * 1e6
    s = 0j
    for l in range(len(rho)):
        for m in range(l + 1, len(rho)):
            Em = E[m, l]
            s += (rho[l] - rho[m]) * (Z[l, m] * 1e-10) ** 2 * (1 / (Em - hw - 1j * GAM) + 1 / (Em + hw + 1j * GAM)) / pr.QE
    return (N * pr.QE ** 2 / pr.EPS0 * s).real


class Segments:
    """Per-bias segment coefficients at one pump detuning: chi (m/V), alpha1, alpha2 (1/m, modal), dk (1/m), K."""

    def __init__(self, states, dhw):
        hw = HW0 + dhw
        w = wg_at(dhw)
        lam1 = w["lam1_um"] * 1e-6
        O = w["O_um"] * 1e6 / (w["w_um"] * 1e-6)
        self.K = (np.pi / lam1) * np.sqrt(2 * O / (w["n1"] ** 2 * w["n2"] * pr.C0 * pr.EPS0))
        self.F = []; self.chi = []; self.a1 = []; self.a2 = []; self.dk = []
        for s in states:
            st = s["st"]
            med = pr.Medium(st, hw, gamma=GAM, tau_ps=1.0)
            rho = np.array(st["rho"])
            al1, al2 = med.alpha(rho)
            dn1 = w["G1"] * chi1_re(st, hw) / (2 * w["n1"]); dn2 = w["G2"] * chi1_re(st, 2 * hw) / (2 * w["n2"])
            self.F.append(s["F"]); self.chi.append(med.chi(rho))
            self.a1.append(w["G1"] * al1 + w["abg1_cm"] * 100); self.a2.append(w["G2"] * al2 + w["abg2_cm"] * 100)
            self.dk.append(4 * np.pi / lam1 * ((w["n2"] + dn2) - (w["n1"] + dn1)))
        for q in ("F", "chi", "a1", "a2", "dk"):
            setattr(self, q, np.array(getattr(self, q)))
        self.lam1 = lam1

    def contrib(self, i, l, a1sq, theta):
        """SH amplitude added by segment i of length l (m) at its end, for pump a1^2 at its start and phase theta."""
        g = self.a2[i] / 2 - self.a1[i] + 1j * self.dk[i]
        I = l if abs(g) * l < 1e-9 else (np.exp(g * l) - 1) / g
        return -1j * self.K * self.chi[i] * a1sq * np.exp(1j * theta) * np.exp(-self.a2[i] * l / 2) * I


def run(seg, pattern, p_um):
    """Propagate an electrode pattern (indices into seg) of pitch p_um (a number, or one length per element);
    returns eta (1/W) at the end and along x."""
    ls = np.broadcast_to(np.asarray(p_um, float) * 1e-6, (len(pattern),))
    a1sq = 1.0 + 0j; a2 = 0j; theta = 0.0; eta = []
    for i, l in zip(pattern, ls):
        a2 = a2 * np.exp(-seg.a2[i] * l / 2) + seg.contrib(i, l, a1sq, theta)
        a1sq *= np.exp(-seg.a1[i] * l); theta += seg.dk[i] * l
        eta.append(abs(a2) ** 2)
    return eta[-1], np.array(eta)


def synthesize(seg, n_el, p_um, allowed):
    """Phase-following pattern of n_el electrodes from the allowed bias indices; returns the index list."""
    l = p_um * 1e-6
    a1sq = 1.0 + 0j; a2 = 0j; theta = 0.0; pat = []
    for j in range(n_el):
        best = None
        for i in allowed:
            c = seg.contrib(i, l, a1sq, theta)
            new = a2 * np.exp(-seg.a2[i] * l / 2) + c
            score = abs(new)
            if best is None or score > best[0]:
                best = (score, i, new)
        _, i, a2 = best
        pat.append(i); a1sq *= np.exp(-seg.a1[i] * l); theta += seg.dk[i] * l
    return pat


def refine(seg, pattern, p_um, allowed, sweeps=3):
    """Coordinate ascent on the largest efficiency along x: each electrode in turn takes the allowed bias that
    maximizes max_x eta (the pattern is truncated at the optimum length after each sweep). Returns (pattern, eta)."""
    pat = list(pattern)
    _, c = run(seg, pat, p_um); n = int(np.argmax(c)) + 1; pat = pat[:n]; best = c[n - 1]
    for _ in range(sweeps):
        improved = False
        for j in range(len(pat)):
            keep = pat[j]
            for i in allowed:
                if i == keep:
                    continue
                pat[j] = i
                _, c = run(seg, pat, p_um)
                if c.max() > best * (1 + 1e-9):
                    best = c.max(); keep = i; improved = True
            pat[j] = keep
        _, c = run(seg, pat, p_um); n = int(np.argmax(c)) + 1; pat = pat[:n]; best = c[n - 1]
        if not improved:
            break
    return pat, float(best)
