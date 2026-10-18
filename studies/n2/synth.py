"""N2 pattern synthesis for saturated, pump-depleted propagation on an electrode array of pitch p.

All searches respect the lateral-breakdown limit between neighbouring electrodes, |F_j - F_j+1| <= dfm (kV/cm).
greedy(): electrodes are assigned in order; each takes the bias that maximizes |a2|^2 + mu |a1|^2 at its end
(mu > 0 values the pump that is left for later electrodes; mu = 0 is pure phase following). The pattern is
truncated where P2 is largest.
refine(): coordinate ascent on the largest P2 along x with abrupt (ideal) fields; the states at the electrode
boundaries of the current pattern are cached, so changing electrode j re-propagates only from j on. The pattern may
grow by `grow` electrodes at its end.
refine_realistic(): the same with the realistic (fringing, three-height) fields, by full propagation.
finalize(): truncates a pattern one electrode beyond the optimum length and re-evaluates it with realistic fields.
Every pattern found is a lower bound on what programming can reach."""
import numpy as np
from qwkit import propagation as pr
import common_n2 as c

MUS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03)


def _ok(bank, i, nb, dfm):
    return all(abs(bank.F[i] - bank.F[j]) <= dfm + 1e-9 for j in nb if j is not None)


def _adv(bank, i, n, dx, a, th, P):
    r = pr.propagate_path(bank.media, bank.dk, [i] * n, dx, bank.wg, P, a0=a, theta0=th, record=False)
    return r["a"], r["theta"]


def greedy(bank, P, pitch, dx, allowed, mu=0.0, Lmax=1500.0, patience=6.0, dfm=np.inf):
    n = int(round(pitch / dx))
    a = np.array([np.sqrt(P) + 0j, 0j]); th = 0.0
    pat = []; best = (0.0, 0); x = 0.0
    while x < Lmax:
        cand = None
        prev = pat[-1] if pat else None
        for i in allowed:
            if not _ok(bank, i, [prev], dfm):
                continue
            a2, th2 = _adv(bank, i, n, dx, a, th, P)
            sc = abs(a2[1]) ** 2 + mu * abs(a2[0]) ** 2
            if cand is None or sc > cand[0]:
                cand = (sc, i, a2, th2)
        _, i, a, th = cand
        pat.append(int(i)); x += pitch
        P2 = abs(a[1]) ** 2
        if P2 > best[0]:
            best = (P2, len(pat))
        if x - best[1] * pitch > patience * np.mean(bank.Lcoh) and P2 < 0.8 * best[0]:
            break
    return pat[:max(best[1], 1)], best[0] / P


def greedy_scan(bank, P, pitch, dx, allowed, Lmax, dfm=np.inf):
    res = [(greedy(bank, P, pitch, dx, allowed, mu=mu, Lmax=Lmax, dfm=dfm), mu) for mu in MUS]
    (pat, eta), mu = max(res, key=lambda t: t[0][1])
    return pat, eta, mu, [(m, e) for (_, e), m in res]


def _forward(bank, pat, n, dx, P):
    a = np.array([np.sqrt(P) + 0j, 0j]); th = 0.0
    states = [(a, th)]; pmax = [0.0]
    for i in pat:
        r = pr.propagate_path(bank.media, bank.dk, [i] * n, dx, bank.wg, P, a0=a, theta0=th, record=True)
        a, th = r["a"], r["theta"]
        states.append((a, th)); pmax.append(max(pmax[-1], float(r["P2"].max())))
    return states, pmax


def _tail(bank, pat, j0, n, dx, P, a, th):
    idx = [i for i in pat[j0:] for _ in range(n)]
    r = pr.propagate_path(bank.media, bank.dk, idx, dx, bank.wg, P, a0=a, theta0=th, record=False)
    return r["eta"] * P            # max P2 over the tail


def _neighbours(pat, j):
    return [pat[j - 1] if j > 0 else None, pat[j + 1] if j + 1 < len(pat) else None]


def refine(bank, pat, P, pitch, dx, allowed, sweeps=2, grow=4, dfm=np.inf):
    n = int(round(pitch / dx))
    pat = list(pat) + [pat[-1]] * grow if pat else [allowed[0]] * grow
    states, pmax = _forward(bank, pat, n, dx, P)
    best = pmax[-1]
    for _ in range(sweeps):
        improved = False
        for j in range(len(pat)):
            a, th = states[j]
            keep = pat[j]
            for i in allowed:
                if i == keep or not _ok(bank, i, _neighbours(pat, j), dfm):
                    continue
                pat[j] = i
                v = max(pmax[j], _tail(bank, pat, j, n, dx, P, a, th))
                if v > best * (1 + 1e-7):
                    best = v; keep = i; improved = True
            pat[j] = keep
            if improved:
                states, pmax = _forward(bank, pat, n, dx, P)
        if not improved:
            break
    states, pmax = _forward(bank, pat, n, dx, P)
    k = int(np.argmax(np.array(pmax) >= pmax[-1] * (1 - 1e-12)))
    return pat[:max(k, 1)], pmax[-1] / P


def eval_realistic(bank, pat, P, pitch, dx, gap_um=c.GAP_UM, record=False):
    idx = c.realistic_steps(bank, [bank.F[i] for i in pat], pitch, dx, gap_um=gap_um)
    return c.propagate(bank, idx, dx, P, record=record)


def refine_realistic(bank, pat, P, pitch, dx, allowed, sweeps=1, gap_um=c.GAP_UM, dfm=np.inf):
    pat = list(pat)
    best = eval_realistic(bank, pat, P, pitch, dx, gap_um)["eta"]
    for _ in range(sweeps):
        improved = False
        for j in range(len(pat)):
            keep = pat[j]
            for i in allowed:
                if i == keep or not _ok(bank, i, _neighbours(pat, j), dfm):
                    continue
                pat[j] = i
                v = eval_realistic(bank, pat, P, pitch, dx, gap_um)["eta"]
                if v > best * (1 + 1e-7):
                    best = v; keep = i; improved = True
            pat[j] = keep
        if not improved:
            break
    return pat, best


def finalize(bank, pat, P, pitch, dx, gap_um=c.GAP_UM):
    """Keep the electrodes up to one beyond the optimum length; return (pattern, eta, L) with realistic fields."""
    r = eval_realistic(bank, pat, P, pitch, dx, gap_um)
    k = min(len(pat), int(np.ceil(r["L_opt_um"] / pitch)) + 1)
    r2 = eval_realistic(bank, pat[:k], P, pitch, dx, gap_um)
    return pat[:k], r2["eta"], r2["L_opt_um"]


def respect(bank, pat, allowed, dfm):
    """Make a pattern obey the neighbour limit by moving offending electrodes to the nearest allowed bias."""
    pat = list(pat)
    for j in range(1, len(pat)):
        if abs(bank.F[pat[j]] - bank.F[pat[j - 1]]) > dfm + 1e-9:
            ok = [i for i in allowed if abs(bank.F[i] - bank.F[pat[j - 1]]) <= dfm + 1e-9]
            pat[j] = min(ok, key=lambda i: abs(bank.F[i] - bank.F[pat[j]]))
    return pat
