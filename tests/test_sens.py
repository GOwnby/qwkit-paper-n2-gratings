"""qwkit.sens: implicit-differentiation tangents agree with finite differences of the full self-consistent model."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from qwkit import sens


def test_tangents_match_finite_differences():
    m = sens.Model([(64, "well"), (10, "barrier"), (29, "well")], F=10.0, ns=5e11, nstates=4)
    s = m.forward(m.theta0)
    tg = m.tangents(s)
    for name, h in [("t0", 0.02), ("F", 0.2), ("dVb", 2e-4), ("mw", 2e-5), ("epsw", 3e-3)]:
        j = m.names.index(name)
        tp = m.theta0.copy(); tp[j] += h; tm = m.theta0.copy(); tm[j] -= h
        a = m.forward(tp, VH0=s["VH"]); b = m.forward(tm, VH0=s["VH"])
        fd = (a["E"][:3] - b["E"][:3]) / (2 * h)
        an = tg["dE"][:3, j]
        assert np.allclose(an, fd, rtol=2e-4, atol=1e-6 * np.max(np.abs(fd))), (name, an, fd)


def _fd_check(m, th, names, steps, obs_keys, ladder, rtol):
    s = m.forward(th)
    tg = m.tangents(s)
    nk = 4
    base, J = sens.observable_jacobian(m, s, tg, 0.124, ladder=ladder, nkeep=nk)
    for name in names:
        j = m.names.index(name); h = steps[name]
        tp = th.copy(); tp[j] += h; tm = th.copy(); tm[j] -= h
        a = m.forward(tp, VH0=s["VH"]); b = m.forward(tm, VH0=s["VH"])
        # states and occupations
        fd_rho = (a["rho"] - b["rho"]) / (2 * h)
        assert np.allclose(tg["drho"][:3, j], fd_rho[:3], rtol=1e-3, atol=1e-4 * np.max(np.abs(fd_rho[:3])) + 1e-30), (name, "rho")
        sg = np.sign(np.sum(a["psi"] * s["psi"], axis=0)); a_psi = a["psi"] * sg
        sg = np.sign(np.sum(b["psi"] * s["psi"], axis=0)); b_psi = b["psi"] * sg
        fd_psi = (a_psi - b_psi)[:, :3] / (2 * h)
        assert np.max(np.abs(tg["dpsi"][:, :3, j] - fd_psi)) <= 1e-3 * np.max(np.abs(fd_psi)) + 1e-30, (name, "psi")
        oa = sens.observables(m, tp, a["E"], a["psi"], a["rho"], 0.124, ladder=ladder, nkeep=nk)
        ob = sens.observables(m, tm, b["E"], b["psi"], b["rho"], 0.124, ladder=ladder, nkeep=nk)
        for k in obs_keys:
            fd = (oa[k] - ob[k]) / (2 * h)
            assert abs(J[k][j] - fd) <= rtol * max(abs(fd), abs(J[k][j])) + 1e-9 * abs(base[k]), (name, k, J[k][j], fd)


def test_observable_jacobian_all_parameter_types():
    m = sens.Model([(64, "well"), (10, "barrier"), (29, "well")], F=10.0, ns=5e11, nstates=4)
    steps = dict(t1=0.02, ns=2e9, mb=4e-5, Egw=3e-4, Egb=5e-4, epsb=3e-3)
    _fd_check(m, m.theta0.copy(), list(steps), steps, ["Et1m", "Et1n", "z1m", "chi_abs", "a1", "a2", "M_wg"], (0, 1, 2), 2e-3)


def test_clipped_mass_cells():
    """A small barrier gap makes the energy-dependent mass hit its 0.2 m_edge floor in the barriers."""
    m = sens.Model([(64, "well"), (10, "barrier"), (29, "well")], F=0.0, ns=5e11, nstates=4)
    th = m.theta0.copy(); th[m.names.index("Egb")] = 0.5
    s = m.forward(th)
    P = s["P"]; mz = P["m"] * (1 + (s["E"][0] - s["V"]) / P["Eg"])
    assert np.any(mz < 0.2 * P["m"])
    steps = dict(mb=4e-5, t0=0.005)
    _fd_check(m, th, list(steps), steps, ["Et1m", "chi_abs"], (0, 1, 2), 2e-3)


def test_matrix_free_tangents_match_dense():
    """The matrix-free path (tridiagonal reduced resolvents + GMRES) reproduces the dense implicit derivatives."""
    lay = [(64, "well"), (10, "barrier"), (29, "well")]
    md = sens.Model(lay, F=10.0, ns=5e11, nstates=4, dense=True)
    mk = sens.Model(lay, F=10.0, ns=5e11, nstates=4)
    s = md.forward(md.theta0)
    a, b = md.tangents(s), mk.tangents(s)
    for k in ("dE", "dpsi", "drho", "dV"):
        assert np.max(np.abs(a[k] - b[k])) <= 1e-8 * np.max(np.abs(a[k])), k
