"""Additional analytic validations: finite square well with BenDaniel-Duke boundary conditions, and the
phase-mismatched (sinc^2) limit of the coupled-mode equations."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.optimize import brentq
from qwkit import core, propagation as pr

HB2M0 = core.HB2M0          # hbar^2 / 2 m0 in eV A^2


def finite_well_levels(L, V0, mw, mb, nmax=3):
    """Bound levels of a finite well with BenDaniel-Duke matching (psi and psi'/m continuous)."""
    def f_even(E):
        k = np.sqrt(E * mw / HB2M0); q = np.sqrt((V0 - E) * mb / HB2M0)
        return (k / mw) * np.sin(k * L / 2) - (q / mb) * np.cos(k * L / 2)
    def f_odd(E):
        k = np.sqrt(E * mw / HB2M0); q = np.sqrt((V0 - E) * mb / HB2M0)
        return (k / mw) * np.cos(k * L / 2) + (q / mb) * np.sin(k * L / 2)
    Es = np.linspace(1e-6, V0 - 1e-6, 20000); roots = []
    for f in (f_even, f_odd):
        v = f(Es)
        for i in np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
            roots.append(brentq(f, Es[i], Es[i + 1]))
    return np.sort(roots)[:nmax]


def test_finite_well_benDaniel_Duke():
    L, V0, mw, mb = 100.0, 0.52, 0.043, 0.075
    Ea = finite_well_levels(L, V0, mw, mb)
    err = []
    for dz in (0.5, 0.25):
        pad = 300.0
        n = int(round((L + 2 * pad) / dz)); z = (np.arange(n) + 0.5) * dz
        inside = (z > pad) & (z < pad + L)
        V = np.where(inside, 0.0, V0); m = np.where(inside, mw, mb)
        E, _ = core.solve_states(z, V, m, np.full_like(z, 1.0), nstates=3, nonparabolic=False)
        err.append(np.max(np.abs(E - Ea)))
    assert err[0] < 1.0e-4                             # within 0.1 meV at the default 0.5 A grid
    assert 3.5 < err[0] / err[1] < 4.5                 # second-order convergence


def test_phase_mismatched_sinc_limit():
    class Stub:
        def populations(self, I1, I2): return None
        def chi(self, rho): return 50e-9
        def alpha(self, rho): return 0.0, 0.0
    WG = dict(n1=3.2, n2=3.3, G1=0.4, G2=0.5, O_um=0.05, abg1_cm=0.0, abg2_cm=0.0, t_stack_um=1.5, w_um=15.0,
              Lcoh_um=20.0, lam1_um=10.0)
    P1 = 1e-6
    r = pr.propagate(Stub(), WG, P1, "none", L_max_um=130.0, steps_per_lcoh=256)
    K, chi, dk = r["K"], 50e-9, np.pi / 20e-6
    x = r["x_um"] * 1e-6
    analytic = (K * chi * P1) ** 2 * (np.sin(dk * x / 2) / (dk / 2)) ** 2
    assert np.max(np.abs(r["P2"] - analytic)) / analytic.max() < 1e-6
