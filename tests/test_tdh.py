"""Validation of the time-dependent Hartree (RPA) module."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwkit import sp, core, tdh

def _adqw(ns=5e11):
    r = sp.self_consistent([(64, "well"), (10, "barrier"), (29, "well")], 0.0, ns_cm2=ns)
    E, psi, Z, rho = sp.prepare(r)
    act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
    return r["z"], psi, E, Z, rho, ns, N, zc

def test_bare_limit_equals_density_matrix_chi2():
    z, psi, E, Z, rho, ns, N, zc = _adqw()
    m = tdh.TDH(z, psi, E, rho, ns, N, gamma=0.010, kernel=False, zc=zc)
    for hw in (0.110, 0.124, 0.130):
        a, b = m.chi2(hw), core.chi2_general(E, Z, rho, hw, 0.010, N)
        assert abs(abs(a) / abs(b) - 1) < 0.01          # differs only by population decay of 2w terms
        assert abs(a + b) < 0.02 * abs(b)                # opposite sign convention (electron charge)

def test_linear_peak_at_ando_energy():
    z, psi, E, Z, rho, ns, N, zc = _adqw()
    m = tdh.TDH(z, psi, E, rho, ns, N, gamma=0.002, zc=zc)
    hws = np.linspace(0.11, 0.14, 1501)
    peak = hws[np.argmax([np.imag(m.chi1(h)) for h in hws])]
    Et = sp.depol_energies(z, psi, E, rho, ns, eps_r=13.5)
    assert abs(peak - Et[1, 0]) < 3e-4

def test_real_time_matches_second_order():
    z, psi, E, Z, rho, ns, N, zc = _adqw()
    m = tdh.TDH(z, psi, E, rho, ns, N, gamma=0.010, zc=zc, tau_ps=0.2)
    hw = 0.118; c1, c2, _, _ = m.response(hw)
    Pk, _ = m.evolve(3e3, hw)
    assert abs(Pk[1] / (tdh.EPS0 * 3e3) / c1 - 1) < 1e-3
    assert abs(Pk[2] / (tdh.EPS0 * 9e6) * 1e12 / c2 - 1) < 2e-3
