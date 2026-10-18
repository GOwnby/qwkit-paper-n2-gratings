"""Validation of LO-phonon rates, level-resolved saturation and segmented propagation."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwkit import scattering as sc, propagation as pr, sp

def test_form_integral_matches_brute_force():
    z = (np.arange(300) + 0.5) * 0.5
    phi = np.sin(np.pi * z / 150) * np.sin(2 * np.pi * z / 150) * (2 / 150)
    for Q in (5e7, 3e8, 2e9):
        zz = z * 1e-10
        brute = np.sum(np.outer(phi, phi) * 1e20 * np.exp(-Q * np.abs(zz[:, None] - zz[None, :]))) * (0.5e-10) ** 2
        assert abs(sc.form_integral(z, phi, Q) / brute - 1) < 1e-9

def test_form_integral_short_range_limit():
    # Q|z - z'| >> 1: I(Q) -> (2/Q) int phi^2 dz
    z = (np.arange(4000) + 0.5) * 0.5
    phi = np.exp(-((z - 1000) / 300) ** 2)
    Q = 2e9
    lim = 2 / Q * np.sum((phi * 1e10) ** 2) * 0.5e-10
    assert abs(sc.form_integral(z, phi, Q) / lim - 1) < 1e-2

def test_detailed_balance_and_stationary_equilibrium():
    r = sp.self_consistent([(64, "well"), (10, "barrier"), (29, "well")], 0.0, ns_cm2=1e11)
    E, psi, Z, rho = sp.prepare(r)
    R = sc.rate_matrix(r["z"], psi, E, T=300.0)
    b = np.exp(-(E - E[0]) / (sc.KB * 300.0)); b /= b.sum()
    flux = R.T @ b - b * R.sum(axis=1)           # net rate into each level for a Boltzmann distribution
    assert np.max(np.abs(flux)) < 1e-6 * R.max()
    assert 0.3e-12 < 1 / R[1, 0] < 3e-12          # ps-scale LO-phonon lifetime for a ~120 meV transition

def test_level_resolved_medium_limits():
    hw = 0.124
    st = dict(E=[0.0, hw, 2 * hw], Z=[[0, 15, 3], [15, 0, 20], [3, 20, 0]],
              Et=[[0, -hw, -2 * hw], [hw, 0, -hw], [2 * hw, hw, 0]], rho=[0.98, 0.02, 0.0], N=1e17)
    R = np.array([[0, 1e10, 1e8], [1e12, 0, 1e9], [5e11, 1e12, 0]])
    med = pr.Medium(st, hw, R=R)
    assert np.allclose(med.populations(0.0, 0.0), st["rho"], atol=1e-12)    # equilibrium is stationary
    rho = med.populations(1e11, 1e10)
    assert abs(rho.sum() - 1.0) < 1e-12 and rho[0] < 0.98                     # conserved, saturating

def test_segments_reproduce_sign_reversal():
    class Stub:
        def populations(self, I1, I2): return None
        def chi(self, rho): return 50e-9
        def alpha(self, rho): return 0.0, 0.0
    WG = dict(n1=3.2, n2=3.3, G1=0.4, G2=0.5, O_um=0.05, abg1_cm=0.0, abg2_cm=0.0, t_stack_um=1.5, w_um=15.0,
              Lcoh_um=20.0, lam1_um=10.0)
    a = pr.propagate(Stub(), WG, 1e-6, "sign", L_max_um=800.0)["P2"][-1]
    b = pr.propagate_segments([(Stub(), 20.0, 1.0), (Stub(), 20.0, -1.0)], WG, 1e-6, L_max_um=800.0)["P2"][-1]
    assert abs(a / b - 1) < 1e-6

def test_gaas_well_benchmark():
    # 10 nm GaAs / Al0.3Ga0.7As well (m = 0.07, offset 0.2138 eV, eps_s = 12.5, hbar w_LO = 36 meV, T = 0):
    # nextnano reproduction of Ferreira & Bastard gives E21 = 85.9 meV, tau21 = 0.834 ps (eps_inf not stated;
    # 10.89 used here). Agreement within 15% is required.
    from qwkit import core as q
    z = (np.arange(1400) + 0.5) * 0.5
    V = np.where((z > 300) & (z < 400), 0.0, 0.2138)
    E, psi = q.solve_states(z, V, np.full_like(z, 0.07), np.full_like(z, 1.42), nstates=3, nonparabolic=False)
    assert abs((E[1] - E[0]) * 1e3 - 85.9) < 0.3
    tau = 1 / sc.lo_rate(z, psi[:, 1], psi[:, 0], E[1] - E[0], T=0.0, hwlo=0.036, eps_inf=10.89, eps_s=12.5, m_eff=0.07)
    assert abs(tau / 0.834e-12 - 1) < 0.15
