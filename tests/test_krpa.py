import numpy as np
from qwkit import core, tdh, krpa, interface


def _states():
    L = [(60.0, "well"), (15.0, "barrier"), (30.0, "well")]
    z, V0, m, Eg, act = core.build(L, pad=100.0, dz=0.5)
    V = core.potential(z, V0, act, 20.0)
    E, psi = core.solve_states(z, V, m, Eg, nstates=3)
    psi = core.orthonormalize(z, psi)
    return z, E, psi, 0.5 * (act[0] + act[1])


def test_single_shell_reduces_to_tdh():
    z, E, psi, zc = _states()
    rho0 = np.array([0.8, 0.15, 0.05]); ns = 5e11; L = 105.0
    for kern in (False, True):
        t = tdh.TDH(z, psi, E, rho0, ns, ns / (L * 1e-8), gamma=0.008, eps_r=13.0, tau_ps=1.0, kernel=kern, zc=zc)
        k = krpa.KRPA(z, psi, E[:, None], rho0[:, None], np.array([ns * 1e4]), gamma=0.008, eps_b=13.0, tau_ps=1.0,
                      kernel=kern, zc=zc)
        for hw in (0.08, 0.11, 0.13):
            c1t, c2t = t.chi1(hw), t.chi2(hw)
            assert abs(k.chi1(hw, L) - c1t) < 1e-9 * abs(c1t) + 1e-14
            assert abs(k.chi2(hw, L) - c2t) < 1e-8 * abs(c2t)


def test_froehlich_weights_sum():
    modes = [(0.0167, 0.0191), (0.0303, 0.0522), (0.0779, 0.0896)]
    einf = 4.0
    C = krpa.froehlich_weights(einf, modes)
    eps0 = einf * np.prod([(l / t) ** 2 for t, l in modes])
    assert abs(C.sum() - (1 / einf - 1 / eps0)) < 1e-12
    assert np.all(C > 0)


def test_shell_occupations_match_subband_densities():
    r = interface.solve(3e13, T=50.0, NA=1e19, E_pin=1.0, nstates=6)
    k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=600, gamma=0.02,
                                eps_b=4.0)
    assert abs(k.ns * 1e-4 / 3e13 - 1) < 2e-3


def test_multimode_lo_rate_single_mode_limit():
    from qwkit import scattering as sc
    z, E, psi, zc = _states()
    dE = E[1] - E[0]
    a = sc.lo_rate(z, psi[:, 1], psi[:, 0], dE, T=300.0)
    b = sc.lo_rate_modes(z, psi[:, 1], psi[:, 0], dE, [(0.0337, 1 / 11.6 - 1 / 13.9)], T=300.0)
    assert abs(a / b - 1) < 1e-12
    c = sc.lo_rate(z, psi[:, 1], psi[:, 0], dE, T=300.0, q_s=1e9)
    assert c < a


def test_inactive_pairs_match_tdh():
    L = [(60.0, "well"), (15.0, "barrier"), (30.0, "well")]
    z, V0, m, Eg, act = core.build(L, pad=100.0, dz=0.5)
    V = core.potential(z, V0, act, 20.0)
    E, psi = core.solve_states(z, V, m, Eg, nstates=4)
    psi = core.orthonormalize(z, psi); zc = 0.5 * (act[0] + act[1])
    rho0 = np.array([0.9, 0.1, 0.0, 0.0]); ns = 8e11; Lp = 105.0
    t = tdh.TDH(z, psi, E, rho0, ns, ns / (Lp * 1e-8), gamma=0.008, eps_r=13.0, tau_ps=1.0, kernel=True, zc=zc)
    k = krpa.KRPA(z, psi, E[:, None], rho0[:, None], np.array([ns * 1e4]), gamma=0.008, eps_b=13.0, tau_ps=1.0, zc=zc)
    assert len(k.off) < 12
    for hw in (0.07, 0.1, 0.12):
        assert abs(k.chi1(hw, Lp) - t.chi1(hw)) < 1e-9 * abs(t.chi1(hw))
        assert abs(k.chi2(hw, Lp) - t.chi2(hw)) < 1e-8 * abs(t.chi2(hw))


def test_lo_rate_k_edge_limit():
    from qwkit import scattering as sc
    z, E, psi, zc = _states()
    dE = E[1] - E[0]
    modes = [(0.0337, 1 / 11.6 - 1 / 13.9)]
    a = sc.lo_rate_modes(z, psi[:, 1], psi[:, 0], dE, modes, T=300.0, m_eff=0.043)
    b = sc.lo_rate_k(z, psi[:, 1], psi[:, 0], 0.0, E[1], E[0], 0.043, 1e9, modes, T=300.0)
    assert abs(a / b - 1) < 1e-6
    # Pauli blocking reduces the rate; screening factor <= 1 for a single occupied subband
    c = sc.lo_rate_k(z, psi[:, 1], psi[:, 0], 0.0, E[1], E[0], 0.043, 1e9, modes, T=300.0, mu=E[0] + 0.08)
    assert c < b
    S = sc.screening_factor(z, psi, 1, 0, [(0, 0.043, 3e8)], [2e8, 5e8, 1e9], 11.6)
    assert np.all(S < 1.0) and np.all(S > 0.0)
