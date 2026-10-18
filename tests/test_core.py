"""Validation of the core Schroedinger solver and chi(2) machinery against analytic limits."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwkit import core as q

def _with_barrier(**kw):
    old = dict(q.MAT["barrier"]); q.MAT["barrier"].update(kw); return old

def test_infinite_well_limit():
    old = _with_barrier(V=200.0, m=0.043, Eg=0.75)
    try:
        z, V0, m, Eg, act = q.build([(100, "well")], pad=20, dz=0.25)
        E, psi = q.solve_states(z, V0, m, Eg, nstates=3, nonparabolic=False)
        Leff = 100 + 2 * np.sqrt(q.HB2M0 / 0.043 / 200.0)   # first-order penetration correction
        Eth = q.HB2M0 / 0.043 * (np.arange(1, 4) * np.pi / Leff) ** 2
        assert np.allclose(E, Eth, rtol=0.01), (E, Eth)
        Z = q.dipoles(z, psi)
        assert abs(abs(Z[0, 1]) - 16 * Leff / (9 * np.pi ** 2)) / abs(Z[0, 1]) < 0.02
    finally:
        q.MAT["barrier"].update(old)

def test_chi_zero_and_odd_for_symmetric():
    hw = 1.23984 / 10
    lay = [(56, "well"), (6, "barrier"), (56, "well")]
    a = q.analyze_T(lay, 60.0, hw)["chiT"][300.0]; b = q.analyze_T(lay, -60.0, hw)["chiT"][300.0]
    c = q.analyze_T(lay, 0.0, hw)["chiT"][300.0]
    assert abs(a + b) < 1e-8 * abs(a) and abs(c) < 1e-8 * abs(a)

def test_translation_invariance():
    hw = 1.23984 / 10
    r = q.analyze_T([(64, "well"), (10, "barrier"), (30, "well")], 0.0, hw)
    rho = r["rhoT"][300.0]
    a = q.chi2_general(r["E"], r["Z"], rho, hw, 0.01, r["N_cm3"])
    b = q.chi2_general(r["E"], r["Z"] + 41.0 * np.eye(len(r["E"])), rho, hw, 0.01, r["N_cm3"])
    assert abs(a - b) < 1e-9 * abs(a)

def test_three_level_limit_matches_book_eq_15_28():
    # With only level 1 populated, the general formula's dominant term is Eq. 15.28.
    E = np.array([0.0, 0.124, 0.248]); Z = np.array([[0, 20.0, 10.0], [20.0, 0, 25.0], [10.0, 25.0, 0]])
    rho = np.array([1.0, 0, 0]); hw = 0.124; G = 0.01; N = 1e18
    full = q.chi2_general(E, Z, rho, hw, G, N)
    book = N * 1e6 * q.E_CH ** 3 / q.EPS0 * (Z[0, 2] * Z[0, 1] * Z[1, 2] * 1e-30) / (
        (E[2] - 2 * hw - 1j * G) * (E[1] - hw - 1j * G) * q.E_CH ** 2) * 1e12
    assert abs(full - book) / abs(book) < 0.05, (full, book)

def test_nonparabolic_states_have_parity():
    z, V0, m, Eg, act = q.build([(50, "well"), (20, "barrier"), (50, "well")])
    E, psi = q.solve_states(z, V0, m, Eg, nstates=4)
    dz = z[1] - z[0]
    par = [np.sum(psi[:, k] * psi[::-1, k]) * dz for k in range(4)]
    assert np.allclose(np.abs(par), 1.0, atol=1e-6) and np.all(np.diff(E) > 0)


def test_mirror_dipoles_nonparabolic():
    """Mirror-image asymmetric wells must give identical |z_ij| (nonparabolic states, any origin)."""
    out = []
    for L in ([(62, "well"), (20, "barrier"), (29, "well")], [(29, "well"), (20, "barrier"), (62, "well")]):
        z, V0, m, Eg, act = q.build(L, pad=100, dz=0.5)
        E, psi = q.solve_states(z, V0, m, Eg, nstates=3)
        out.append(np.abs(q.dipoles(z, psi)))
        out.append(np.abs(q.dipoles(z + 1000.0, psi)))            # origin shift
    for a in out[1:]:
        d = np.abs(a - out[0]); d[np.diag_indices(3)] = 0
        assert np.max(d) < 1e-6, np.max(d)


def test_kane_two_band_orthogonality():
    """The states of the energy-dependent-mass Hamiltonian are orthogonal as two-component Kane states, and the
    two-band dipoles are close to the Loewdin ones."""
    import numpy as np
    from qwkit import core as q
    z, V0, m, Eg, act = q.build([(70.0, "well"), (12.0, "barrier"), (32.0, "well")], pad=100.0, dz=0.5)
    E, psi = q.solve_states(z, V0, m, Eg, nstates=5)
    dz = z[1] - z[0]
    psi = psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz)
    S1 = psi.T @ psi * dz
    assert np.max(np.abs(S1 - np.diag(np.diag(S1)))) > 0.02          # one-band overlaps are not small
    zc = 0.5 * (act[0] + act[1])
    Z2, ov = q.kane_dipoles(z, psi, E, V0, m, Eg, zc)
    assert ov < 2e-3
    ZL = q.dipoles(z - zc, q.orthonormalize(z, psi))
    assert abs(Z2[0, 1] / ZL[0, 1] - 1) < 0.02


def test_buffered_loewdin_box_independent():
    """Loewdin dipoles of the localized states (sp.prepare) agree with the two-band Kane dipoles and do not depend on the
    box: double well 64/10/29 with three localized states in a 200 A box (z13 was 7% too large when only the localized
    states were orthonormalized)."""
    import numpy as np
    from qwkit import sp, core as q
    lay = [(64, "well"), (10, "barrier"), (29, "well")]
    z13 = {}
    for spacer in (100.0, 200.0):
        r = sp.self_consistent(lay, 0.0, ns_cm2=5e11, T=300.0, spacer=spacer, dz=0.5, doping="spacer", nstates=6)
        E, psi, Z, rho = sp.prepare(r, nkeep=3)
        z, act = r["z"], r["active"]; zc = 0.5 * (act[0] + act[1])
        keep = np.where(q.localized(z, r["psi"], act))[0][:3]
        ZK, _ = q.kane_dipoles(z, r["psi"][:, keep], r["E"][keep], r["V"], r["m"], r["Eg"], zc)
        assert abs(Z[0, 2] / ZK[0, 2] - 1) < 0.012 and abs(Z[1, 2] / ZK[1, 2] - 1) < 0.006
        z13[spacer] = Z[0, 2]
    assert abs(z13[200.0] / z13[100.0] - 1) < 0.01


def test_kane_two_band_orthogonality_exact():
    """The exact discrete light-particle partner makes the two-component states orthogonal to the eigenvalue accuracy."""
    import numpy as np
    from qwkit import core as q
    z, V0, m, Eg, act = q.build([(64, "well"), (10, "barrier"), (29, "well")], pad=100.0, dz=0.5)
    V = q.potential(z, V0, act, 0.0)
    E, psi = q.solve_states(z, V, m, Eg, nstates=6, tol=1e-12, iters=80)
    dz = z[1] - z[0]; psi = psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz)
    Z, ov = q.kane_dipoles(z, psi, E, V, m, Eg)
    S1 = psi.T @ psi * dz
    assert np.max(np.abs(S1 - np.diag(np.diag(S1)))) > 1e-3        # one-band states overlap
    assert ov < 1e-8
