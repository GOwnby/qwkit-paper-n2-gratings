"""Tests of qwkit.transport: screened remote-impurity mobility against an independent quadrature of the same
Golden-rule integral, and the Drude Hall algebra of parallel channels."""
import numpy as np
from scipy.integrate import quad
from qwkit import transport as tr


def test_remote_impurity_mobility_matches_quadrature():
    QE, EPS0, HBAR, M0 = tr.QE, tr.EPS0, tr.HBAR, tr.M0
    z = np.arange(-300, 100, 0.5); psi = np.exp(-(z / 5.0) ** 2 / 2)[:, None]
    n, d, N, eps, m = 1e12, 100.0, 1e11, 12.9, 0.067 * M0
    r = dict(z=z, psi=psi, E=np.array([0.0]), mu=0.01, n_n=np.array([n]), m_n=np.array([0.067]), G_n=np.array([1e9]))
    o = tr.mobilities(r, eps=eps, sheets=[(-d, N, 1)], nth=200)
    k = np.sqrt(2 * np.pi * n * 1e4); qtf = m * QE ** 2 / (2 * np.pi * EPS0 * eps * HBAR ** 2)
    zz = z * 1e-10; dz = 0.5e-10; p2 = psi[:, 0] ** 2 / np.sum(psi[:, 0] ** 2 * dz)

    def integrand(t):
        q = 2 * k * np.sin(t / 2)
        if q < 1:
            return 0.0
        F = np.sum(p2 * np.exp(-q * np.abs(zz + d * 1e-10))) * dz
        V = QE ** 2 / (2 * EPS0 * eps * q) * F / (1 + qtf / q * tr._G(zz, p2, p2, q))
        return V ** 2 * (1 - np.cos(t))
    rate = m * N * 1e4 / (2 * np.pi * HBAR ** 3) * quad(integrand, 0, 2 * np.pi, limit=300)[0]
    assert abs(o["mu"][0] / (QE / (rate * m) * 1e4) - 1) < 1e-3


def test_hall_of_parallel_channels():
    nH, muH = tr.hall([3e13, 1e13], [1000, 1000])
    assert abs(nH - 4e13) < 1e6 and abs(muH - 1000) < 1e-6
    nH, _ = tr.hall([8e13, 2.6e13], [640, 2900])          # a more mobile minority channel lowers the Hall density
    assert nH < 8e13
    nH0, _ = tr.hall([3e13, 1e13], [2000, 100]); nH7, _ = tr.hall([3e13, 1e13], [2000, 100], B=7.0)
    assert abs(nH7 / nH0 - 1) < 0.02                       # weakly nonlinear: a slow channel hardly bends R_xy


def test_interface_roughness_matches_quadrature():
    QE, EPS0, HBAR, M0 = tr.QE, tr.EPS0, tr.HBAR, tr.M0
    z = np.arange(-40, 200, 0.5); psi = np.where(z > 0, z * np.exp(-z / 15.0), 0.02 * np.exp(z / 3.0))[:, None]
    n, eps, m, V0, D, L = 5e12, 20.0, 0.19 * M0, 1.6, 2.0, 40.0
    r = dict(z=z, psi=psi, E=np.array([0.0]), mu=0.01, n_n=np.array([n]), m_n=np.array([0.19]), G_n=np.array([1e9]))
    o = tr.mobilities(r, eps=eps, roughness=(V0, D, L, 0.0), nth=200)
    k = np.sqrt(2 * np.pi * n * 1e4); qtf = m * QE ** 2 / (2 * np.pi * EPS0 * eps * HBAR ** 2)
    zz = z * 1e-10; dz = 0.5e-10; ps = psi[:, 0] / np.sqrt(np.sum(psi[:, 0] ** 2 * dz)); p2 = ps ** 2
    i0 = int(np.argmin(np.abs(zz))); pI = 0.5 * (ps[i0] + ps[i0 + 1])

    def integrand(t):
        q = 2 * k * np.sin(t / 2)
        if q < 1:
            return 0.0
        V = V0 * QE * pI ** 2 / (1 + qtf / q * tr._G(zz, p2, p2, q))
        return V ** 2 * np.pi * (D * 1e-10 * L * 1e-10) ** 2 * np.exp(-(q * L * 1e-10) ** 2 / 4) * (1 - np.cos(t))
    rate = m / (2 * np.pi * HBAR ** 3) * quad(integrand, 0, 2 * np.pi, limit=300)[0]
    assert abs(o["mu"][0] / (QE / (rate * m) * 1e4) - 1) < 2e-3


def test_coupled_relaxation_matches_direct_boltzmann():
    """Two subbands with different Fermi wavevectors and masses, a kernel W_ij(theta) = w_ij (1 + c_ij cos theta) with
    w_ij symmetric (detailed balance); the linearized Boltzmann equation on an angular grid,
    e E v_i cos(phi) = sum_j (m_j / 2 pi hbar^2) int dphi' W_ij(phi - phi') [g_i(phi) - g_j(phi')] (in units of the drive),
    is solved directly and the relaxation times g_i = v_i tau_i cos(phi) compared with transport.relax."""
    k = np.array([2.0e9, 0.8e9]); md = np.array([0.30, 0.22]) * tr.M0; HB = tr.HBAR
    w = np.array([[3.0, 1.2], [1.2, 2.0]]) * 1e-40; c = np.array([[0.6, 0.3], [0.3, 0.8]])
    M = 64; ph = 2 * np.pi * np.arange(M) / M; d = 2 * np.pi / M
    v = HB * k / md
    # unknowns g_i(phi_m); equation per (i, m)
    A = np.zeros((2 * M, 2 * M)); b = np.zeros(2 * M)
    for i in range(2):
        for m in range(M):
            row = i * M + m; b[row] = v[i] * np.cos(ph[m])
            for j in range(2):
                pref = md[j] / (2 * np.pi * HB ** 3)
                W = w[i, j] * (1 + c[i, j] * np.cos(ph[m] - ph))
                A[row, i * M + m] += pref * np.sum(W) * d
                A[row, j * M:(j + 1) * M] -= pref * W * d
    g = np.linalg.solve(A, b)
    tau_direct = np.array([np.sum(g[i * M:(i + 1) * M] * np.cos(ph)) / (v[i] * M / 2) for i in range(2)])
    a0 = np.array([[md[j] / (2 * np.pi * HB ** 3) * w[i, j] * 2 * np.pi for j in range(2)] for i in range(2)])
    a1 = np.array([[md[j] / (2 * np.pi * HB ** 3) * w[i, j] * c[i, j] * np.pi for j in range(2)] for i in range(2)])
    assert np.allclose(tr.relax(a0, a1, k, md), tau_direct, rtol=1e-9)


def test_dielectric_boundary_limits():
    """eps_cap = eps reproduces the uniform medium; without screening, a charge sheet in the cap scales the mobility by
    ((eps + eps_cap) / (2 eps))^2."""
    z = np.arange(-60, 200, 0.5); psi = np.where(z > 0, z * np.exp(-z / 15.0), 0.0)[:, None]
    r = dict(z=z, psi=psi, E=np.array([0.0]), mu=0.05, n_n=np.array([3e12]), m_n=np.array([0.19]), G_n=np.array([1e9]))
    sh = [(-5.0, 1e12, 1)]
    a = tr.mobilities(r, eps=20.0, sheets=sh, nth=64)["mu"][0]
    b = tr.mobilities(r, eps=20.0, sheets=sh, nth=64, eps_cap=20.0)["mu"][0]
    assert abs(a / b - 1) < 1e-12
    a = tr.mobilities(r, eps=20.0, sheets=sh, nth=64, screening=False)["mu"][0]
    b = tr.mobilities(r, eps=20.0, sheets=sh, nth=64, screening=False, eps_cap=40.0)["mu"][0]
    assert abs(b / a - (60.0 / 40.0) ** 2) < 1e-9


def test_magnetotransport_reduces_to_parallel_channels():
    """Without intersubband coupling (a1 diagonal, a0 diagonal) the coupled model gives the parallel-Drude MR and Hall."""
    k = np.array([2.0e9, 0.9e9]); md = np.array([0.30, 0.22])
    a0 = np.diag([2.0e12, 5.0e11]); a1 = np.diag([0.6e12, 1.0e11])
    tau = tr.relax(a0, a1, k, md * tr.M0); mu = tr.QE * tau / (md * tr.M0) * 1e4
    n = np.array([6e13, 1.5e13])
    o = dict(a0=a0, a1=a1, k=k, md=md, n=n)
    r0, _, _ = tr.magnetotransport(o, 0.0); r7, RH, nH = tr.magnetotransport(o, 7.0)
    assert abs((r7 / r0 - 1) - tr.magnetoresistance(n, mu, 7.0)) < 1e-9
    assert abs(nH / tr.hall(n, mu, 7.0)[0] - 1) < 1e-9


def test_coupled_subbands_equal_weighted_drude_modes():
    """With detailed balance (n_i m_i G_ij symmetric up to the k factors), coupled subbands in a field are exactly a set
    of independent Drude channels: the eigenmodes of M G with non-negative weights summing to n_tot."""
    rng = np.random.default_rng(3)
    k = np.array([2.2e9, 1.3e9, 0.7e9]); md = np.array([0.33, 0.26, 0.22]); n = k ** 2 / (2 * np.pi) * 1e-4
    # rates from a symmetric |M|^2 kernel: a0_ij, a1_ij proportional to m_j * W_ij with W symmetric
    W0 = rng.uniform(0.5, 2.0, (3, 3)); W0 = W0 + W0.T; W1 = 0.6 * W0 * rng.uniform(0.2, 1.0, (3, 3)); W1 = 0.5 * (W1 + W1.T)
    a0 = 1e12 * W0 * md[None, :]; a1 = 1e12 * W1 * md[None, :]
    o = dict(a0=a0, a1=a1, k=k, md=md, n=n)
    G = np.diag(a0.sum(axis=1)) - a1 * (k[None, :] / k[:, None])
    lam, V = np.linalg.eig((md * tr.M0)[:, None] * G)
    # weights: expand the unit drive in eigenvectors, current weight per mode
    c = np.linalg.solve(V, np.ones(3))
    w = np.real((n * 1e4) @ V * c)
    assert np.all(w > -1e-6 * w.sum()) and abs(w.sum() / (n.sum() * 1e4) - 1) < 1e-9
    mu = tr.QE / np.real(lam) * 1e4
    for B in (0.5, 3.0, 7.0):
        r0, _, _ = tr.magnetotransport(o, 0.0); rB, _, nH = tr.magnetotransport(o, B)
        assert abs((rB / r0 - 1) - tr.magnetoresistance(w * 1e-4, mu, B)) < 1e-9
        assert abs(nH / tr.hall(w * 1e-4, mu, B)[0] - 1) < 1e-9
