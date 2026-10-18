"""
scattering.py -- intersubband LO-phonon scattering rates (Froehlich interaction, bulk phonons) and the
relaxation matrix used by propagation.Medium.

For an electron at the bottom of subband i (k = 0) scattering to subband f (Ferreira & Bastard,
Phys. Rev. B 40, 1074 (1989)):
    W_if = e^2 m* w_LO C / (4 eps0 hbar^2) * [ (n+1) I(Q_e)/Q_e  +  n I(Q_a)/Q_a ]
    C = 1/eps_inf - 1/eps_s,   n = 1/(exp(hbar w_LO / kT) - 1)
    I(Q) = int int phi(z) phi(z') exp(-Q |z - z'|) dz dz',   phi = psi_i psi_f
    hbar^2 Q_e^2 / 2m* = E_i - E_f - hbar w_LO   (emission, only if > 0)
    hbar^2 Q_a^2 / 2m* = E_i - E_f + hbar w_LO   (absorption)
Downward rates (E_i > E_f) are computed from this expression; upward rates follow from detailed
balance, W_fi = W_if exp(-(E_i - E_f)/kT), so that the Boltzmann distribution is stationary.
Defaults are for In0.53Ga0.47As (hbar w_LO = 33.7 meV, eps_inf = 11.6, eps_s = 13.9, m* = 0.043).
Interface-roughness, alloy, acoustic-phonon and electron-electron scattering are neglected, so the
rates are lower bounds and lifetimes upper bounds (conservative for population bottlenecks).
Units: z in Angstrom, psi normalised as sum |psi|^2 dz = 1 (dz in Angstrom), E in eV; rates in 1/s.
"""
import numpy as np

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12; HBAR = 1.054571817e-34; M0 = 9.1093837015e-31
KB = 8.617333262e-5


def form_integral(z, phi, Q):
    """I(Q) = int int phi(z) phi(z') exp(-Q|z-z'|) dz dz'  (z in A, Q in 1/m, phi in 1/A). Returns 1/m."""
    zm = z * 1e-10
    dz = (z[1] - z[0]) * 1e-10
    ph = phi * 1e10                                   # 1/m
    # O(N) evaluation with running exponential sums (exact for a uniform grid)
    e = np.exp(-Q * dz)
    left = np.zeros_like(ph); right = np.zeros_like(ph)
    acc = 0.0
    for i in range(len(ph)):
        acc = acc * e + ph[i]
        left[i] = acc
    acc = 0.0
    for i in range(len(ph) - 1, -1, -1):
        acc = acc * e + ph[i]
        right[i] = acc
    inner = left + right - ph                         # sum_j phi_j exp(-Q|z_i - z_j|)
    return float(np.sum(ph * inner) * dz * dz)


def lo_rate(z, psi_i, psi_f, dE, T=300.0, hwlo=0.0337, eps_inf=11.6, eps_s=13.9, m_eff=0.043, C=None, q_s=0.0):
    """Downward LO-phonon rate (1/s) from subband i to subband f with E_i - E_f = dE > 0.
    C overrides the Froehlich factor 1/eps_inf - 1/eps_s (one mode of a multimode crystal, krpa.froehlich_weights);
    q_s (1/m) applies static two-dimensional Thomas-Fermi screening, |M|^2 / (1 + q_s/Q)^2."""
    C = 1.0 / eps_inf - 1.0 / eps_s if C is None else C
    m = m_eff * M0
    wlo = hwlo * QE / HBAR
    n = 1.0 / np.expm1(hwlo / (KB * T)) if T > 0 else 0.0
    phi = psi_i * psi_f
    pref = QE ** 2 * m * wlo * C / (4 * EPS0 * HBAR ** 2)
    W = 0.0
    if dE > hwlo:
        Qe = np.sqrt(2 * m * (dE - hwlo) * QE) / HBAR
        W += (n + 1) * form_integral(z, phi, Qe) / Qe / (1 + q_s / Qe) ** 2
    Qa = np.sqrt(2 * m * (dE + hwlo) * QE) / HBAR
    W += n * form_integral(z, phi, Qa) / Qa / (1 + q_s / Qa) ** 2
    return pref * W


def rate_matrix(z, psi, E, T=300.0, **kw):
    """R[l, m] = rate (1/s) from subband l to subband m, with detailed balance for upward rates."""
    n = len(E)
    R = np.zeros((n, n))
    for i in range(n):
        for f in range(n):
            if E[i] > E[f]:
                R[i, f] = lo_rate(z, psi[:, i], psi[:, f], E[i] - E[f], T=T, **kw)
                R[f, i] = R[i, f] * np.exp(-(E[i] - E[f]) / (KB * T)) if T > 0 else 0.0
    return R


def lo_rate_modes(z, psi_i, psi_f, dE, modes, T=300.0, m_eff=0.043, q_s=0.0):
    """Sum of lo_rate over polar LO modes [(hw_LO, C_j), ...] (eV, Froehlich factor per mode)."""
    return sum(lo_rate(z, psi_i, psi_f, dE, T=T, hwlo=hw, m_eff=m_eff, C=Cj, q_s=q_s) for hw, Cj in modes)


def tf_wavevector(m_eff, eps, n_sub=1):
    """Two-dimensional Thomas-Fermi screening wavevector (1/m) for n_sub occupied subbands: n_sub e^2 m/(2 pi eps eps0 hbar^2)."""
    return n_sub * QE ** 2 * m_eff * M0 / (2 * np.pi * eps * EPS0 * HBAR ** 2)


# ------------------------------------------------------------------------------------------------------------------
# k-resolved, Pauli-blocked intersubband LO-phonon rates with static RPA screening by the occupied subbands
# ------------------------------------------------------------------------------------------------------------------
def form_factor(z, phi_a, phi_b, Q):
    """F_ab(Q) = int int phi_a(z) phi_b(z') exp(-Q|z-z'|) dz dz' (phi in 1/A, z in A, Q in 1/m): dimensionless."""
    zm = z * 1e-10
    dz = (z[1] - z[0]) * 1e-10
    pa = phi_a * 1e10; pb = phi_b * 1e10
    e = np.exp(-Q * dz)
    left = np.zeros_like(pb); right = np.zeros_like(pb)
    acc = 0.0
    for i in range(len(pb)):
        acc = acc * e + pb[i]; left[i] = acc
    acc = 0.0
    for i in range(len(pb) - 1, -1, -1):
        acc = acc * e + pb[i]; right[i] = acc
    return float(np.sum(pa * (left + right - pb)) * dz * dz)


def screening_factor(z, psi, i, f, occ, Qs, eps):
    """Static RPA screening of the i->f Froehlich vertex by intrasubband polarization of the occupied subbands.
    occ: list of (l, m_l [m0], kF_l [1/m]) for occupied subbands. Returns S(Q) = |M_scr|^2/|M|^2 summed over q_z,
    S = [F_if,if + 2 sum_l c_l F_if,ll + sum_lm c_l c_m F_ll,mm] / F_if,if, c = -v F_if,. (1 + Pi v F)^-1 Pi."""
    phi_if = psi[:, i] * psi[:, f]
    L = [o[0] for o in occ]
    out = []
    for Q in np.atleast_1d(Qs):
        v = QE ** 2 / (2 * eps * EPS0 * Q)                                       # J m^2
        Pi = []
        for (l, ml, kF) in occ:
            D = ml * M0 / (np.pi * HBAR ** 2)                                    # 1/(J m^2)
            Pi.append(D * (1.0 if Q <= 2 * kF else 1.0 - np.sqrt(1.0 - (2 * kF / Q) ** 2)))
        Pi = np.array(Pi)
        Fll = np.array([[form_factor(z, psi[:, a] ** 2, psi[:, b] ** 2, Q) for b in L] for a in L])
        Fil = np.array([form_factor(z, phi_if, psi[:, a] ** 2, Q) for a in L])
        Fii = form_factor(z, phi_if, phi_if, Q)
        A = np.eye(len(L)) + (Pi[:, None] * v) * Fll
        c = -v * Fil @ np.linalg.solve(A, np.diag(Pi))
        out.append((Fii + 2 * c @ Fil + c @ Fll @ c) / Fii)
    return np.array(out)


def lo_rate_k(z, psi_i, psi_f, k, E_i, E_f0, m_f, G_f, modes, T=300.0, mu=None, S=None, ntheta=48, IQ=None):
    """Rate (1/s) out of an electron of in-plane wavevector k (1/m) and energy E_i (eV) into subband f (edge E_f0,
    in-plane mass m_f [m0], nonparabolicity G_f: hbar^2k^2/2m_f = eps(1+eps/G_f)) by emission and absorption of the
    polar LO modes [(hw_j, C_j)], with angular integration over the final wavevector, the nonparabolic final density
    of states, Pauli blocking (Fermi level mu; None = empty final subband) and an optional screening factor S(Q).
    IQ: optional callable Q -> form_integral(Q)/Q (e.g. an interpolant) to avoid recomputing the form factor."""
    W = 0.0
    th = (np.arange(ntheta) + 0.5) * 2 * np.pi / ntheta
    phi = psi_i * psi_f
    for hw, Cj in modes:
        n = 1.0 / np.expm1(hw / (KB * T)) if T > 0 else 0.0
        pref = QE ** 2 * (hw * QE / HBAR) * Cj / (4 * EPS0 * HBAR ** 2) * M0
        for sgn, occ in ((-1, n + 1.0), (+1, n)):
            eps = E_i + sgn * hw - E_f0
            if eps <= 0:
                continue
            x0 = m_f * eps * (1 + eps / G_f)                                      # hbar^2 k'^2 / 2 m0 (eV)
            kp = np.sqrt(2 * M0 * x0 * QE) / HBAR
            mdos = m_f * (1 + 2 * eps / G_f)
            block = 1.0 if mu is None else 1.0 - 1.0 / (1.0 + np.exp((E_f0 + eps - mu) / (KB * T)))
            Q = np.sqrt(np.maximum(k ** 2 + kp ** 2 - 2 * k * kp * np.cos(th), 1e4))
            vals = IQ(Q) if IQ is not None else np.array([form_integral(z, phi, q_) / q_ for q_ in Q])
            if S is not None:
                vals = vals * S(Q)
            W += pref * mdos * occ * block * vals.mean()
    return W
