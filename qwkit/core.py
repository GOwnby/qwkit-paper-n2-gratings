"""
core.py -- one-band envelope-function solver and resonant intersubband optics for layered
heterostructures under a static field.

Model:
  * Conduction band only, BenDaniel-Duke kinetic operator, finite differences on a cell-centred grid
    (mirror-symmetric stacks are exactly symmetric on the grid).
  * Kane-type nonparabolicity m(z, E) = m_edge(z) [1 + (E - V(z)) / Eg(z)], solved per state by
    index tracking: the k-th state is the k-th eigenvalue of H(E), which preserves node counting.
  * States of different energy-dependent Hamiltonians are Loewdin-orthonormalised (all localized states, before
    any truncation) before dipoles are formed; dipoles are measured from the active-region centre.
    kane_valence/kane_dipoles give the exact alternative: as two-component Kane states (conduction envelope plus
    the valence component the energy-dependent mass stands for) the states are orthogonal without Loewdin.
  * Populations: Fermi-Dirac with a parabolic 2D density of states (populations).
  * chi2_general: density-matrix SHG susceptibility for arbitrary populations (translation
    invariant; contains permanent-dipole terms). chi2_shg / alpha_w: ground-state-only forms.
Materials are looked up by name in MAT (offset V in eV relative to In0.53Ga0.47As, band-edge mass m
in units of m0, gap Eg in eV); add more with add_material().
Units: Angstrom and eV inside; field in kV/cm (positive F lowers the potential towards +z);
chi(2) returned in pm/V.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal

HB2M0 = 3.80998  # hbar^2 / (2 m0) in eV*A^2
E_CH = 1.602176634e-19
EPS0 = 8.8541878128e-12

# Material parameters (InP lattice-matched, standard literature values)
MAT = {
    "well": dict(V=0.0, m=0.043, Eg=0.75),     # In0.53Ga0.47As
    "barrier": dict(V=0.52, m=0.075, Eg=1.45),  # In0.52Al0.48As, conduction-band offset 0.52 eV
}


def add_material(name, V, m, Eg, eps=None):
    """Register a material for use in layer lists: conduction-band offset V (eV, relative to the
    InGaAs well), band-edge mass m (m0), gap Eg (eV) and, optionally, static permittivity eps for the
    self-consistent solver (sp.EPS_R)."""
    MAT[name] = dict(V=float(V), m=float(m), Eg=float(Eg))
    if eps is not None:
        from . import sp
        sp.EPS_R[name] = float(eps)


def build(layers, pad=100.0, dz=0.5, pad_material="barrier"):
    """layers: list of (thickness_A, kind) with kind a key of MAT. Returns (z, V0, m, Eg, active):
    cell-centred grid (A), band-edge potential (eV), mass (m0), gap (eV) and the (start, end) of the
    active region. Pads with `pad` A of `pad_material` on each side."""
    seq = [(pad, pad_material)] + list(layers) + [(pad, pad_material)]
    total = sum(t for t, _ in seq)
    # cell-centred grid: no node sits on an interface, so mirror-symmetric layer
    # stacks give exactly mirror-symmetric discrete potentials
    n = int(round(total / dz))
    z = (np.arange(n) + 0.5) * dz
    V0 = np.empty_like(z); m = np.empty_like(z); Eg = np.empty_like(z)
    edges = np.cumsum([0] + [t for t, _ in seq])
    for i, (t, kind) in enumerate(seq):
        sel = (z >= edges[i]) & (z < edges[i + 1])
        V0[sel] = MAT[kind]["V"]; m[sel] = MAT[kind]["m"]; Eg[sel] = MAT[kind]["Eg"]
    active = (pad, total - pad)
    return z, V0, m, Eg, active


def potential(z, V0, active, F_kVcm):
    """Add a uniform field F (kV/cm; 1 kV/cm = 1e-5 V/A), zero of tilt at the active-region centre."""
    zc = 0.5 * (active[0] + active[1])
    return V0 - F_kVcm * 1e-5 * (z - zc)


def _solve(z, V, mz, nstates, first=0):
    """Eigenstates first..nstates-1 of -d/dz (HB2M0/m dpsi/dz) + V psi with hard walls (psi=0 beyond grid)."""
    dz = z[1] - z[0]
    mh = 0.5 * (mz[:-1] + mz[1:])           # mass on half points
    t = HB2M0 / mh / dz**2                    # coupling
    diag = V.copy()
    diag[:-1] += t; diag[1:] += t
    # hard walls: psi = 0 at ghost nodes one step beyond each end (Dirichlet; without these two terms the ends
    # would be zero-flux boundaries, which admit spurious states localized at the box edges)
    diag[0] += HB2M0 / mz[0] / dz**2; diag[-1] += HB2M0 / mz[-1] / dz**2
    off = -t
    w, v = eigh_tridiagonal(diag, off, select="i", select_range=(first, nstates - 1))
    v /= np.sqrt(dz)                          # normalise: sum |psi|^2 dz = 1
    return w, v


def solve_states(z, V, m_edge, Eg, nstates=6, nonparabolic=True, iters=30, tol=1e-7, E_guess=None):
    """Return energies (eV) and wavefunctions (columns), including nonparabolicity per state."""
    E, psi = _solve(z, V, m_edge, nstates)
    if not nonparabolic:
        return E, psi
    # Index tracking: the k-th state satisfies E = k-th eigenvalue of H(E). With an
    # energy-dependent (Sturm-Liouville) mass the node count, hence the index, is preserved,
    # so this cannot jump between states or return duplicates (nearest-energy matching can).
    Es, Ps = [], []
    for k in range(nstates):
        Ek = E[k] if (E_guess is None or k >= len(E_guess)) else E_guess[k]
        # root of g(E) = lambda_k(E) - E by secant steps (g decreases monotonically with E), falling back to the
        # damped fixed point if a secant step misbehaves
        Ep = gp = None
        for _ in range(iters):
            mz = np.clip(m_edge * (1.0 + (Ek - V) / Eg), 0.2 * m_edge, None)
            w, v = _solve(z, V, mz, k + 1, first=k)       # only the k-th eigenpair (O(N) instead of O(N k))
            g = w[0] - Ek
            if abs(g) < tol:
                Ek = w[0]; break
            if Ep is not None and gp != g:
                En = Ek - g * (Ek - Ep) / (g - gp)
                if not np.isfinite(En) or abs(En - Ek) > 2 * abs(g) + 1e-6:
                    En = Ek + 0.5 * g
            else:
                En = Ek + 0.5 * g
            Ep, gp, Ek = Ek, g, En
        mz = np.clip(m_edge * (1.0 + (Ek - V) / Eg), 0.2 * m_edge, None)
        w, v = _solve(z, V, mz, k + 1, first=k)
        Es.append(w[0]); Ps.append(v[:, 0])
    return np.array(Es), np.array(Ps).T


def localized_fraction(z, psi, active, margin=25.0):
    """Probability of each state within `margin` A of the active region."""
    dz = z[1] - z[0]
    sel = (z >= active[0] - margin) & (z <= active[1] + margin)
    return np.sum(psi[sel] ** 2, axis=0) * dz / np.maximum(np.sum(psi ** 2, axis=0) * dz, 1e-300)


def localized(z, psi, active, margin=25.0, thresh=0.8):
    """Boolean mask of states with at least `thresh` of their probability within `margin` A of the
    active region (drops continuum-like states of the padded box)."""
    dz = z[1] - z[0]
    sel = (z >= active[0] - margin) & (z <= active[1] + margin)
    return (np.sum(psi[sel] ** 2, axis=0) * dz) >= thresh


def lowdin_localized(z, psi, active, E=None, V=None, m_edge=None, Eg=None, nbuf=10, margin=25.0, thresh=0.8,
                     tol=1e-7, nmax=80, ebuf=None):
    """Localized states, Loewdin-orthonormalized as part of a set that extends at least `nbuf` box states above the
    highest localized one. Returns (E_loc, psi_loc, idx) with idx the indices of the localized states.
    Orthonormalizing the localized states alone gives the topmost one spurious dipoles when few states are localized
    (it cannot be orthogonalized against the states left out): for a double well in a 200 A box with three localized
    states, z13 comes out 7% too large, against 0.7% with the buffer (two-band Kane metric as reference).
    If fewer than nbuf states are available above the highest localized one and V, m_edge, Eg are given, more states
    are solved at the same potential (index-tracked eigenpairs of the same Hamiltonians, so the given states are
    reproduced); without V the available states are used as they are. ebuf (eV, optional): also include every state
    up to ebuf above the highest localized one (a buffer of a fixed number of states spans less energy in a larger box;
    METH-01 recommends >= 0.5 eV). The two-band dipoles (kane_dipoles) need no buffer at all."""
    dz = z[1] - z[0]
    psi = psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz)
    while True:
        idx = np.where(localized(z, psi, active, margin=margin, thresh=thresh))[0]
        need = (int(idx[-1]) + 1 if len(idx) else 0) + nbuf
        if ebuf is not None and E is not None and len(idx):
            Etop = np.asarray(E)[idx[-1]] + ebuf
            above = np.where(np.asarray(E) > Etop)[0]
            need = max(need, int(above[0]) if len(above) else psi.shape[1] + 5)
        if psi.shape[1] >= need or V is None or psi.shape[1] >= nmax:
            break
        E, psi = solve_states(z, V, m_edge, Eg, nstates=min(need, nmax), E_guess=E, tol=tol)
        psi = psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz)
    P = orthonormalize(z, psi)[:, idx]
    return (None if E is None else np.asarray(E)[idx]), P, idx


def orthonormalize(z, psi):
    """Loewdin-orthonormalise wavefunction columns (minimal change, preserves parity). Needed for states of
    different energy-dependent (nonparabolic) Hamiltonians, which are not exactly orthogonal."""
    dz = z[1] - z[0]
    S = psi.T @ psi * dz
    w_, U_ = np.linalg.eigh(S)
    return psi @ (U_ @ np.diag(w_ ** -0.5) @ U_.T)


def kane_valence(z, psi, E, V, m_edge, Eg):
    """Valence (light-particle) components of the states of the energy-dependent-mass Hamiltonian in the two-band
    Kane model it derives from, m(E) = m*(1 + (E - V)/Eg): continuum phi_n = sqrt(HB2M0 Eg / m*) psi_n' / (E_n - V + Eg).
    Discretized exactly as the solver (_solve) discretizes H(E): on each half point the mass is the mean of the node
    masses, m_h(E) = A_h + B_h E with B_h = (m*_i / Eg_i + m*_i+1 / Eg_i+1) / 2, and phi_h = sqrt(HB2M0 B_h) / m_h(E)
    (psi_i+1 - psi_i) / dz; the hard walls add one point each, half a step outside the grid. With this partner the
    two-component states (psi_n, phi_n) are orthogonal to the accuracy of the eigenvalues (the product phi_m phi_n is
    the divided difference of the discrete H(E) between E_m and E_n); psi_n alone are not. Returns (zh, phi) with zh
    the N + 1 points (walls and half points) carrying phi."""
    dz = z[1] - z[0]
    E = np.asarray(E)
    zh = np.concatenate([[z[0] - dz / 2], 0.5 * (z[1:] + z[:-1]), [z[-1] + dz / 2]])
    b_node = m_edge / Eg                                                     # dm/dE at the nodes
    m_node = np.clip(m_edge[:, None] * (1.0 + (E[None, :] - V[:, None]) / Eg[:, None]), 0.2 * m_edge[:, None], None)
    mh = np.vstack([m_node[:1], 0.5 * (m_node[1:] + m_node[:-1]), m_node[-1:]])
    bh = np.concatenate([b_node[:1], 0.5 * (b_node[1:] + b_node[:-1]), b_node[-1:]])
    pad = np.zeros((1, psi.shape[1]))
    dpsi = np.diff(np.vstack([pad, psi, pad]), axis=0) / dz
    return zh, np.sqrt(HB2M0 * bh)[:, None] / mh * dpsi


def kane_dipoles(z, psi, E, V, m_edge, Eg, zc=0.0):
    """Dipole matrix (A) of the two-component Kane states, each normalized over both components; no Loewdin step.
    Returns (Z, overlap) with overlap the largest off-diagonal two-band overlap (a discretization check)."""
    dz = z[1] - z[0]
    zh, phi = kane_valence(z, psi, E, V, m_edge, Eg)
    S = psi.T @ psi * dz + phi.T @ phi * dz
    nrm = np.sqrt(np.diag(S))
    Z = ((psi.T * (z - zc)) @ psi * dz + (phi.T * (zh - zc)) @ phi * dz) / np.outer(nrm, nrm)
    Sn = S / np.outer(nrm, nrm)
    return Z, float(np.max(np.abs(Sn - np.eye(len(nrm)))))


def dipoles(z, psi, orthonormal=True):
    """Dipole matrix z_ij (A) for wavefunction columns psi on grid z (A).
    With orthonormal=True (default) the states are first Loewdin-orthonormalised, so the off-diagonal
    elements do not depend on the coordinate origin. Raw nonparabolic states from solve_states overlap
    slightly, and <i|z|j> of non-orthogonal states shifts by -a<i|j> when the origin moves by a."""
    dz = z[1] - z[0]
    if orthonormal:
        S = psi.T @ psi * dz
        if np.max(np.abs(S - np.eye(S.shape[0]))) > 1e-12:
            psi = orthonormalize(z, psi)
    return (psi.T * z) @ psi * dz            # Angstrom


def chi2_shg(E, Z, hw, gamma=0.010, N_cm3=1e18):
    """Resonant SHG chi(2) from ground state (index 0). Returns complex chi in pm/V."""
    N = N_cm3 * 1e6
    tot = 0j
    n = len(E)
    # Origin independence: diagonal elements enter as (z_nn - z_11), the permanent-dipole
    # difference that carries field-induced chi(2) (Fejer et al. 1989 mechanism).
    Z = Z - Z[0, 0] * np.eye(n)
    for mi in range(1, n):
        for ni in range(1, n):
            num = Z[0, ni] * Z[ni, mi] * Z[mi, 0] * 1e-30          # m^3
            d1 = (E[ni] - E[0] - 2 * hw - 1j * gamma) * E_CH
            d2 = (E[mi] - E[0] - hw - 1j * gamma) * E_CH
            tot += num / (d1 * d2)
    chi = N * E_CH**3 / EPS0 * tot                                  # m/V
    return chi * 1e12


def alpha_w(E, Z, hw, gamma=0.010, N_cm3=1e18, nref=3.1):
    """Linear intersubband absorption at hw from the ground state, Lorentzian, TM, cm^-1."""
    N = N_cm3 * 1e6
    hbar = 1.054571817e-34; c = 2.99792458e8
    w = hw * E_CH / hbar
    a = 0.0
    for mi in range(1, len(E)):
        Em = (E[mi] - E[0])
        L = (gamma / np.pi) / ((Em - hw) ** 2 + gamma**2) / E_CH   # per Joule
        a += (Z[0, mi] * 1e-10) ** 2 * L
    a *= np.pi * w * N * E_CH**2 / (nref * c * EPS0)               # m^-1: alpha = pi w N e^2 z^2 L(E) / (n c eps0)
    return a / 100.0


def analyze(layers, F_kVcm, hw, gamma=0.010, nstates=6, pad=100.0, N_cm3=1e18, dz=0.5):
    """Fixed-potential (non-self-consistent) analysis: localized, Loewdin-orthonormalised states, dipoles,
    ground-state-only chi(2) (pm/V) and absorption (1/cm) at photon energy hw (eV) for density N_cm3."""
    z, V0, m, Eg, active = build(layers, pad=pad, dz=dz)
    V = potential(z, V0, active, F_kVcm)
    E, psi = solve_states(z, V, m, Eg, nstates=nstates + 2)
    # States of different energy-dependent Hamiltonians are not exactly orthogonal; a residual
    # overlap times the absolute coordinate fakes a dipole. Loewdin-orthonormalise (preserves
    # parity, minimal change) the localized states together with a buffer of box states above them BEFORE truncating
    # to nstates (orthonormalizing a truncated set makes its top states depend on the truncation), and measure z from
    # the active-region centre. kane_dipoles gives the exact two-band alternative.
    E, psi, _ = lowdin_localized(z, psi, active, E=E, V=V, m_edge=m, Eg=Eg)
    E, psi = E[:nstates], psi[:, :nstates]
    zc = 0.5 * (active[0] + active[1])
    Z = dipoles(z - zc, psi)
    chi = chi2_shg(E, Z, hw, gamma, N_cm3)
    a = alpha_w(E, Z, hw, gamma, N_cm3)
    return dict(E=E, Z=Z, chi=chi, alpha=a, z=z, V=V, psi=psi, active=active)


def wkb_escape_time(z, V, E_level, active, m_b=0.075, attempt_fs=None):
    """Crude WKB escape time (ps) of a level through the downhill (+z) barrier.
    Integrates kappa from the last classical turning point inside the active region outward.
    `active` is the (start, end) tuple returned by build()."""
    if active is None:
        raise ValueError("active region (start, end) is required")
    hbar = 1.054571817e-34; m0 = 9.1093837e-31
    zr = z[z >= active[1] - 1.0]; Vr = V[z >= active[1] - 1.0]
    above = Vr > E_level
    if not above.any():
        return 0.0
    i0 = np.argmax(above)
    if above[i0:].all():
        return np.inf  # never escapes within the grid (no turning point) -> bound
    i1 = i0 + np.argmin(above[i0:])
    dz = (z[1] - z[0]) * 1e-10
    kappa = np.sqrt(2 * m_b * m0 * (Vr[i0:i1] - E_level) * E_CH) / hbar
    T = np.exp(-2 * np.sum(kappa) * dz)
    if attempt_fs is None:
        attempt_fs = 50.0  # classical round-trip in a ~70 A well at ~0.2 eV kinetic energy
    return attempt_fs * 1e-3 / T


def escape_broadening(r, n, F_kVcm, m_b=0.075, m_w=0.043):
    """Tunnelling broadening hbar/tau (meV) of state n through the downhill barrier, WKB for a
    triangular barrier starting at the active-region edge. Returns (Gamma_meV, U_eV, d_A):
    U = barrier height above the level at the edge, d = distance to the classical turning point.
    U <= 0 means the state is not bound on that side (returns inf)."""
    hbar = 1.054571817e-34; m0 = 9.1093837e-31
    z, V, act = r["z"], r["V"], r["active"]
    F = abs(F_kVcm)
    if F == 0:
        return 0.0, None, None
    edge = act[1] if F_kVcm > 0 else act[0]
    zc = 0.5 * (act[0] + act[1])
    Vedge = MAT["barrier"]["V"] - F * 1e-5 * abs(edge - zc)
    E = r["E"][n]
    U = Vedge - E
    if U <= 0:
        return np.inf, U, 0.0
    eF = F * 1e5 * E_CH  # J/m
    exponent = (4.0 / 3.0) * np.sqrt(2 * m_b * m0) * (U * E_CH) ** 1.5 / (hbar * eF)
    T = np.exp(-exponent)
    # attempt frequency: classical bounce across the active region
    sel = (z >= act[0]) & (z <= act[1])
    K = max(E - V[sel].min(), 0.02)
    v = np.sqrt(2 * K * E_CH / (m_w * m0))
    W = (act[1] - act[0]) * 1e-10
    nu = v / (2 * W)
    G = hbar * nu * T / E_CH * 1e3
    d = U / (F * 1e-5)
    return G, U, d


def populations(E, ns_cm2, T, m_eff=0.043):
    """Fermi-Dirac subband occupations (fractions summing to 1) for sheet density ns at T (K),
    parabolic 2D density of states. T = 0 -> ground state only (if it holds all carriers)."""
    if T <= 0:
        rho = np.zeros(len(E)); rho[0] = 1.0; return rho
    from scipy.optimize import brentq
    kB = 8.617333e-5
    hbar = 1.054571817e-34; m0 = 9.1093837e-31
    D = m_eff * m0 / (np.pi * hbar**2) * E_CH * 1e-4   # states / (eV cm^2)
    kT = kB * T
    def ntot(mu):
        x = (mu - E) / kT
        return np.sum(D * kT * np.logaddexp(0, x))
    mu = brentq(lambda mu: ntot(mu) - ns_cm2, E[0] - 2.0, E[-1] + 2.0)
    n_i = D * kT * np.logaddexp(0, (mu - E) / kT)
    return n_i / n_i.sum()


def chi2_general(E, Z, rho, hw, gamma=0.010, N_cm3=1e18):
    """Resonant density-matrix SHG susceptibility with arbitrary subband populations:
    chi = (N e^3/eps0) sum_{l,m,n} z_lm z_mn z_nl / (E_nl - 2hw - iG) *
          [ (rho_l - rho_m)/(E_ml - hw - iG) - (rho_m - rho_n)/(E_nm - hw - iG) ]
    Reduces to Eq. 15.28 for three levels with only level 1 populated, and contains the
    permanent-dipole term z12^2 (z22 - z11) automatically. Translation invariant. pm/V."""
    N = N_cm3 * 1e6
    n = len(E)
    D = E[:, None] - E[None, :]            # D[a, b] = E_a - E_b
    tot = 0j
    for l in range(n):
        for m in range(n):
            for k in range(n):             # k plays the role of n
                num = Z[l, m] * Z[m, k] * Z[k, l]
                if num == 0.0:
                    continue
                br = (rho[l] - rho[m]) / (D[m, l] - hw - 1j * gamma) - (rho[m] - rho[k]) / (D[k, m] - hw - 1j * gamma)
                tot += num / (D[k, l] - 2 * hw - 1j * gamma) * br
    return N * E_CH**3 / EPS0 * tot * 1e-30 / E_CH**2 * 1e12


def analyze_T(layers, F_kVcm, hw, Ts=(300.0, 77.0), gamma=0.010, nstates=6, pad=100.0,
              ns_cm2=5e11, dz=0.5, spacer=200.0):
    """Solve once, then chi(2) at each temperature in Ts for a fixed sheet density per period.
    N_3D = ns / (active + spacer). Returns r with r['chi'][T], r['rho'][T], r['N_cm3']."""
    z, V0, m, Eg, active = build(layers, pad=pad, dz=dz)
    Lp = (active[1] - active[0] + spacer) * 1e-8
    N_cm3 = ns_cm2 / Lp
    r = analyze(layers, F_kVcm, hw, gamma=gamma, nstates=nstates, pad=pad, N_cm3=N_cm3, dz=dz)
    r["N_cm3"] = N_cm3
    r["chiT"], r["rhoT"] = {}, {}
    for T in Ts:
        rho = populations(r["E"], ns_cm2, T)
        r["rhoT"][T] = rho
        r["chiT"][T] = chi2_general(r["E"], r["Z"], rho, hw, gamma, N_cm3)
    return r
