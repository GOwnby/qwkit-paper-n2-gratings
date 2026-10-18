"""
thermal.py -- intersubband (ISB) thermal emitters at elevated temperature, and their use in thermophotovoltaic (TPV)
detailed balance (research thread TPV-05).

Pieces (all functions are independent and cheap; the self-consistent state comes from qwkit.sp):
  * Temperature-dependent material data (MATT): Varshni band gaps, density-of-states masses of electrons (Gamma and
    L valleys) and holes, Gamma-L separation, LO-phonon energy, permittivities, mobility law for free carriers.
  * Carrier statistics: effective densities of states, intrinsic density n_i(T) (non-degenerate), fraction of
    electrons in the L valleys of a bulk layer at a given Fermi level.
  * isb_alpha: ISB absorption spectrum of a period with thermal subband populations, Ando-shifted transitions and the
    inhomogeneous broadening from in-plane nonparabolicity (k-resolved transition energies averaged over the
    Fermi-Dirac distributions of the two subbands).
  * linewidth: homogeneous HWHM gamma(T) = gamma_0 + (hbar/2)(W_1 + W_2) with LO-phonon out-scattering rates
    (qwkit.scattering, Bose factors). Out-scattering broadening is an upper bound (in/out-scattering correlations
    narrow the line; Ning & Li); gamma_0 stands for interface roughness and other temperature-independent terms.
  * drude_alpha: free-carrier absorption.
  * cmt_absorptance: single-mode coupled-mode absorptance (= emissivity by Kirchhoff) of a resonator with radiative
    rate g_rad, ISB absorption rate g_isb(w) and background absorption rate g_bg(w).
  * Detailed balance of a TPV cell facing an emitter: photon flux and power from a spectral emissivity, cell
    current-voltage from absorbed flux, EQE and external radiative efficiency (ERE), maximum power point.
Units: eV, Angstrom, cm^-3 / cm^-2, K; spectra on photon-energy grids in eV.
"""
import numpy as np

KB = 8.617333262e-5          # eV/K
QE = 1.602176634e-19
HBAR = 1.054571817e-34
H = 6.62607015e-34
C0 = 2.99792458e8
M0 = 9.1093837015e-31
EPS0 = 8.8541878128e-12

# Temperature-dependent material data. Varshni Eg(T) = Eg0 - a T^2 / (T + b) (eV, K). Source: Vurgaftman, Meyer &
# Ram-Mohan, J. Appl. Phys. 89, 5815 (2001): binaries directly; alloys with Varshni a, b interpolated linearly
# between the binaries and Eg0 fixed by the 300 K gap (In0.53Ga0.47As 0.735 eV, In0.52Al0.48As 1.450 eV, Al0.3Ga0.7As 1.798 eV);
# Gamma-L separations from the interpolated L gaps with the compilation's bowing parameters (InGaAs 0.555 eV,
# InAlAs 0.18 eV, Al0.3Ga0.7As 0.06 eV at 0 K; their temperature dependence follows the L-gap Varshni slope aL). m_e: Gamma band-edge mass; m_L: L-valley
# DOS mass of ONE valley (4 equivalent valleys); m_h: hole DOS mass (heavy + light). Mobilities are 300 K values
# for moderately doped material.
MATT = {
    "InGaAs": dict(Eg0=0.816, a=4.00e-4, b=145.0, m_e=0.043, m_Ll=1.23, m_Lt=0.062, dGL=0.555, aL=4.31e-4, m_h=0.47, hwlo=0.0337,
                   eps_s=13.9, eps_inf=11.6, mu300=1.0e4),
    "InAlAs": dict(Eg0=1.535, a=5.68e-4, b=303.0, m_e=0.075, m_Ll=0.97, m_Lt=0.098, dGL=0.18, aL=4.34e-4, m_h=0.62, hwlo=0.047,
                   eps_s=12.4, eps_inf=10.2, mu300=3.0e3),
    "InP": dict(Eg0=1.4236, a=3.63e-4, b=162.0, m_e=0.080, m_Ll=1.0, m_Lt=0.1, dGL=0.59, aL=3.63e-4, m_h=0.60, hwlo=0.0428,
                eps_s=12.5, eps_inf=9.6, mu300=4.0e3),
    "GaAs": dict(Eg0=1.519, a=5.405e-4, b=204.0, m_e=0.067, m_Ll=1.9, m_Lt=0.0754, dGL=0.296, aL=6.05e-4, m_h=0.53, hwlo=0.0365,
                 eps_s=12.9, eps_inf=10.9, mu300=7.0e3),
    "AlGaAs30": dict(Eg0=1.937, a=6.438e-4, b=301.8, m_e=0.092, m_Ll=1.73, m_Lt=0.098, dGL=0.06, aL=6.05e-4, m_h=0.60, hwlo=0.0365,
                     eps_s=12.0, eps_inf=10.1, mu300=1.5e3),
}
# L valleys: longitudinal / transverse masses (m_Ll, m_Lt), linearly interpolated for the alloys; DOS mass per valley
# (m_Ll m_Lt^2)^(1/3); Gamma-L separation dGL at 0 K, with the L gap following its own Varshni slope aL (same b):
# dGL(T) = dGL - (aL - a) T^2 / (T + b).


def mL_dos(mat):
    p = MATT[mat]
    return (p["m_Ll"] * p["m_Lt"] ** 2) ** (1 / 3)


def mL_z(mat):
    """Confinement mass of an L valley for growth along [001]: 1/m_z = (1/3)/m_l + (2/3)/m_t."""
    p = MATT[mat]
    return 1.0 / (1 / (3 * p["m_Ll"]) + 2 / (3 * p["m_Lt"]))


def dGL(mat, T):
    p = MATT[mat]
    return p["dGL"] - (p["aL"] - p["a"]) * T ** 2 / (T + p["b"])


def gap(mat, T):
    """Varshni band gap (eV) of material `mat` at temperature T (K)."""
    p = MATT[mat]
    return p["Eg0"] - p["a"] * T ** 2 / (T + p["b"])


def Neff(m_dos, T):
    """Effective density of states (cm^-3) for a parabolic band of DOS mass m_dos (m0) at T."""
    return 2.0 * (m_dos * M0 * KB * T * QE / (2 * np.pi * HBAR ** 2)) ** 1.5 * 1e-6


def intrinsic(mat, T):
    """Intrinsic carrier density (cm^-3), non-degenerate, Gamma + L conduction states."""
    p = MATT[mat]
    Nc = Neff(p["m_e"], T) + 4 * Neff(mL_dos(mat), T) * np.exp(-dGL(mat, T) / (KB * T))
    Nv = Neff(p["m_h"], T)
    return np.sqrt(Nc * Nv) * np.exp(-gap(mat, T) / (2 * KB * T))


def L_fraction(mat, T, dE=0.0):
    """Fraction of conduction electrons in the L valleys of a non-degenerate bulk layer whose Gamma states start dE
    (eV) above the Gamma band edge (e.g. a quantum-well ground state): L valleys sit dGL - dE above them."""
    p = MATT[mat]
    NG = Neff(p["m_e"], T); NL = 4 * Neff(mL_dos(mat), T) * np.exp(-(dGL(mat, T) - dE) / (KB * T))
    return NL / (NG + NL)


def kane_gap(Eg_T):
    """The nonparabolicity energy of the one-band Kane model follows the band gap."""
    return Eg_T


# ---------------------------------------------------------------------------------------------------------------
# ISB absorption with thermal populations and in-plane nonparabolicity
# ---------------------------------------------------------------------------------------------------------------
def inplane_masses(z, psi, E, V, m_edge, Eg):
    """In-plane masses of the subbands at k = 0: <m(z, E_n)> weighted by |psi_n|^2 (Kane model)."""
    dz = z[1] - z[0]
    out = []
    for n in range(len(E)):
        mz = m_edge * (1.0 + (E[n] - V) / Eg)
        out.append(np.sum(psi[:, n] ** 2 * mz) * dz / np.sum(psi[:, n] ** 2 * dz))
    return np.array(out)


def isb_alpha(hw, E, Z, rho_k, m_par, ns, L_period, T, gamma, Et=None, nref=3.3, nk=200):
    """Material ISB absorption spectrum alpha(hw) (1/cm) of one period, all transitions l < m.
    E: subband energies (eV); Z: dipoles (A); m_par: in-plane masses (m0); ns: sheet density in the bound subbands
    (cm^-2); rho_k: fractional populations (only used if T == 0); L_period (A); gamma: homogeneous HWHM (eV), scalar
    or per-transition matrix; Et: depolarization-shifted transition energies Et[m, l] (k = 0); defaults to E[m] - E[l].
    In-plane nonparabolicity: E_m(k) = E_m + hbar^2 k^2 / 2 m_m, so the transition energy at k is
    Et[m, l] + hbar^2 k^2/2 (1/m_m - 1/m_l); the absorption sums Lorentzians over k weighted by f_l(k) - f_m(k)
    (Fermi-Dirac with a common Fermi level)."""
    hw = np.atleast_1d(np.asarray(hw, float))
    n = len(E)
    if Et is None:
        Et = E[:, None] - E[None, :]
    G = np.broadcast_to(np.asarray(gamma, float), (n, n)) if np.ndim(gamma) == 2 else np.full((n, n), float(gamma))
    kT = KB * T
    # Fermi level for the bound subbands, with their own in-plane masses
    from scipy.optimize import brentq
    D = m_par * M0 / (np.pi * HBAR ** 2) * QE * 1e-4                       # states / (eV cm^2), per subband
    mu = brentq(lambda mu: np.sum(D * kT * np.logaddexp(0, (mu - E) / kT)) - ns, E[0] - 3.0, E[-1] + 3.0)
    # kinetic-energy grid of subband l: x = hbar^2 k^2 / 2 m_l, up to mu - E_l + 12 kT (or 12 kT)
    N3 = ns / (L_period * 1e-8)                                              # cm^-3
    pref = hw * QE / HBAR / (nref * C0 * EPS0) * QE ** 2 * 1e-20             # SI, times dipole^2 (A^2 -> m^2)
    alpha = np.zeros_like(hw)
    for l in range(n):
        xmax = max(mu - E[l], 0.0) + 14 * kT
        x = np.linspace(0, xmax, nk)
        wts = np.full(nk, x[1] - x[0]); wts[0] *= 0.5; wts[-1] *= 0.5     # trapezoid rule
        for m in range(l + 1, n):
            if Z[l, m] == 0:
                continue
            fl = 1.0 / (1.0 + np.exp((E[l] + x - mu) / kT))
            xm = x * m_par[l] / m_par[m]                                    # same k in subband m
            fm = 1.0 / (1.0 + np.exp((E[m] + xm - mu) / kT))
            dens = D[l] * wts * (fl - fm)                                    # cm^-2 per k-shell (2D DOS of subband l)
            et = Et[m, l] + x * (m_par[l] / m_par[m] - 1.0)
            lor = G[m, l] / ((et[None, :] - hw[:, None]) ** 2 + G[m, l] ** 2) / QE  # 1/J
            s = (lor * dens[None, :]).sum(axis=1) / (L_period * 1e-8) * 1e6  # m^-3 / J
            alpha += pref * (Z[l, m]) ** 2 * s * QE                          # 1/m (QE: Lorentzian in 1/eV -> 1/J)
    return alpha / 100.0 / QE                                                # 1/cm


def drude_alpha(hw, N_cm3, m_eff, mobility_cm2Vs, nref=3.3):
    """Free-carrier absorption coefficient (1/cm) of carriers of density N, mass m_eff, mobility mu (Drude)."""
    hw = np.asarray(hw, float)
    w = hw * QE / HBAR
    tau = mobility_cm2Vs * 1e-4 * m_eff * M0 / QE
    sig0 = N_cm3 * 1e6 * QE ** 2 * tau / (m_eff * M0)
    sig = sig0 / (1 + (w * tau) ** 2)
    return sig / (nref * C0 * EPS0) / 100.0


def mobility(mat, T, mu300=None):
    """Phonon-limited mobility scaling mu ~ T^-1.5 from its 300 K value (cm^2/Vs)."""
    mu300 = MATT[mat]["mu300"] if mu300 is None else mu300
    return mu300 * (T / 300.0) ** -1.5


# ---------------------------------------------------------------------------------------------------------------
# Linewidth
# ---------------------------------------------------------------------------------------------------------------
def linewidth(z, psi, E, T, gamma0, pair=(0, 1), **kw):
    """Homogeneous HWHM (eV) of transition pair = (l, m): gamma0 + (hbar/2)(W_l + W_m), with W_i the total
    LO-phonon out-scattering rate of subband i at k = 0 (all final subbands, emission and absorption)."""
    from . import scattering
    R = scattering.rate_matrix(z, psi, E, T=T, **kw)
    l, m = pair
    W = R.sum(axis=1)
    # intrasubband LO absorption (k = 0 electron absorbing a phonon within its own subband)
    for i in (l, m):
        W[i] += scattering.lo_rate(z, psi[:, i], psi[:, i], 0.0, T=T, **kw) if T > 0 else 0.0
    return gamma0 + 0.5 * HBAR * (W[l] + W[m]) / QE


# ---------------------------------------------------------------------------------------------------------------
# Resonator and detailed balance
# ---------------------------------------------------------------------------------------------------------------
def cmt_absorptance(hw, hw0, g_rad, g_isb, g_bg=0.0):
    """Absorptance (= emissivity) of one resonant mode at hw0 (eV) coupled to free space at rate g_rad (eV, HWHM
    units) with internal absorption rates g_isb(hw) and g_bg(hw) (eV): A = 4 g_rad g_abs / ((hw-hw0)^2 + g_tot^2)."""
    ga = np.asarray(g_isb) + np.asarray(g_bg)
    return 4 * g_rad * ga / ((np.asarray(hw) - hw0) ** 2 + (g_rad + ga) ** 2)


def alpha_to_rate(alpha_cm, confinement, ng=3.3):
    """Mode absorption rate (eV, HWHM convention): g = hbar * (c / n_g) * Gamma * alpha / 2."""
    return HBAR * C0 / ng * confinement * np.asarray(alpha_cm) * 100.0 / 2.0 / QE


def planck_flux(hw, T):
    """Hemispherical spectral photon flux of a blackbody, photons / (s m^2 eV)."""
    hw = np.asarray(hw, float)
    return 2 * np.pi / (H ** 3 * C0 ** 2) * (hw * QE) ** 2 / np.expm1(hw / (KB * T)) * QE


def tpv(hw, emiss, Te, Tc=300.0, Eg=0.3, eqe=1.0, ere=1.0, sub_reflect=0.0):
    """Detailed balance of a planar emitter (spectral hemispherical emissivity emiss(hw), temperature Te) facing a
    planar cell (gap Eg, temperature Tc, back mirror), view factor 1, far field, with multiple reflections between
    the two surfaces. The cell absorbs all above-gap photons and a fraction 1 - R of sub-gap photons
    (R = sub_reflect); the exchange factors are F_a = eps and F_b = eps (1 - R) / (1 - (1 - eps) R).
      J(V) = q EQE int_above F_a Phi_bb(Te) - (q / ERE) int_above F_a Phi_bb(Tc) (exp(qV/kTc) - 1),
    with the dark term the radiative recombination flux exchanged with the emitter, enhanced by 1/ERE for
    nonradiative recombination. Heat drawn from the emitter: Q = int_above F_a [Phi(Te) - Phi(Tc) e^{qV/kTc}] hw
    + int_below F_b [Phi(Te) - Phi(Tc)] hw. Returns P_max (W/m^2), V_mp, J_mp, Jsc, Q_in (W/m^2) at the maximum
    power point, eta = P_max / Q_in, and the above- and below-gap radiant powers absorbed by the cell."""
    from scipy.optimize import minimize_scalar
    hw = np.asarray(hw, float); e = np.clip(np.asarray(emiss, float), 0, 1)
    dhw = np.gradient(hw)
    above = hw >= Eg
    R = sub_reflect
    Fa = e
    Fb = e * (1 - R) / np.maximum(1 - (1 - e) * R, 1e-300)
    pe = planck_flux(hw, Te); pc = planck_flux(hw, Tc)
    kTc = KB * Tc
    Jsc = QE * eqe * np.sum((Fa * pe * dhw)[above])
    J0 = QE * np.sum((Fa * pc * dhw)[above]) / ere
    if Jsc <= 0 or J0 <= 0:
        return dict(P_max=0.0, V_mp=0.0, J_mp=0.0, Jsc=0.0, Q_in=0.0, eta=0.0, P_above=0.0, P_below=0.0)
    Voc = kTc * np.log1p(Jsc / J0)
    Pf = lambda V: -(Jsc - J0 * np.expm1(V / kTc)) * V
    r = minimize_scalar(Pf, bounds=(0.0, Voc), method="bounded", options=dict(xatol=1e-9 * max(Voc, 1e-12)))
    V = float(r.x); J = Jsc - J0 * np.expm1(V / kTc); P = J * V
    Pa = np.sum((Fa * (pe - pc * np.exp(V / kTc)) * hw * QE * dhw)[above])
    Pb = np.sum((Fb * (pe - pc) * hw * QE * dhw)[~above])
    Q = Pa + Pb
    return dict(P_max=float(P), V_mp=V, J_mp=float(J), Jsc=float(Jsc), Q_in=float(Q), eta=float(P / Q) if Q > 0 else 0.0,
                P_above=float(Pa), P_below=float(Pb))


# ---------------------------------------------------------------------------------------------------------------
# Self-consistent emitter state at temperature T
# ---------------------------------------------------------------------------------------------------------------
class _gaps_at:
    """Context manager: temporarily set the Kane gaps of qwkit materials to their Varshni values at T.
    `mapping` maps qwkit material names to MATT names."""

    def __init__(self, T, mapping):
        self.T, self.mapping = T, mapping

    def __enter__(self):
        from . import core as q
        self.saved = {k: q.MAT[k]["Eg"] for k in self.mapping}
        for k, v in self.mapping.items():
            q.MAT[k]["Eg"] = gap(v, self.T)

    def __exit__(self, *a):
        from . import core as q
        for k, v in self.saved.items():
            q.MAT[k]["Eg"] = v


def emitter_state(layers, ns, T, gamma0=0.005, mapping=None, nstates=22, spacer=200.0, F=0.0, hw=None, eps_depol=13.5,
                  lo_kw=None, valleys=True, iters=3, pad_material="barrier", Tgap=None, m_dos=None, L_model="bulk"):
    """Self-consistent period at temperature T with a box basis that includes continuum (delocalized) states, so
    thermal escape over the barriers is counted, and (valleys=True) a bulk-like L-valley reservoir in the well
    material: n_L = W_wells * 4 N_L(T) F(mu - E_L), E_L = Gamma edge of the well + dGL, non-degenerate. The Fermi level
    is common to Gamma subbands, continuum states and L valleys; the Gamma part of the charge is re-solved
    self-consistently (`iters` outer iterations; the L electrons, which sit in the wells, are kept in the Hartree
    density with the Gamma ground-state profile). Returns bound subbands (Loewdin-orthonormal), their populations,
    sheet densities in bound, continuum and L states, Ando energies, in-plane masses, the homogeneous linewidth of the
    1-2 transition and, if hw is given, the ISB absorption spectrum alpha(hw) (1/cm, averaged over the period).
    Tgap: temperature used for the band gaps (Kane nonparabolicity), default T (for decomposition studies).
    m_dos: 2D density-of-states mass of the Gamma subbands (default: band-edge mass of the well material).
    L_model: "bulk" (L reservoir at the bulk L edge of the well material), "confined" (raised by the L ground-state
    confinement energy hbar^2 pi^2 / 2 m_z (W + 10 A)^2) or "bulk+barrier" (also the L states of the barrier and
    spacer material, at the barrier Gamma offset + its Gamma-L separation)."""
    from . import core as q, sp
    from scipy.optimize import brentq
    mapping = mapping or {"well": "InGaAs", "barrier": "InAlAs"}
    wkey = layers[0][1] if layers[0][1] in mapping else "well"
    wmat = mapping.get(wkey, "InGaAs")
    bmat = mapping.get(pad_material, "InAlAs")
    Wwell = sum(t for t, k in layers if k == wkey)
    Wbar = sum(t for t, k in layers if k != wkey) + spacer
    m_dos = MATT[wmat]["m_e"] if m_dos is None else m_dos
    nG = ns; VH0 = None
    kT = KB * T
    D = m_dos * M0 / (np.pi * HBAR ** 2) * QE * 1e-4
    for it in range(iters if valleys else 1):
        with _gaps_at(T if Tgap is None else Tgap, mapping):
            r = sp.self_consistent(layers, F, ns_cm2=nG, T=T, nstates=nstates, tol=1e-6, maxit=400, spacer=spacer, VH0=VH0,
                                   pad_material=pad_material, m_dos=m_dos)
        VH0 = r["VH"]
        z, psi, E, act = r["z"], r["psi"], r["E"], r["active"]
        V0 = r["Vext"] + F * 1e-5 * (z - 0.5 * (act[0] + act[1]))           # band-edge profile without the field
        inw = (z > act[0]) & (z < act[1]) & (V0 < V0.min() + 1e-6)
        EG = float(np.min(r["V"][inw])) if inw.any() else float(np.min(r["V"]))
        if not valleys:
            nL = 0.0; break
        EL = EG + dGL(wmat, T)
        if L_model == "confined":
            EL += 3.80998 * np.pi ** 2 / (mL_z(wmat) * (Wwell / max(1, sum(1 for t, k in layers if k == wkey)) + 10.0) ** 2)
        NL2 = Wwell * 1e-8 * 4 * Neff(mL_dos(wmat), T)                         # cm^-2 per unit exp
        terms = [(NL2, EL)]
        if L_model == "bulk+barrier":
            from . import core as qq
            ELb = EG + qq.MAT[pad_material]["V"] + dGL(bmat, T)
            terms.append((Wbar * 1e-8 * 4 * Neff(mL_dos(bmat), T), ELb))
        f = lambda mu: np.sum(D * kT * np.logaddexp(0, (mu - E) / kT)) + sum(N * np.exp((mu - e) / kT) for N, e in terms) - ns
        mu = brentq(f, E[0] - 3.0, E[-1] + 3.0)
        nL = sum(N * np.exp((mu - e) / kT) for N, e in terms)
        if abs((ns - nL) - nG) < 1e-4 * ns:
            break
        nG = ns - nL
    rho = q.populations(E, nG, T, m_eff=m_dos)
    loc = q.localized(z, psi, act)
    keep = np.where(loc)[0]
    nb = nG * rho[keep].sum()
    Eb = E[keep]
    dz = z[1] - z[0]
    # Loewdin over the whole box basis (bound and continuum states), then keep the bound ones (core.lowdin_localized)
    pb = q.orthonormalize(z, psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz))[:, keep]
    zc = 0.5 * (act[0] + act[1]); Z = q.dipoles(z - zc, pb)
    rb = rho[keep] / rho[keep].sum()
    Et = sp.depol_energies(z, pb, Eb, rb, nb, eps_r=eps_depol)
    mpar = inplane_masses(z, pb, Eb, r["V"], r["m"], r["Eg"])               # r["Eg"] holds the gaps at T
    L = len(z) * dz
    gam = linewidth(z, pb, Eb, T, gamma0, pair=(0, 1), **(lo_kw or {})) if len(Eb) > 1 else gamma0
    out = dict(T=T, z=z, V=r["V"], E=Eb, psi=pb, Z=Z, rho=rb, n_bound=nb, n_cont=nG - nb, n_L=nL, Et=Et, m_par=mpar, L=L,
               gamma=gam, E1_above_edge=float(Eb[0] - EG), E_gamma_edge=EG, r=r)
    if hw is not None:
        out["alpha"] = isb_alpha(hw, Eb, Z, rb, mpar, nb, L, T, gam, Et=Et)
    return out


def leakage(T, mat, width_A, tau_g=1e-9, V=10.0, shift=0.0):
    """Generation (SRH) leakage of a reverse-biased layer of total thickness width_A (A) of material `mat`:
    J = q n_i W / tau_g (A/cm^2; Sze & Ng, tau_g the generation lifetime) and the dissipated power density J V
    (W/cm^2). shift: confinement energies (E1e + E1h) of quantum wells, which raise the effective gap:
    n_i -> n_i exp(-shift / 2kT)."""
    J = QE * intrinsic(mat, T) * np.exp(-shift / (2 * KB * T)) * width_A * 1e-8 / tau_g
    return J, J * V


# ---------------------------------------------------------------------------------------------------------------
# Background absorption and the emitter spectrum
# ---------------------------------------------------------------------------------------------------------------
A_IB = {"InGaAs": 3.0e4, "InAlAs": 3.0e4, "InP": 3.0e4, "GaAs": 3.5e4, "AlGaAs30": 3.5e4}   # 1/cm/sqrt(eV)


def interband_alpha(hw, mat, T, urbach=None, shift=0.0):
    """Direct-gap interband absorption of bulk `mat` at T: A sqrt(hw - Eg(T)) above the gap (A ~ 3e4 1/cm/sqrt(eV),
    which gives ~7.5e3 1/cm 65 meV above the gap of In0.53Ga0.47As at 300 K), with an exponential Urbach tail below
    it, joined continuously at Eg + E_U/2. Urbach energy (default): phonon-broadened form
    E_U(T) = E_U(300 K) coth(hw_LO / 2kT) / coth(hw_LO / 600 k), E_U(300 K) = 8 meV."""
    hw = np.asarray(hw, float)
    Eg = gap(mat, T) + shift; A = A_IB[mat]                    # shift: confinement energies in a quantum well
    hl = MATT[mat]["hwlo"]
    EU = 0.008 / np.tanh(hl / (2 * KB * T)) * np.tanh(hl / (2 * KB * 300.0)) if urbach is None else urbach
    x0 = Eg + EU / 2
    above = A * np.sqrt(np.clip(hw - Eg, 0, None))
    a0 = A * np.sqrt(EU / 2)
    tail = a0 * np.exp((hw - x0) / EU)
    return np.where(hw >= x0, above, tail)


def emitter_spectrum(hw, alpha_isb, T, hw0, Q_rad, confinement=0.3, ng=3.3, Q_other=np.inf, alpha_bg_mode=0.0,
                     slab=None, f_omega=1.0):
    """Hemispherical spectral emissivity of a single-mode ISB emitter (Kirchhoff), as the sum of
      the resonant channel: f_omega * 4 g_r g_i / ((hw - hw0)^2 + (g_r + g_i)^2), g_i = g_isb(hw) + g_bg(hw) + g_o,
        g_r = hw0 / (2 Q_rad), g_isb = mode ISB absorption rate from alpha_isb (period-averaged material alpha, 1/cm)
        with the given confinement factor, g_bg from alpha_bg_mode (1/cm, modal background), g_o = hw0/(2 Q_other)
        (e.g. metal or scattering loss); f_omega = share of the hemispherical etendue the resonance covers;
      and the non-resonant slab channel: 1 - exp(-sum_k alpha_k d_k) for slab = list of (alpha_k(hw) 1/cm, d_k cm)
        (effective double-pass thicknesses already included), combined so that emissivity <= 1.
    Returns (emissivity, resonant part, slab part)."""
    hw = np.asarray(hw, float)
    g_isb = alpha_to_rate(alpha_isb, confinement, ng)
    g_bg = alpha_to_rate(alpha_bg_mode, 1.0, ng)
    g_o = hw0 / (2 * Q_other)
    g_r = hw0 / (2 * Q_rad)
    gi = g_isb + g_bg + g_o
    res = f_omega * 4 * g_r * gi / ((hw - hw0) ** 2 + (g_r + gi) ** 2)
    sl = np.zeros_like(hw)
    if slab:
        sl = 1 - np.exp(-sum(np.asarray(a) * d for a, d in slab))
    return 1 - (1 - res) * (1 - sl), res, sl
