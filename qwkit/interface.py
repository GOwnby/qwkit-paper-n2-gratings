"""
interface.py -- self-consistent two-dimensional electron gas at a single polar heterointerface (MAT-01).

Geometry (z in Angstrom): a barrier cap of thickness d_cap on -d_cap < z < 0 (e.g. LaInO3, under an electric-double-
layer gate) and the well material on 0 < z < L (e.g. BaSnO3), hard walls at both ends. Charges:
  * electrons n(z) = sum_n n_n |psi_n|^2 in the subbands;
  * a positive sheet sigma_+ (polarization bound charge and gate charge) split between the interface (fraction f_int)
    and the cap surface (1 - f_int); it only changes the field inside the cap;
  * ionized acceptors N_A over 0 < z < w_d (depletion layer); w_d follows from Fermi-level pinning: the conduction band
    deep in the well material lies E_pin above the Fermi level. Neutrality: sigma_+ = n_s + N_A w_d.
The Hartree potential energy V_H(z) = int D/(eps eps0) dz with D the displacement accumulated from the cap side.
Optional local exchange-correlation (Hedin-Lundqvist) with the static permittivity and band-edge mass of the well.

Subbands: one-band Kane Schroedinger equation with energy-dependent mass m(z, E) = m_edge (1 + (E - V)/Eg) (core.
solve_states; Eg is the nonparabolicity energy 1/beta). In-plane dispersion of subband n, consistent with the same model:
    hbar^2 k^2 / (2 m_n) = eps (1 + eps / G_n),   m_n = <m(z, E_n)>_n,   G_n = m_n / <m_edge / Eg>_n,
so the subband density of states is (m_n / pi hbar^2)(1 + 2 eps / G_n). Fermi-Dirac occupation at temperature T.
Units: Angstrom, eV, cm^-2 (sheet), cm^-3 (volume).
"""
import numpy as np
from scipy.optimize import brentq
from . import core as q

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12; HBAR = 1.054571817e-34; M0 = 9.1093837015e-31
KB = 8.617333262e-5

# MAT-01 defaults (docs/MAT01_literature.md): BaSnO3 well, LaInO3 cap.
BSO = dict(V=0.0, m=0.19, Eg=1.0 / 0.56, eps_s=20.0, eps_inf=4.0)
LIO = dict(V=1.6, m=0.46, Eg=5.0, eps_s=24.0, eps_inf=4.4)


def dos_factor(m_n):
    """2D density of states per subband at the band edge, m_n / (pi hbar^2), in states / (eV cm^2) (spin included)."""
    return m_n * M0 / (np.pi * HBAR ** 2) * QE * 1e-4


def subband_density(E_n, m_n, G_n, mu, T, ne=600):
    """Sheet density (cm^-2) of a nonparabolic subband: int D (1 + 2 eps/G) f(E_n + eps - mu) d eps."""
    kT = KB * max(T, 1e-3)
    top = max(mu - E_n, 0.0) + 40 * kT
    e = np.linspace(0.0, top, ne)
    f = 0.5 * (1.0 - np.tanh((E_n + e - mu) / (2 * kT)))
    g = (1.0 + 2.0 * e / G_n) * f
    return dos_factor(m_n) * float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(e)))


def _mass(m_edge, Eg, E, V):
    return np.clip(m_edge * (1.0 + (E - V) / Eg), 0.2 * m_edge, None)


def bandedge_masses(z, psi, E, V, m_edge, Eg):
    """Exact band-edge in-plane masses (m0) of the subbands of the energy-dependent-mass Hamiltonian
    H(E) = -d/dz hbar^2/(2 m(z,E)) d/dz + V + hbar^2 k^2/(2 m(z,E)): first-order perturbation in k^2 with the
    energy dependence of H, 1/m_par = <1/m(z,E_n)> / (1 - <dH/dE>), <dH/dE> = -(hbar^2/2) int (dm/dE)/m^2 |psi'|^2.
    For a flat well this gives m*(1 + 2 beta T) (T: kinetic energy), the Kane result."""
    dz = z[1] - z[0]
    out = []
    for n in range(len(E)):
        p = psi[:, n] / np.sqrt(np.sum(psi[:, n] ** 2) * dz)
        m = _mass(m_edge, Eg, E[n], V)
        inv = np.sum(p ** 2 / m) * dz
        mh = 0.5 * (m[1:] + m[:-1]); dmh = 0.5 * ((m_edge / Eg)[1:] + (m_edge / Eg)[:-1])
        dp = np.diff(p) / dz
        corr = q.HB2M0 * np.sum(dmh / mh ** 2 * dp ** 2) * dz
        out.append((1.0 + corr) / inv)
    return np.array(out)


def subband_dispersion(z, V, m_edge, Eg, n, E_n, x0, iters=60, tol=1e-9):
    """Exact in-plane dispersion of subband n: energies E_n(k) - E_n for x0 = hbar^2 k^2/(2 m0) (eV), from the
    self-consistent eigenproblem with the in-plane term x0 m0/m(z,E) added to the potential."""
    out = []
    Ek = E_n
    for x in np.atleast_1d(x0):
        for _ in range(iters):
            m = _mass(m_edge, Eg, Ek, V)
            w, _v = q._solve(z, V + x / m, m, n + 1, first=n)
            if abs(w[0] - Ek) < tol:
                Ek = w[0]; break
            Ek = 0.5 * Ek + 0.5 * w[0]
        out.append(Ek - E_n)
    return np.array(out)


def inplane_params(z, psi, E, V, m_edge, Eg, emax=None, npts=3):
    """In-plane band-edge masses m_n (m0, exact) and nonparabolicity energies G_n (eV) of the subbands, with
    hbar^2 k^2/(2 m_n) = eps (1 + eps/G_n) fitted to the exact dispersion for eps up to emax (if emax is None,
    the cruder estimate G_n = m_n / <m_edge/Eg>_n is returned)."""
    dz = z[1] - z[0]
    m_n = bandedge_masses(z, psi, E, V, m_edge, Eg)
    G_n = []
    for n in range(len(E)):
        w = psi[:, n] ** 2; w = w / (np.sum(w) * dz)
        G0 = m_n[n] / (np.sum(w * m_edge / Eg) * dz)
        if emax is None:
            G_n.append(G0); continue
        eps_t = emax * np.arange(1, npts + 1) / npts
        x0 = m_n[n] * eps_t * (1 + eps_t / G0)                # trial k grid
        eps = subband_dispersion(z, V, m_edge, Eg, n, E[n], x0)
        b = np.sum(eps ** 2 * (x0 - m_n[n] * eps)) / np.sum(eps ** 4)
        G_n.append(m_n[n] / b if b > 1e-6 else 1e6)
    return np.array(m_n), np.array(G_n)


def fermi_level(E, m_n, G_n, ns, T):
    return brentq(lambda mu: sum(subband_density(E[i], m_n[i], G_n[i], mu, T) for i in range(len(E))) - ns,
                  E[0] - 1.0, E[-1] + 3.0, xtol=1e-9)


def xc_potential(n_cm3, eps, m):
    """Hedin-Lundqvist local exchange-correlation potential (eV) for volume density n (cm^-3), permittivity eps,
    mass m (m0)."""
    a_star = 0.529177e-8 * eps / m                     # cm
    Ry = 13.605693 * m / eps ** 2                      # eV
    n = np.maximum(n_cm3, 1e10)
    rs = (3.0 / (4 * np.pi * n)) ** (1 / 3) / a_star
    alpha = (4.0 / (9 * np.pi)) ** (1 / 3)
    x = rs / 21.0
    return -(2.0 / (np.pi * alpha * rs)) * (1.0 + 0.7734 * x * np.log(1.0 + 1.0 / x)) * Ry


def build(d_cap=50.0, L=500.0, dz=0.25, well=BSO, cap=LIO):
    z = np.arange(-d_cap + dz / 2, L, dz)
    incap = z < 0
    V0 = np.where(incap, cap["V"], well["V"])
    m = np.where(incap, cap["m"], well["m"])
    Eg = np.where(incap, cap["Eg"], well["Eg"])
    eps = np.where(incap, cap["eps_s"], well["eps_s"])
    return z, V0, m, Eg, eps


def hartree(z, n_cm3, NA_cm3, w_d, sig_int, sig_cap, eps, d_cap, nd_cm3=None):
    """Electron potential energy (eV) from the electrons, the depleted acceptors, the positive sheets and an optional
    fixed positive (ionized-donor) density nd_cm3 on the grid."""
    dz = z[1] - z[0]
    # ionized acceptors over 0 < z < w_d: each cell carries the fraction of its width inside the depletion layer
    # (continuous in w_d; the earlier version was discontinuous when w_d fell in the left half of a cell, which made
    # the self-consistency oscillate at the depletion edge)
    na = np.where(z > 0, NA_cm3 * np.clip((w_d - (z - dz / 2)) / dz, 0.0, 1.0), 0.0)
    rho = -(na + n_cm3) * 1e6 * 1e-10 * dz             # negative charge per cell, m^-2 (electrons and acceptors)
    if nd_cm3 is not None:
        rho = rho + nd_cm3 * 1e6 * 1e-10 * dz           # fixed positive charge (ionized donors)
    D = np.cumsum(rho) - 0.5 * rho                      # count of charges to the left of each cell centre (m^-2)
    D += sig_cap * 1e4                                  # cap-surface sheet at z = -d_cap (left of all cells)
    D += np.where(z > 0, sig_int * 1e4, 0.0)            # interface sheet at z = 0
    E = QE * D / (eps * EPS0)                           # V/m (cell centres)
    # potential at the cell centres: integral of the piecewise-constant field up to each centre (second order; the
    # earlier cumsum(E) dz put every value half a cell to the right, which displaced the cap by F dz/2)
    V = (np.cumsum(E) - 0.5 * E) * dz * 1e-10           # eV (electron potential energy rises along +z for D > 0)
    i0 = np.searchsorted(z, 0.0)
    return V - 0.5 * (V[i0 - 1] + V[i0]) if i0 > 0 else V - V[0]     # zero at the interface


def _sheets(sig, f_int, sigma_int):
    """Interface and cap-surface positive sheets (cm^-2) for a total sig."""
    if sigma_int is None:
        return f_int * sig, (1 - f_int) * sig
    return sigma_int, sig - sigma_int


def solve(ns, T=10.0, NA=5e19, E_pin=1.0, f_int=1.0, xc=False, d_cap=50.0, L=300.0, dz=0.5, nstates=10,
          well=BSO, cap=LIO, mix=0.2, hist=6, tol=2e-5, maxit=300, verbose=False, _VH0=None, _Vxc0=None, _E0=None,
          kw_retry=True, exact_inplane=True, _G_fixed=None, xc_eps=None, sigma_int=None, nd_cap=None, rest_to="surface"):
    """Self-consistent interface 2DEG for sheet density ns (cm^-2). NA: acceptor density in the well material (cm^-3,
    0 for none); E_pin: conduction band minus Fermi level deep in the well (eV); f_int: fraction of the positive
    sheet at the interface; sigma_int (cm^-2), if given, fixes the interface sheet instead (the rest, ns + N_dep -
    sigma_int, sits on the cap surface: electric-double-layer gate charge). nd_cap: optional ionized-donor density in the
    cap, a function of z (A, z < 0; cm^-3), e.g. oxygen vacancies created by an electrolyte; it is taken as fully
    ionized and the positive sheets carry the rest, ns + N_dep - N_D (N_D its sheet density). rest_to says where the
    positive charge not fixed by sigma_int / nd_cap goes: "surface" (default: the cap surface, as above), "interface"
    (all of it at the interface; nd_cap fixed) or "donors" (nd_cap is then only the SHAPE of the donor profile, scaled
    self-consistently to carry ns + N_dep - sigma_int, so that the change of the depletion charge goes with them).
    Returns a dict with z, V (total), VH, Vxc, E, psi, mu, m_n, G_n, n_n, w_d, N_dep."""
    z, V0, m_edge, Eg, eps = build(d_cap, L, dz, well, cap)
    zc = z
    nd = None if nd_cap is None else np.where(z < 0, np.asarray(nd_cap(np.minimum(z, 0.0)), float), 0.0)
    ND = 0.0 if nd is None else float(np.sum(nd) * dz * 1e-8)       # cm^-2
    if rest_to not in ("surface", "interface", "donors"):
        raise ValueError("rest_to must be 'surface', 'interface' or 'donors'")
    if rest_to == "donors" and (nd is None or sigma_int is None or ND <= 0):
        raise ValueError("rest_to='donors' needs sigma_int and a positive nd_cap shape")

    def charges(wd):
        """(interface sheet, cap-surface sheet, donor density on the grid, donor sheet density) for depletion width wd."""
        tot = ns + NA * wd * 1e-8                       # positive charge needed in total
        if rest_to == "donors":
            dd = tot - sigma_int
            return sigma_int, 0.0, nd * (dd / ND), dd
        if rest_to == "interface":
            return tot - ND, 0.0, nd, ND
        si_, ss_ = _sheets(tot - ND, f_int, sigma_int)
        return si_, ss_, nd, ND
    VH = np.zeros_like(z); Vxc = np.zeros_like(z)
    # initial guess: triangular well of the full sheet field
    F0 = QE * ns * 1e4 / (well["eps_s"] * EPS0)         # V/m
    VH = np.where(z > 0, F0 * z * 1e-10, 0.0) if _VH0 is None else _VH0.copy()
    if _Vxc0 is not None:
        Vxc = _Vxc0.copy()
    n_cm3 = np.zeros_like(z)
    w_d = 0.0
    Eprev = _E0
    hx, hf = [], []
    mu_prev = None
    for it in range(maxit):
        V = V0 + VH + Vxc
        E, psi = q.solve_states(z, V, m_edge, Eg, nstates=nstates, E_guess=Eprev, tol=1e-7)
        Eprev = E
        psi = psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz)
        m_n, G_n = inplane_params(z, psi, E, V, m_edge, Eg)
        if _G_fixed is not None:
            G_n = _G_fixed[:len(E)]
        mu = fermi_level(E, m_n, G_n, ns, T); mu_prev = mu
        n_n = np.array([subband_density(E[i], m_n[i], G_n[i], mu, T) for i in range(len(E))])
        n_new = (psi ** 2) @ n_n * 1e8                  # cm^-3
        # depletion width from pinning: V(deep) - mu = E_pin (the deep potential grows with w_d)
        def deep(wd):
            si, ss, ndw, _ = charges(wd)
            Vt = hartree(z, n_new, NA, wd, si, ss, eps, d_cap, ndw)
            return Vt[-1] + V0[-1] - mu - E_pin
        if NA > 0:
            lo, hi = 0.0, 0.95 * L
            if deep(lo) >= 0:
                wd_new = 0.0
            elif deep(hi) < 0:
                wd_new = hi
            else:
                wd_new = brentq(deep, lo, hi, xtol=1e-3)
        else:
            wd_new = 0.0
        si, ss, ndw, ND_now = charges(wd_new)
        VH_new = hartree(z, n_new, NA, wd_new, si, ss, eps, d_cap, ndw)
        Vxc_new = xc_potential(n_new, xc_eps or well["eps_s"], well["m"]) * (z > 0) if xc else np.zeros_like(z)
        x_in = VH + Vxc; g = VH_new + Vxc_new; F = g - x_in
        err = float(np.max(np.abs(F))); F_last = F
        w_d = wd_new; n_cm3 = n_new
        if verbose:
            print(it, err, mu, E[:3])
        if err < tol:
            break
        # Anderson (Pulay) mixing of the total self-consistent potential
        hx.append(x_in.copy()); hf.append(F.copy())
        if len(hx) > hist:
            hx.pop(0); hf.pop(0)
        if len(hx) > 1:
            dF = np.array([hf[i + 1] - hf[i] for i in range(len(hf) - 1)]).T
            dX = np.array([hx[i + 1] - hx[i] for i in range(len(hx) - 1)]).T
            gam = np.linalg.lstsq(dF, F, rcond=None)[0]
            x_new = x_in + mix * F - (dX + mix * dF) @ gam
        else:
            x_new = x_in + mix * F
        Vxc = Vxc_new if xc else np.zeros_like(z)
        VH = x_new - Vxc
    else:
        if mix > 0.051 and kw_retry:
            # restart from the current potential with gentler mixing (charge sloshing near level crossings)
            return solve(ns, T=T, NA=NA, E_pin=E_pin, f_int=f_int, xc=xc, d_cap=d_cap, L=L, dz=dz, nstates=nstates,
                         well=well, cap=cap, mix=mix / 2, hist=hist, tol=tol, maxit=maxit, verbose=verbose,
                         _VH0=VH, _Vxc0=Vxc, _E0=Eprev, kw_retry=mix / 2 > 0.051, exact_inplane=exact_inplane,
                         _G_fixed=_G_fixed, xc_eps=xc_eps, sigma_int=sigma_int, nd_cap=nd_cap, rest_to=rest_to)
        import warnings
        warnings.warn(f"interface.solve: not converged ({err:.2e} eV)")
    V = V0 + VH + Vxc
    E, psi = q.solve_states(z, V, m_edge, Eg, nstates=nstates, E_guess=Eprev, tol=1e-8)
    psi = q.orthonormalize(z, psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz))
    if exact_inplane and _G_fixed is None:
        # second stage: exact in-plane nonparabolicity, held fixed while the potential is re-converged
        m_n, G_ex = inplane_params(z, psi, E, V, m_edge, Eg, emax=max(mu_prev - E[0], 0.0) + 0.3)
        return solve(ns, T=T, NA=NA, E_pin=E_pin, f_int=f_int, xc=xc, d_cap=d_cap, L=L, dz=dz, nstates=nstates,
                     well=well, cap=cap, mix=mix, hist=hist, tol=tol, maxit=maxit, verbose=verbose, _VH0=VH,
                     _Vxc0=Vxc, _E0=Eprev, kw_retry=kw_retry, exact_inplane=True, _G_fixed=G_ex, xc_eps=xc_eps,
                     sigma_int=sigma_int, nd_cap=nd_cap, rest_to=rest_to)
    if exact_inplane:
        m_n, G_n = inplane_params(z, psi, E, V, m_edge, Eg, emax=max(mu_prev - E[0], 0.0) + 0.3)
    else:
        m_n, G_n = inplane_params(z, psi, E, V, m_edge, Eg)
    mu = fermi_level(E, m_n, G_n, ns, T)
    n_n = np.array([subband_density(E[i], m_n[i], G_n[i], mu, T) for i in range(len(E))])
    return dict(z=z, V=V, V0=V0, VH=VH, Vxc=Vxc, E=E, psi=psi, mu=mu, m_n=m_n, G_n=G_n, n_n=n_n, w_d=w_d,
                N_dep=NA * w_d * 1e-8, N_D=charges(w_d)[3], sig_int=charges(w_d)[0], sig_cap=charges(w_d)[1], ns=ns, T=T, m_edge=m_edge, Eg=Eg, eps=eps, iters=it + 1, err=err, resid=F_last,
                well=well, cap=cap)


def dispersion(eps_k_par, m_n, G_n):
    """Kinetic energy eps(x) of a subband for x = hbar^2 k^2 / (2 m_n): eps (1 + eps/G) = x."""
    x = np.asarray(eps_k_par, float)
    return 0.5 * G_n * (np.sqrt(1.0 + 4.0 * x / G_n) - 1.0)


def xc_kernel(n_cm3, eps, m):
    """Adiabatic local exchange-correlation kernel f_xc = dV_xc/dn (eV m^3) of the Hedin-Lundqvist potential."""
    n = np.maximum(n_cm3, 1e12)
    h = 1e-4
    d = (xc_potential(n * (1 + h), eps, m) - xc_potential(n * (1 - h), eps, m)) / (2 * h * n)     # eV cm^3
    return d * 1e-6
