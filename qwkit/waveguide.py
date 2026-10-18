"""
waveguide.py -- TM modes of a planar multilayer waveguide (field H_y, propagation along x, layers along z).

TM wave equation for H_y with complex permittivity eps(z):
    eps d/dz( (1/eps) dH/dz ) + k0^2 eps H = beta^2 H
solved by finite differences on a uniform grid with H = 0 far outside (thick terminal layers).
E_z = beta H / (omega eps0 eps),  E_x = (i / (omega eps0 eps)) dH/dz.
n_eff = beta / k0 (complex); modal power loss alpha = 2 k0 Im(n_eff).

Optical constants (approximate, mid-infrared; state them in any paper):
  * InP: Pettit & Turner Sellmeier (valid 0.95-10 um).
  * In0.53Ga0.47As and In0.52Al0.48As: n = 3.43 and 3.20 at 10 um, with the same dispersion as InP
    (n(lambda) - n(10 um) taken from InP). This sets the coherence length only approximately; the
    electrode period is a lithographic design parameter that is tuned to the measured value.
  * Free carriers: Drude, eps = eps_inf - wp^2 / (w (w + i/tau)).
  * Gold: Drude, Ordal et al. (1985): plasma 7.28e4 cm^-1, damping 215 cm^-1.
"""
import numpy as np
from scipy.linalg import eig

C0 = 2.99792458e8; EPS0 = 8.8541878128e-12; QE = 1.602176634e-19; M0 = 9.1093837e-31


def n_inp(lam_um):
    """Refractive index of InP (Pettit & Turner Sellmeier), wavelength in um."""
    l2 = lam_um ** 2
    return np.sqrt(7.255 + 2.316 * l2 / (l2 - 0.6263 ** 2) + 2.765 * l2 / (l2 - 32.935 ** 2))


def n_alloy(name, lam_um):
    """Index of "InGaAs" or "InAlAs" (lattice-matched to InP): value at 10 um plus the InP dispersion."""
    base = {"InGaAs": 3.43, "InAlAs": 3.20}[name]
    return base + (n_inp(lam_um) - n_inp(10.0))


def drude(eps_inf, N_cm3, m_eff, mobility_cm2Vs, lam_um):
    """Complex permittivity with a Drude free-carrier term: density N (cm^-3), mass m_eff (m0), mobility
    (cm^2/Vs), wavelength (um)."""
    w = 2 * np.pi * C0 / (lam_um * 1e-6)
    if N_cm3 <= 0:
        return eps_inf + 0j
    N = N_cm3 * 1e6; m = m_eff * M0
    wp2 = N * QE ** 2 / (m * EPS0)
    tau = mobility_cm2Vs * 1e-4 * m / QE
    return eps_inf - wp2 / (w * (w + 1j / tau))


def eps_gold(lam_um):
    """Complex permittivity of gold from Ordal's Drude parameters, wavelength in um."""
    wn = 1e4 / lam_um                      # cm^-1
    wp, wt = 7.28e4, 215.0                 # Ordal et al., Appl. Opt. 24, 4493 (1985)
    return 1 - wp ** 2 / (wn ** 2 + 1j * wn * wt)


def layer_eps(layer, lam_um):
    """layer: dict(material=..., t_um=..., N=doping cm^-3 (optional), frac_well=(stack only))."""
    mat = layer["material"]; N = layer.get("N", 0.0)
    if mat == "Au":
        return eps_gold(lam_um)
    if mat == "InP":
        return drude(n_inp(lam_um) ** 2, N, 0.08, layer.get("mu", 2500.0), lam_um)
    if mat == "InGaAs":
        return drude(n_alloy("InGaAs", lam_um) ** 2, N, 0.043, layer.get("mu", 6000.0), lam_um)
    if mat == "InAlAs":
        return drude(n_alloy("InAlAs", lam_um) ** 2, N, 0.075, layer.get("mu", 2000.0), lam_um)
    if mat == "stack":
        f = layer["frac_well"]
        e = f * n_alloy("InGaAs", lam_um) ** 2 + (1 - f) * n_alloy("InAlAs", lam_um) ** 2
        return drude(e, N, 0.05, layer.get("mu", 2000.0), lam_um) + layer.get("deps_isb", 0.0)
    if mat == "air":
        return 1.0 + 0j
    raise ValueError(mat)


def tm_modes(layers, lam_um, dz_um=0.01, nmodes=3):
    """Return list of modes (n_eff, z, Hy, Ez, Ex) sorted by Re(n_eff) descending, guided only."""
    k0 = 2 * np.pi / (lam_um * 1e-6)
    zs, eps = [], []
    z0 = 0.0
    for L in layers:
        n = max(int(round(L["t_um"] / dz_um)), 1)
        zz = z0 + (np.arange(n) + 0.5) * dz_um
        zs.append(zz); eps.append(np.full(n, layer_eps(L, lam_um), dtype=complex))
        z0 += n * dz_um
    z = np.concatenate(zs); e = np.concatenate(eps)
    h = dz_um * 1e-6
    N = len(z)
    eh = 0.5 * (e[:-1] + e[1:])                      # eps at half points
    A = np.zeros((N, N), dtype=complex)
    for i in range(N):
        A[i, i] = k0 ** 2 * e[i]
        if i > 0:
            A[i, i - 1] = e[i] / eh[i - 1] / h ** 2
            A[i, i] -= e[i] / eh[i - 1] / h ** 2
        if i < N - 1:
            A[i, i + 1] = e[i] / eh[i] / h ** 2
            A[i, i] -= e[i] / eh[i] / h ** 2
    w, v = eig(A)
    neff = np.sqrt(w.astype(complex)) / k0
    # guided TM modes: Re(neff) between cladding and core indices; reject plasmon-like modes at metal
    out = []
    eps_r = e.real
    nmax = np.sqrt(eps_r[eps_r > 0].max())
    order = np.argsort(-neff.real)
    for j in order:
        ne = neff[j]
        if not (1.0 < ne.real < nmax + 0.05):
            continue
        H = v[:, j]
        # localisation: field must decay at both ends of the grid
        if abs(H[0]) > 0.05 * np.abs(H).max() or abs(H[-1]) > 0.05 * np.abs(H).max():
            continue
        Ez = ne * H / e                                  # up to constant factor beta/(w eps0)
        dH = np.gradient(H, h)
        Ex = 1j * dH / (k0 * e)
        # normalise so that int |Ez|^2 + |Ex|^2 dz = 1 (z in um)
        nrm = np.sqrt(np.sum(np.abs(Ez) ** 2 + np.abs(Ex) ** 2) * dz_um)
        out.append(dict(neff=ne, z=z, H=H / nrm, Ez=Ez / nrm, Ex=Ex / nrm, eps=e,
                        alpha_cm=2 * k0 * ne.imag / 100.0))
        if len(out) >= nmodes:
            break
    return out


def overlap(mode, z0_um, z1_um):
    """Fraction of |E|^2 that is |Ez|^2 inside [z0, z1] -- confinement factor for ISB absorption."""
    z = mode["z"]; dz = z[1] - z[0]
    sel = (z >= z0_um) & (z < z1_um)
    return float(np.sum(np.abs(mode["Ez"][sel]) ** 2) * dz)


def shg_overlap(m1, m2, z0_um, z1_um):
    """O = | int_QW f1^2 f2* dz |^2 in um^-1 with f = Ez normalised to int(|E|^2)=1 (um^-1/2 each)."""
    z = m1["z"]; dz = z[1] - z[0]
    sel = (z >= z0_um) & (z < z1_um)
    return float(np.abs(np.sum(m1["Ez"][sel] ** 2 * np.conj(m2["Ez"][sel])) * dz) ** 2)
