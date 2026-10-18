"""
sp.py -- self-consistent Schroedinger-Poisson (Hartree) for one period of a doped multi-quantum-well
stack under a fixed mean electric field, plus depolarisation-shifted resonant chi(2).

Electrostatics (one period, periodic):
  * The stack is held at a fixed voltage, so the period-averaged field equals the applied F.
    The Hartree field therefore has zero mean over the period (periodic boundary condition).
  * Charge: ionised donors N_D(z) (sheet density ns per period, placed by `doping`) minus electrons
    n(z) = sum_i ns*rho_i*|psi_i|^2. Each period is neutral.
  * Position-dependent permittivity (InGaAs 13.9, InAlAs 12.4); Gauss's law in D, E = D/eps, with the
    constant in D chosen so that E has zero mean over the period (D stays continuous at interfaces).
Depolarisation (Ando): E~_ml^2 = E_ml^2 * (1 + alpha_ml),
  alpha_ml = 2 e^2 ns (rho_l - rho_m) S_lm / (eps eps0 E_ml),  S_lm = int dz [int^z psi_l psi_m dz']^2.
  Applied transition by transition (no inter-transition coupling, no exciton-like term, no local-field
  correction of the driving field) -- the usual first-order treatment.
Units: Angstrom, eV inside; chi in pm/V.
"""
import numpy as np
from . import core as q

EPS_R = {"well": 13.9, "barrier": 12.4}


def doping_profile(z, active, layers, pad, ns_cm2, where="spacer", width=20.0):
    """Donor density (cm^-2 per A, i.e. sheet density per unit length) integrating to ns_cm2.
    where = 'spacer': two half-sheets of `width` at the period edges (mirror-symmetric).
            'centre': uniform over the central layer of the active region (mirror-symmetric if the
                      structure is).
            'wells' : uniform over all well layers."""
    nd = np.zeros_like(z)
    L = z[-1] + (z[1] - z[0]) / 2
    if where == "spacer":
        nd[(z < width / 2) | (z > L - width / 2)] = 1.0
    else:
        edges = np.cumsum([pad] + [t for t, _ in layers])
        starts = edges[:-1]
        kinds = [k for _, k in layers]
        if where == "centre":
            i = len(layers) // 2
            nd[(z >= starts[i]) & (z < edges[i + 1])] = 1.0
        elif where == "wells":
            for i, k in enumerate(kinds):
                if k == "well":
                    nd[(z >= starts[i]) & (z < edges[i + 1])] = 1.0
        else:
            raise ValueError(where)
    dz = z[1] - z[0]
    return nd / (nd.sum() * dz) * ns_cm2


def eps_profile(z, layers, pad, pad_material="barrier"):
    """Static relative permittivity on the grid from EPS_R (padding uses `pad_material`)."""
    eps = np.full_like(z, EPS_R[pad_material])
    edges = np.cumsum([pad] + [t for t, _ in layers])
    for i, (_, k) in enumerate(layers):
        eps[(z >= edges[i]) & (z < edges[i + 1])] = EPS_R[k]
    return eps


def hartree(z, nd, n, eps):
    """Electron potential energy (eV) from net charge e(nd - n), both in cm^-2/A.
    Periodic: zero-mean field, zero-mean potential."""
    dz = z[1] - z[0]
    rho = q.E_CH * (nd - n) * 1e4 / 1e-10          # C/m^3  (cm^-2 per A -> m^-2 per m)
    D = (np.cumsum(rho) - 0.5 * rho) * dz * 1e-10    # C/m^2, centred cumulative integral
    # Gauss's law fixes D up to a constant; choose it so that the field E = D/(eps eps0) has zero mean over
    # the period (periodic, fixed voltage). Shifting D (not E) keeps D continuous at dielectric interfaces.
    D -= np.sum(D / eps) / np.sum(1.0 / eps)
    Ef = D / (eps * q.EPS0)                          # V/m
    phi = -(np.cumsum(Ef) - 0.5 * Ef) * dz * 1e-10   # V, centred
    VH = -phi                                        # electron potential energy, eV
    return VH - VH.mean()


def self_consistent(layers, F_kVcm, ns_cm2=5e11, T=300.0, spacer=200.0, dz=0.5, doping="spacer",
                    nstates=7, mix=0.4, tol=1e-5, maxit=200, VH0=None, pad_material="barrier", m_dos=0.043):
    """Self-consistent Hartree solution for one period: layers (list of (thickness_A, kind)), applied mean
    field F (kV/cm), sheet density ns (cm^-2 per period), temperature T (K), spacer thickness (A, split
    equally on both sides, made of `pad_material`), donor placement `doping`, density-of-states mass m_dos
    (m0) for the Fermi-Dirac populations. VH0: optional starting Hartree potential (warm start).
    Returns a dict with grid z (A), total potential V and Hartree part VH (eV), energies E (eV), wavefunctions
    psi, iteration count and final change `err` (eV). Warns if not converged to `tol` within `maxit`."""
    pad = spacer / 2.0
    z, V0, m, Eg, active = q.build(layers, pad=pad, dz=dz, pad_material=pad_material)
    Vext = q.potential(z, V0, active, F_kVcm)
    nd = doping_profile(z, active, layers, pad, ns_cm2, doping)
    eps = eps_profile(z, layers, pad, pad_material)
    VH = np.zeros_like(z) if VH0 is None or len(VH0) != len(z) else VH0.copy()
    Eprev = None
    for it in range(maxit):
        E, psi = q.solve_states(z, Vext + VH, m, Eg, nstates=nstates, E_guess=Eprev, tol=1e-6)
        Eprev = E
        rho = q.populations(E, ns_cm2, T, m_eff=m_dos)
        n = (psi ** 2) @ (rho * ns_cm2)              # cm^-2 per A
        VHn = hartree(z, nd, n, eps)
        err = np.max(np.abs(VHn - VH))
        VH = (1 - mix) * VH + mix * VHn
        if err < tol:
            break
    else:
        import warnings
        warnings.warn(f"self_consistent: not converged after {maxit} iterations (change {err:.2e} eV)")
    V = Vext + VH
    E, psi = q.solve_states(z, V, m, Eg, nstates=nstates, E_guess=Eprev)
    return dict(z=z, V=V, Vext=Vext, VH=VH, E=E, psi=psi, nd=nd, active=active, layers=layers,
                iters=it + 1, err=err, ns=ns_cm2, T=T, eps=eps, m=m, Eg=Eg, F=F_kVcm, spacer=spacer, m_dos=m_dos)


def prepare(r, nkeep=6, nbuf=10):
    """Keep localised states, Loewdin-orthonormalise, dipoles about the active-region centre,
    populations at r['T']."""
    z, psi, act = r["z"], r["psi"], r["active"]
    # Loewdin over the localized states plus at least nbuf box states above them (solved at the converged potential if
    # r holds fewer), then truncate: orthonormalizing a truncated set makes its top states depend on the truncation
    # (core.lowdin_localized)
    full = all(k in r for k in ("V", "m", "Eg"))      # hand-made r without the band profile: use the states given
    E, psi, _ = q.lowdin_localized(z, psi, act, E=r["E"], V=r["V"] if full else None, m_edge=r.get("m"),
                                   Eg=r.get("Eg"), nbuf=nbuf)
    E, psi = E[:nkeep], psi[:, :nkeep]
    zc = 0.5 * (act[0] + act[1])
    Z = q.dipoles(z - zc, psi)
    rho = q.populations(E, r["ns"], r["T"], m_eff=r.get("m_dos", 0.043))
    return E, psi, Z, rho


def depol_energies(z, psi, E, rho, ns_cm2, eps_r=13.5):
    """Matrix Et[a,b] = signed depolarisation-shifted transition energy E~_ab (eV)."""
    dzA = z[1] - z[0]
    n = len(E)
    Et = E[:, None] - E[None, :]
    ns = ns_cm2 * 1e4
    for l in range(n):
        for mm in range(l + 1, n):
            Eml = E[mm] - E[l]
            dr = rho[l] - rho[mm]
            if Eml <= 0 or dr == 0:
                continue
            cum = np.cumsum(psi[:, l] * psi[:, mm]) * dzA      # dimensionless overlap integral
            S = np.sum(cum ** 2) * dzA * 1e-10                 # m
            alpha = 2 * q.E_CH ** 2 * ns * dr * S / (eps_r * q.EPS0 * Eml * q.E_CH)
            Et[mm, l] = Eml * np.sqrt(max(1 + alpha, 0.0))
            Et[l, mm] = -Et[mm, l]
    return Et


def chi2_depol(Et, Z, rho, hw, gamma, N_cm3):
    """chi2_general with an arbitrary (depolarisation-shifted) transition-energy matrix."""
    N = N_cm3 * 1e6
    n = Z.shape[0]
    tot = 0j
    for l in range(n):
        for m in range(n):
            for k in range(n):
                num = Z[l, m] * Z[m, k] * Z[k, l]
                if num == 0.0:
                    continue
                br = (rho[l] - rho[m]) / (Et[m, l] - hw - 1j * gamma) - (rho[m] - rho[k]) / (Et[k, m] - hw - 1j * gamma)
                tot += num / (Et[k, l] - 2 * hw - 1j * gamma) * br
    return N * q.E_CH ** 3 / q.EPS0 * tot * 1e-30 / q.E_CH ** 2 * 1e12


def chi2_sc(layers, F_kVcm, hw, ns_cm2=5e11, T=300.0, gamma=0.010, spacer=200.0, dz=0.5,
            doping="spacer", depol=True, sc=True, VH0=None, pad_material="barrier", m_dos=0.043, eps_depol=13.5):
    """Resonant SHG chi(2) (pm/V, complex) at photon energy hw (eV) and HWHM gamma (eV).
    sc: self-consistent Hartree potential on/off; depol: depolarisation shifts on/off (permittivity eps_depol).
    Returns dict(chi, E, Et, Z, rho, r, N) with N the period-averaged density (cm^-3) and r the raw solution."""
    if sc:
        r = self_consistent(layers, F_kVcm, ns_cm2, T, spacer, dz, doping, VH0=VH0, pad_material=pad_material,
                            m_dos=m_dos)
    else:
        pad = spacer / 2.0
        z, V0, m, Eg, active = q.build(layers, pad=pad, dz=dz, pad_material=pad_material)
        V = q.potential(z, V0, active, F_kVcm)
        E, psi = q.solve_states(z, V, m, Eg, nstates=8)
        r = dict(z=z, V=V, E=E, psi=psi, active=active, ns=ns_cm2, T=T, iters=0, err=0.0, m_dos=m_dos, m=m, Eg=Eg)
    E, psi, Z, rho = prepare(r)
    Lp = (r["active"][1] - r["active"][0] + spacer) * 1e-8
    N = ns_cm2 / Lp
    Et = depol_energies(r["z"], psi, E, rho, ns_cm2, eps_r=eps_depol) if depol else (E[:, None] - E[None, :])
    chi = chi2_depol(Et, Z, rho, hw, gamma, N)
    return dict(chi=chi, E=E, Et=Et, Z=Z, rho=rho, r=r, N=N)


def state_set(o):
    """Convert a chi2_sc result into the state-set dict used by propagation.Medium
    (lists E, Z, Et, rho and the volume density N in cm^-3)."""
    return dict(E=list(map(float, o["E"])), Z=o["Z"].tolist(), Et=o["Et"].tolist(),
                rho=list(map(float, o["rho"])), N=float(o["N"]))
