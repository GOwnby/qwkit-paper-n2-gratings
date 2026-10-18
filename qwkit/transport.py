"""
transport.py -- low-temperature multisubband transport of a two-dimensional electron gas (MAT-03).

Elastic scattering at the Fermi level by Coulomb scatterers, with static multisubband screening, and the momentum
relaxation times from the coupled Boltzmann equations of Siggia and Kwok [Phys. Rev. B 2, 1024 (1970)]; Hall density
and Hall mobility of the parallel subband channels in the Drude picture.

Scatterers (each an argument of `mobilities`; densities per area in cm^-2, volume densities in cm^-3, z in Angstrom):
  * sheets of point charges Z e at z0 (e.g. ionized donors next to an interface, charged defects at the interface);
  * a distribution of point charges N(z) (e.g. depleted acceptors), integrated over its extent;
  * charged threading dislocations: lines along z with line charge e f / c (filling f of sites spaced c), whose
    potential does not depend on z and therefore scatters only within a subband [Jena, Gossard and Mishra, Appl. Phys.
    Lett. 76, 1707 (2000)];
  * interface roughness of an abrupt offset V0 at z_I with Gaussian height correlations (rms Delta, correlation length
    Lambda): V_ij(q)^2 = (V0 psi_i(z_I) psi_j(z_I))^2 pi Delta^2 Lambda^2 exp(-q^2 Lambda^2 / 4) [finite-barrier form; Ando,
    Fowler and Stern, Rev. Mod. Phys. 54, 437 (1982)].

Matrix elements: a point charge Z e at z0 gives, for an in-plane momentum transfer q,
    V_ij(q) = Z e^2 / (2 eps0 eps q) F_ij(q, z0),   F_ij = int psi_i psi_j exp(-q |z - z0|) dz.
Screening: static random-phase approximation with intrasubband polarizabilities only (the 2D Lindhard function at
T = 0 with the density-of-states mass at the Fermi level, spin included); the induced potential acts on intra- and
intersubband matrix elements alike:
    phi_b = phi0_b - sum_c C G_{bb,cc} Pi_c phi_c   (C = e^2 / (2 eps0 eps q)),
    V_ij  = V0_ij - sum_c C G_{ij,cc} Pi_c phi_c,     G_{ij,cc}(q) = int int psi_i psi_j(z) psi_c^2(z') exp(-q|z-z'|).
Nonparabolicity enters through the Fermi wavevector k_i = sqrt(2 pi n_i) and the density-of-states (transport) mass at
the Fermi level, m_i (1 + 2 eps_F,i / G_i), from the dispersion hbar^2 k^2 / 2 m_i = eps (1 + eps / G_i) of the interface
module. Mobility of subband i: mu_i = e tau_i / m_d,i. Phonons are not included (T = 0).
Dielectric mismatch (optional, eps_cap): the 2DEG lies on the z > 0 side (permittivity eps) of a planar boundary with
a cap of permittivity eps_cap (z < 0). A charge in the cap then acts on the 2DEG with the factor 2 eps / (eps + eps_cap),
and a charge at z' > 0 has an image term r exp(-q (z + z')), r = (eps - eps_cap) / (eps + eps_cap), which also enters
the screening kernel (|z| is used for the small part of the wavefunctions inside the cap).
Coupled relaxation times (Siggia-Kwok): 1 = sum_j [tau_i / tau0_ij - tau_j (v_j / v_i) / tau1_ij], with the Fermi
velocities v_i = hbar k_i / m_d,i.
Units: SI internally; inputs in Angstrom, cm^-2, cm^-3; mobilities returned in cm^2/Vs.
"""
import numpy as np

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12; HBAR = 1.054571817e-34; M0 = 9.1093837015e-31


def _G(z, a, b, q):
    """int int a(z) b(z') exp(-q |z - z'|) dz dz' (z in m, uniform grid) by two first-order recursions, O(N)."""
    from scipy.signal import lfilter
    dz = z[1] - z[0]
    w = np.exp(-q * dz)
    left = lfilter([dz], [1.0, -w], b)                       # sum_{z' <= z} b(z') exp(-q (z - z')) dz
    right = lfilter([dz * w], [1.0, -w], b[::-1])[::-1]
    right = np.concatenate([right[1:], [0.0]])                # sum_{z' > z} b(z') exp(-q (z' - z)) dz
    return float(np.sum(a * (left + right)) * dz)


class Subbands:
    """Occupied subbands at the Fermi level from an interface.solve result r (or any dict with z, psi, E, mu, n_n,
    m_n, G_n). keep: indices of subbands to include (default: all with n_i > 1e-4 n_s)."""

    def __init__(self, r, keep=None):
        z = np.asarray(r["z"]) * 1e-10
        dz = z[1] - z[0]
        n = np.asarray(r["n_n"])
        if keep is None:
            keep = [i for i in range(len(n)) if n[i] > 1e-4 * n.sum()]
        self.keep = list(keep)
        self.z = z
        psi = np.asarray(r["psi"])[:, self.keep]
        self.psi = psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz)          # 1/sqrt(m)
        self.n = n[self.keep] * 1e4                                       # m^-2
        self.k = np.sqrt(2 * np.pi * self.n)                              # 1/m (spin 2)
        eF = np.asarray(r["mu"]) - np.asarray(r["E"])[self.keep]           # eV
        m = np.asarray(r["m_n"])[self.keep]; G = np.asarray(r["G_n"])[self.keep]
        self.md = m * (1 + 2 * np.maximum(eF, 0) / G) * M0                # kg, DOS / transport mass at E_F
        self.eF = eF
        self.ns = len(self.keep)

    def pi(self, q):
        """Static 2D Lindhard polarizability of each subband (1/(J m^2))."""
        x = 2 * self.k / q
        base = self.md / (np.pi * HBAR ** 2)
        return np.where(x >= 1, base, base * (1 - np.sqrt(np.clip(1 - x ** 2, 0, None))))


def _form(sb, q, z0, r=0.0, fcap=1.0):
    """F_ij(q, z0) for a point charge at z0 (m): matrix (ns, ns). With a dielectric boundary at z = 0: a charge in the
    cap (z0 < 0) is scaled by fcap = 2 eps / (eps + eps_cap); a charge at z0 > 0 gets the image term r exp(-q(|z| + z0))."""
    dz = sb.z[1] - sb.z[0]
    if z0 < 0:
        e = fcap * np.exp(-q * np.abs(sb.z - z0))
    else:
        e = np.exp(-q * np.abs(sb.z - z0)) + r * np.exp(-q * (np.abs(sb.z) + z0))
    return (sb.psi.T * e) @ sb.psi * dz


def _screen(sb, q, eps, phi0, Gcache):
    """Screened matrix elements V_ij from unscreened V0_ij (ns, ns) at momentum transfer q."""
    C = QE ** 2 / (2 * EPS0 * eps * q)
    P = sb.pi(q)
    ns = sb.ns
    Gd = np.array([[Gcache[(b, b, c)] for c in range(ns)] for b in range(ns)])      # G_{bb,cc}
    A = np.eye(ns) + C * Gd * P[None, :]
    phi = np.linalg.solve(A, np.diag(phi0))                                          # screened diagonal
    V = phi0 - C * np.einsum("ijc,c->ij", np.array([[[Gcache[(i, j, c)] for c in range(ns)] for j in range(ns)]
                                                    for i in range(ns)]), P * phi)
    for b in range(ns):
        V[b, b] = phi[b]
    return V


def _gcache(sb, q, r=0.0):
    dens = [sb.psi[:, c] ** 2 for c in range(sb.ns)]
    dz = sb.z[1] - sb.z[0]; ez = np.exp(-q * np.abs(sb.z))
    out = {}
    for i in range(sb.ns):
        for j in range(i, sb.ns):
            a = sb.psi[:, i] * sb.psi[:, j]
            for c in range(sb.ns):
                g = _G(sb.z, a, dens[c], q)
                if r != 0.0:
                    g += r * float(np.sum(a * ez) * dz) * float(np.sum(dens[c] * ez) * dz)
                out[(i, j, c)] = out[(j, i, c)] = g
    return out


def mobilities(r, eps=20.0, sheets=(), volume=(), dislocations=None, roughness=None, keep=None, nth=96, screening=True,
               eps_cap=None):
    """Low-temperature subband mobilities (cm^2/Vs) and transport times of the 2DEG in r.
    sheets: iterable of (z0_A, N_cm2, Z) point-charge sheets; volume: iterable of (z1_A, z2_A, N_cm3, Z) uniform slabs;
    dislocations: (N_cm2, f, c_A) charged lines along z; roughness: (V0_eV, Delta_A, Lambda_A, zI_A); eps_cap: permittivity
    of the z < 0 side (None: uniform eps). Returns dict(mu, tau, n, keep, eF, md)."""
    sb = Subbands(r, keep)
    rim = 0.0 if eps_cap is None else (eps - eps_cap) / (eps + eps_cap)
    fcap = 1.0 if eps_cap is None else 2 * eps / (eps + eps_cap)
    ns = sb.ns
    # angular grid for each pair i->j: q^2 = k_i^2 + k_j^2 - 2 k_i k_j cos(theta)
    th = (np.arange(nth) + 0.5) * np.pi / nth                     # symmetric in theta -> integrate 0..pi, x2
    a0 = np.zeros((ns, ns)); a1 = np.zeros((ns, ns))
    slabs = []
    for (z1, z2, N, Z) in volume:
        zs = np.linspace(z1, z2, max(int(abs(z2 - z1) / 4.0), 2) + 1)
        w = np.full(len(zs), (abs(z2 - z1) * 1e-8) / (len(zs) - 1)); w[0] /= 2; w[-1] /= 2
        for zz, ww in zip(zs, w):
            slabs.append((zz, N * ww, Z))                          # sheets of N dz (cm^-2)
    allsheets = list(sheets) + slabs
    if roughness is not None:
        V0r, Dr, Lr, zI = roughness
        iz = int(np.argmin(np.abs(sb.z - zI * 1e-10)))
        jz = iz - 1 if sb.z[iz] > zI * 1e-10 else iz + 1
        pI = 0.5 * (sb.psi[iz] + sb.psi[jz])                         # amplitudes at the interface plane (1/sqrt(m))
        GIR = V0r * QE * np.outer(pI, pI)                             # J/m
    cache = {}
    for i in range(ns):
        for j in range(ns):
            for t in th:
                q = np.sqrt(max(sb.k[i] ** 2 + sb.k[j] ** 2 - 2 * sb.k[i] * sb.k[j] * np.cos(t), 1e-6 * sb.k[i] ** 2))
                key = round(q / sb.k[0], 9)
                if key not in cache:
                    G = _gcache(sb, q, rim) if screening else None
                    M2 = np.zeros((ns, ns))
                    C = QE ** 2 / (2 * EPS0 * eps * q)
                    for (z0, N, Z) in allsheets:
                        V0 = Z * C * _form(sb, q, z0 * 1e-10, rim, fcap)
                        V = _screen(sb, q, eps, V0, G) if screening else V0
                        M2 += N * 1e4 * V ** 2
                    if roughness is not None:
                        V = _screen(sb, q, eps, GIR.copy(), G) if screening else GIR
                        M2 += V ** 2 * np.pi * (Dr * 1e-10) ** 2 * (Lr * 1e-10) ** 2 * np.exp(-(q * Lr * 1e-10) ** 2 / 4)
                    if dislocations is not None:
                        Nd, f, c = dislocations
                        lam = QE * f / (c * 1e-10)                    # C/m
                        V0 = np.eye(ns) * QE * lam / (EPS0 * eps * q ** 2)
                        V = _screen(sb, q, eps, V0, G) if screening else V0
                        M2 += Nd * 1e4 * V ** 2
                    cache[key] = M2
                M2 = cache[key]
                pref = sb.md[j] / (2 * np.pi * HBAR ** 3) * 2 * (np.pi / nth)        # x2: theta in (0, pi) mirrored
                a0[i, j] += pref * M2[i, j]
                a1[i, j] += pref * M2[i, j] * np.cos(t)
    tau = relax(a0, a1, sb.k, sb.md)
    mu = QE * tau / sb.md * 1e4
    return dict(mu=mu, tau=tau, n=sb.n * 1e-4, keep=sb.keep, eF=sb.eF, md=sb.md / M0, a0=a0, a1=a1, k=sb.k)


def relax(a0, a1, k, md):
    """Momentum relaxation times from the angular moments a0_ij = sum over i->j of the rate, a1_ij = the same weighted
    with cos(theta): Siggia-Kwok, 1 = sum_j a0_ij tau_i - sum_j a1_ij (v_j / v_i) tau_j with v = hbar k / m_d."""
    v = np.asarray(k) / np.asarray(md)
    A = np.diag(np.sum(a0, axis=1)) - a1 * (v[None, :] / v[:, None])
    return np.linalg.solve(A, np.ones(len(v)))


def magnetotransport(o, B):
    """Coupled-subband magnetotransport from a mobilities() result o at field B (T). Per subband the drift velocity u_i
    obeys the momentum balance m_i sum_j G_ij u_j = -e (E + u_i x B), G_ij = delta_ij sum_l a0_il - a1_ij k_j / k_i
    (exact for isotropic subbands with the same angular kernels as in mobilities; G diagonal gives independent Drude
    channels). Returns (rho_xx [Ohm], R_H = rho_xy / B [m^2/C], n_H = 1 / (e R_H) [cm^-2])."""
    a0, a1, k = np.asarray(o["a0"]), np.asarray(o["a1"]), np.asarray(o["k"])
    m = np.asarray(o["md"]) * M0; n = np.asarray(o["n"]) * 1e4
    G = np.diag(a0.sum(axis=1)) - a1 * (k[None, :] / k[:, None])
    Mx = m[:, None] * G - 1j * QE * B * np.eye(len(m))
    u = np.linalg.solve(Mx, -QE * np.ones(len(m)))            # complex drift per unit field (E along x)
    s = -QE * np.sum(n * u)                                     # sigma_xx + i sigma_yx
    sxx, syx = s.real, s.imag
    rxx = sxx / (sxx ** 2 + syx ** 2); rxy = syx / (sxx ** 2 + syx ** 2)
    if B == 0:
        return float(rxx), None, None
    RH = abs(rxy) / B
    return float(rxx), float(RH), float(1 / (QE * RH) * 1e-4)


def magnetoresistance(n, mu, B):
    """Classical magnetoresistance rho_xx(B) / rho_xx(0) - 1 of parallel Drude channels n_i (cm^-2), mu_i (cm^2/Vs)."""
    n = np.asarray(n, float); mu = np.asarray(mu, float) * 1e-4

    def rxx(b):
        x = mu * b
        sxx = np.sum(n * mu / (1 + x ** 2)); sxy = np.sum(n * mu * x / (1 + x ** 2))
        return sxx / (sxx ** 2 + sxy ** 2)
    return float(rxx(B) / rxx(0.0) - 1)


def hall(n, mu, B=0.0):
    """Hall density (cm^-2) and Hall mobility (cm^2/Vs) of parallel Drude channels n_i (cm^-2), mu_i (cm^2/Vs) at field
    B (T; 0 = low-field limit)."""
    n = np.asarray(n, float); mu = np.asarray(mu, float) * 1e-4      # m^2/Vs
    x = mu * B
    sxx = np.sum(n * mu / (1 + x ** 2)); sxy = np.sum(n * mu * x / (1 + x ** 2))
    if B == 0:
        nH = np.sum(n * mu) ** 2 / np.sum(n * mu ** 2)
        return float(nH), float(np.sum(n * mu ** 2) / np.sum(n * mu) * 1e4)
    Rxy = sxy / (sxx ** 2 + sxy ** 2) / B                             # per unit B, in units of 1/(n e)
    nH = 1 / Rxy
    return float(nH), float(1 / (nH * (sxx / (sxx ** 2 + sxy ** 2))) * 1e4)
