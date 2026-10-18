"""
krpa.py -- k-resolved time-dependent Hartree (RPA) intersubband response with in-plane nonparabolic dispersions.

Generalizes qwkit.tdh from subband populations to Fermi occupations f_n(k) on in-plane wavevector shells, keeping
the subband wavefunctions of k = 0 (the usual approximation for intersubband plasmons). For a z-polarised drive given
as the field E_d = D/eps_b the stack would have without the electrons (eps_b may depend on frequency, e.g. polar
phonons), the density-matrix equations of each k-shell are coupled only through the Hartree potential of the total
induced density:
    (hw - E_ab(k) + i g_ab) rho1_ab(k) = (f_b(k) - f_a(k)) Ut_ab,           Ut = U_ext + U_H[X1],  X1 = sum_k w_k rho1(k)
    (2hw - E_ab(k) + i g_ab) rho2_ab(k) = [Ut, rho1(k)]_ab + (f_b(k) - f_a(k)) U_H[X2]_ab        (a != b)
    (2hw + i hbar/tau) rho2_aa(k)      = [Ut, rho1(k)]_aa
with U_ext = e E_d z, U_H[X]_ab = (e / eps0 eps_b) sum_cd S_abcd X_cd (S: tdh.hartree_kernel), E_ab(k) = E_a(k) - E_b(k),
w_k the sheet density of a k-shell (k dk / pi, spin included). The sheet dipole per unit field is
p = -e sum_ab z_ba X_ab; dividing by a thickness L gives chi1 = P/(eps0 E_d), chi2 = P2/(eps0 E_d^2).
With a single shell (no dispersion) this reduces exactly to tdh.TDH with a single population time tau.
Units: energies eV, z Angstrom, sheet densities m^-2 internally; chi2 in pm/V.
"""
import numpy as np
from .tdh import hartree_kernel

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12; HBAR = 1.054571817e-34; M0 = 9.1093837015e-31
HBAR_EVS = HBAR / QE
KB = 8.617333262e-5


class KRPA:
    """z (A), psi (orthonormal columns), Ek (n, nk) subband energies on the k-shells (eV), fk (n, nk) occupations,
    wk (nk) shell sheet densities (m^-2 per unit occupation), gamma: HWHM (eV, scalar or (n, n) matrix), eps_b: float or
    callable hw -> complex background permittivity, tau_ps: population time, kernel: include the Hartree coupling."""

    def __init__(self, z, psi, Ek, fk, wk, gamma=0.010, eps_b=13.5, tau_ps=1.0, kernel=True, zc=None, occ_tol=1e-6):
        self.n = n = psi.shape[1]
        dz = z[1] - z[0]
        zc = 0.5 * (z[0] + z[-1]) if zc is None else zc
        self.Z = (psi.T * (z - zc)) @ psi * dz                      # A
        self.Ek = np.asarray(Ek, float); self.fk = np.asarray(fk, float); self.wk = np.asarray(wk, float)
        self.S = hartree_kernel(z, psi)                             # m
        G = np.broadcast_to(np.asarray(gamma, float), (n, n)) if np.ndim(gamma) == 2 else np.full((n, n), float(gamma))
        self.G = np.array(G, float)
        self.g_pop = HBAR_EVS / (tau_ps * 1e-12)
        self.eps_b = eps_b
        self.kernel = kernel
        self.Eab = self.Ek[:, None, :] - self.Ek[None, :, :]        # (n, n, nk)
        self.df = self.fk[None, :, :] - self.fk[:, None, :]         # f_b - f_a, (n, n, nk)
        # pairs that can carry a coherence: chi0_ab = 0 identically when f_a = f_b on every shell (both empty)
        act = np.max(np.abs(self.df), axis=2) > occ_tol
        self.off = [(a, b) for a in range(n) for b in range(n) if a != b and act[a, b]]
        self.fxc = None
        self.ns = float(np.sum(self.fk * self.wk[None, :]))         # m^-2

    # ---- construction from an interface (or any) self-consistent solution with Kane in-plane dispersions ----------
    @classmethod
    def from_subbands(cls, z, psi, E, m_n, G_n, mu, T, nk=300, kfrac_max=None, **kw):
        """Shells on a common k grid; subband n disperses as eps(1 + eps/G_n) = hbar^2 k^2 / 2 m_n."""
        from .interface import dispersion
        kT = KB * max(T, 1e-3)
        # x0 = hbar^2 k^2 / 2 m0 (eV); highest occupied kinetic energy among subbands
        x0max = 0.0
        for n in range(len(E)):
            emax = max(mu - E[n], 0.0) + 30 * kT
            x0max = max(x0max, m_n[n] * emax * (1 + emax / G_n[n]))
        x0 = np.linspace(0.0, x0max, nk)
        w = np.full(nk, x0[1] - x0[0]); w[0] *= 0.5; w[-1] *= 0.5
        wk = M0 * QE / (np.pi * HBAR ** 2) * w                       # m^-2 per shell
        Ek = np.array([E[n] + dispersion(x0 / m_n[n], m_n[n], G_n[n]) for n in range(len(E))])
        fk = 0.5 * (1.0 - np.tanh((Ek - mu) / (2 * kT)))
        obj = cls(z, psi, Ek, fk, wk, **kw)
        obj.x0 = x0
        return obj

    # ---- helpers -----------------------------------------------------------------------------------------------------
    def _eps(self, hw):
        e = self.eps_b(hw) if callable(self.eps_b) else self.eps_b
        return e

    def _Kp(self, hw):
        return QE / (EPS0 * self._eps(hw)) if self.kernel else 0.0      # eV per (m * m^-2)

    def chi0(self, hw):
        """k-summed bare response chi0_ab = sum_k w (f_b - f_a)/(hw - E_ab + i g_ab), (n, n) in m^-2 / eV."""
        den = hw - self.Eab + 1j * self.G[:, :, None]
        c = np.sum(self.wk[None, None, :] * self.df / den, axis=2)
        np.fill_diagonal(c, 0.0)
        return c

    def set_xc(self, z, psi, fxc_eVm2):
        """Adiabatic local exchange-correlation kernel: fxc(z) = dV_xc/dn (eV m^3 per unit 3D density, given on z in A
        as eV*m^3); adds U_xc[X]_ab = sum_cd (int psi_a psi_b fxc psi_c psi_d dz) X_cd (X in m^-2)."""
        dzm = (z[1] - z[0]) * 1e-10
        pm = psi * 1e5                                               # 1/sqrt(m)
        P2 = pm[:, :, None] * pm[:, None, :]                         # (Nz, n, n)
        self.Kxc = np.einsum("zij,z,zkl->ijkl", P2, fxc_eVm2, P2) * dzm   # eV m^2
        return self

    def _Kslice(self, hw, ia, ib, ic, id_):
        """Total kernel (eV per m^-2) for row pairs (ia, ib) and column pairs (ic, id_): Hartree with the
        frequency-dependent background plus the optional XC kernel; never forms the full complex tensor."""
        K = self._Kp(hw) * self.S[ia[:, None], ib[:, None], ic[None, :], id_[None, :]]
        if getattr(self, "Kxc", None) is not None:
            K = K + self.Kxc[ia[:, None], ib[:, None], ic[None, :], id_[None, :]]
        return K

    def UH(self, X, hw):
        """Induced (Hartree + XC) potential matrix for density matrix X, using only its nonzero entries."""
        if not self.kernel:
            return np.zeros_like(X)
        ic, id_ = np.nonzero(np.abs(X) > 0)
        if len(ic) == 0:
            return np.zeros_like(X)
        n = self.n
        ia, ib = np.indices((n, n)).reshape(2, -1)
        U = self._Kslice(hw, ia, ib, ic, id_) @ X[ic, id_]
        return U.reshape(n, n)

    def _solve_offdiag(self, chi0, hw, rhs):
        """Solve X = rhs + chi0 o (K X) on the active off-diagonal pairs (K: total kernel at frequency hw)."""
        if not self.kernel or len(self.off) == 0:
            return rhs * self._mask()
        ia = np.array([a for a, b in self.off]); ib = np.array([b for a, b in self.off])
        Koff = self._Kslice(hw, ia, ib, ia, ib)
        A = np.eye(len(ia), dtype=complex) - chi0[ia, ib][:, None] * Koff
        x = np.linalg.solve(A, rhs[ia, ib])
        X = np.zeros((self.n, self.n), complex)
        X[ia, ib] = x
        return X

    def _mask(self):
        M = np.zeros((self.n, self.n))
        for a, b in self.off:
            M[a, b] = 1.0
        return M

    # ---- response ------------------------------------------------------------------------------------------------
    def first_order(self, hw):
        U = self.Z * 1e-10                                          # eV for E_d = 1 V/m
        c0 = self.chi0(hw)
        X1 = self._solve_offdiag(c0, hw, c0 * U)
        Ut = U + self.UH(X1, hw)
        return X1, Ut

    def sheet_p1(self, hw):
        """Induced sheet dipole per unit drive field (C/m per V/m = F)."""
        X1, _ = self.first_order(hw)
        return -QE * np.sum(self.Z.T * 1e-10 * X1)

    def chi1(self, hw, L):
        """Dimensionless susceptibility of a layer of thickness L (A) containing the sheet."""
        return self.sheet_p1(hw) / (EPS0 * L * 1e-10)

    def loss(self, hw):
        """Power absorbed per unit area for unit drive field, (omega/2) Im p1 (W/m^2 per (V/m)^2)."""
        w = hw * QE / HBAR
        return 0.5 * w * np.imag(self.sheet_p1(hw))

    def second_order(self, hw):
        X1, Ut = self.first_order(hw)
        if not hasattr(self, "_EabK"):
            self._EabK = np.ascontiguousarray(np.moveaxis(self.Eab, 2, 0))   # (nk, n, n)
            self._dfK = np.ascontiguousarray(np.moveaxis(self.df, 2, 0))
        r1 = self._dfK * Ut[None, :, :] / (hw - self._EabK + 1j * self.G[None, :, :])   # rho1(k), (nk, n, n)
        comm = np.matmul(Ut[None, :, :], r1) - np.matmul(r1, Ut[None, :, :])
        G2 = self.G.copy(); np.fill_diagonal(G2, self.g_pop)
        X2s = np.einsum("k,kab->ab", self.wk, comm / (2 * hw - self._EabK + 1j * G2[None, :, :]))  # source part of X2
        c02 = self.chi0(2 * hw)
        if not self.kernel:
            X2 = X2s
        else:
            M = self._mask()
            Xfix = X2s * (1.0 - M)                                   # diagonal and inactive pairs: no feedback on themselves
            rhs = (X2s + c02 * self.UH(Xfix, 2 * hw)) * M
            X2 = self._solve_offdiag(c02, 2 * hw, rhs) + Xfix
        return X2

    def sheet_p2(self, hw):
        X2 = self.second_order(hw)
        return -QE * np.sum(self.Z.T * 1e-10 * X2)

    def chi2(self, hw, L):
        """Second-order susceptibility chi(2)(2w; w, w) (pm/V) of a layer of thickness L (A)."""
        return self.sheet_p2(hw) / (EPS0 * L * 1e-10) * 1e12


def phonon_eps(eps_inf, modes, damping=0.002):
    """Factorized background permittivity eps(hw) = eps_inf prod (hw_L^2 - hw^2 - i hw g)/(hw_T^2 - hw^2 - i hw g);
    modes = [(hw_TO, hw_LO), ...] in eV."""
    def f(hw):
        e = complex(eps_inf)
        for t, l in modes:
            e *= (l ** 2 - hw ** 2 - 1j * hw * damping) / (t ** 2 - hw ** 2 - 1j * hw * damping)
        return e
    return f


def froehlich_weights(eps_inf, modes):
    """Per-mode Froehlich factors C_j = (1/eps_inf - 1/eps_0)_j from the residues of 1/eps(omega) at the LO poles of
    the factorized dielectric function (sum_j C_j = 1/eps_inf - 1/eps_0). modes = [(hw_TO, hw_LO), ...]."""
    xT = np.array([t ** 2 for t, _ in modes]); xL = np.array([l ** 2 for _, l in modes])
    C = []
    for j in range(len(modes)):
        num = np.prod(xL[j] - xT)
        den = xL[j] * np.prod([xL[j] - xL[k] for k in range(len(modes)) if k != j])
        C.append(num / den / eps_inf)
    return np.array(C)
