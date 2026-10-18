"""
tdh.py -- self-consistent (time-dependent Hartree, RPA) intersubband response in the subband basis.

Density-matrix equation for the conduction-subband populations and coherences of one period, with the
z-polarised drive given as the field the stack would have without the electrons, E_d = D/eps_b:
    d rho/dt = -(i/hbar) [H0 + U_ext(t) + U_H[rho - rho0], rho] + Gamma(rho - rho0)
    U_ext = e E_d(t) z                                   (electron potential energy in a field along +z)
    U_H[d rho]_ij = (e^2 n_s / eps0 eps_r) sum_kl S_ijkl d rho_kl,  S_ijkl = int F_ij(z) F_kl(z) dz,
    F_ij(z) = int_{-inf}^{z} psi_i psi_j dz'
U_H is the Hartree (depolarisation) potential of the induced charge with D unchanged, which reproduces Ando's
depolarisation shift E~^2 = E^2 + 2 E (e^2 n_s dRho S / eps) for two levels. Relaxation Gamma: coherences decay
at gamma/hbar; populations follow a rate matrix R[l, m] (1/s, from l to m) or a single time tau.

Frequency domain (E_d(t) = E1 exp(-i w t) + c.c., Boyd convention):
    (-i w - L0) rho1 = -(i/hbar) [U1, rho0],                      U1 = e E1 z
    (-2i w - L0) rho2 = -(i/hbar) [U1 + U_H[rho1], rho1]
with L0(x) = -(i/hbar)([H0, x] + [U_H[x], rho0]) + Gamma(x). Then
    chi1 = P1/(eps0 E1),  chi2 = P2/(eps0 E1^2),  P = -e N Tr(z rho),  N = volume density (period average).
Setting the kernel to zero gives the bare (single-particle) susceptibilities; core.chi2_general is recovered.
Units: z in Angstrom, energies in eV, rates in 1/s; chi1 dimensionless, chi2 in pm/V.
"""
import numpy as np

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12; HBAR = 1.054571817e-34; HBAR_EVS = HBAR / QE


def hartree_kernel(z, psi):
    """S_ijkl = int F_ij F_kl dz (m), F_ij(z) = int_{-inf}^z psi_i psi_j (z in A, psi normalised in 1/sqrt(A))."""
    dz = z[1] - z[0]
    n = psi.shape[1]
    F = np.cumsum(psi[:, :, None] * psi[:, None, :], axis=0) * dz          # (Nz, n, n), dimensionless
    F = F - 0.5 * psi[:, :, None] * psi[:, None, :] * dz                    # cell-centred cumulative sum
    F2 = F.reshape(F.shape[0], n * n)
    S = (F2.T @ F2).reshape(n, n, n, n) * dz * 1e-10                        # m
    return S


class TDH:
    """Self-consistent subband response of one period.
    E (eV), psi (columns, on grid z in A), rho0 (fractional populations), ns_cm2 (sheet density per period),
    N_cm3 (volume density for P), gamma (HWHM, eV), eps_r (static-background permittivity for the Hartree field),
    R (rate matrix 1/s) or tau_ps (single population time)."""

    def __init__(self, z, psi, E, rho0, ns_cm2, N_cm3, gamma=0.010, eps_r=13.5, R=None, tau_ps=1.0, kernel=True,
                 zc=None):
        self.n = n = len(E)
        dz = z[1] - z[0]
        zc = 0.5 * (z[0] + z[-1]) if zc is None else zc
        self.E = np.array(E, float)
        self.Z = (psi.T * (z - zc)) @ psi * dz                                # A (states orthonormal)
        self.rho0 = np.diag(np.array(rho0, float)).astype(complex)
        self.N = N_cm3 * 1e6
        self.g = gamma
        ns = ns_cm2 * 1e4
        self.S = hartree_kernel(z, psi)
        self.Kpref = QE * ns / (EPS0 * eps_r)                                 # eV per (m of S): e^2 ns S/(eps0 eps_r)/e
        self.use_kernel = kernel
        self.R = None if R is None else np.array(R, float)
        self.tau = tau_ps * 1e-12
        self._build()

    # ---- operators -------------------------------------------------------------------------------
    def UH(self, d):
        """Hartree potential matrix (eV) for density-matrix change d."""
        if not self.use_kernel:
            return np.zeros_like(d)
        return self.Kpref * np.einsum("ijkl,kl->ij", self.S, d)

    def relax(self, d):
        """Gamma(d): coherence decay and population relaxation (1/s units times d)."""
        n = self.n
        out = -(self.g / HBAR_EVS) * d
        pops = np.real(np.diag(d))
        if self.R is None:
            dp = -pops / self.tau
        else:
            dp = self.R.T @ pops - pops * self.R.sum(axis=1)
        out[np.diag_indices(n)] = dp
        return out

    def _L0(self, d):
        H0 = np.diag(self.E)
        U = self.UH(d)
        return -(1j / HBAR_EVS) * (H0 @ d - d @ H0 + U @ self.rho0 - self.rho0 @ U) + self.relax(d)

    def _build(self):
        n = self.n
        M = np.zeros((n * n, n * n), complex)
        for a in range(n * n):
            e = np.zeros(n * n, complex); e[a] = 1.0
            M[:, a] = self._L0(e.reshape(n, n)).reshape(-1)
        self.L0 = M

    def _solve(self, w, src):
        A = -1j * w * np.eye(self.n ** 2) - self.L0
        return np.linalg.solve(A, src.reshape(-1)).reshape(self.n, self.n)

    # ---- susceptibilities -------------------------------------------------------------------------
    def response(self, hw):
        """Return chi1(w) (dimensionless), chi2(2w; w, w) (pm/V) and the first- and second-order density
        matrices for unit drive amplitude E1 = 1 V/m."""
        w = hw / HBAR_EVS
        U1 = self.Z * 1e-10                      # e E z with E = 1 V/m -> eV (e * 1 V/m * z[m] = z[m] eV)
        rho1 = self._solve(w, -(1j / HBAR_EVS) * (U1 @ self.rho0 - self.rho0 @ U1))
        Ut = U1 + self.UH(rho1)
        rho2 = self._solve(2 * w, -(1j / HBAR_EVS) * (Ut @ rho1 - rho1 @ Ut))
        P1 = -QE * self.N * np.trace(self.Z * 1e-10 @ rho1)
        P2 = -QE * self.N * np.trace(self.Z * 1e-10 @ rho2)
        return P1 / EPS0, P2 / EPS0 * 1e12, rho1, rho2

    def chi1(self, hw):
        return self.response(hw)[0]

    def chi2(self, hw):
        return self.response(hw)[1]

    def alpha(self, hw, nref=3.3):
        """Absorption coefficient (1/m) of the medium referenced to the drive field: (w/(n c)) Im chi1."""
        w = hw * QE / HBAR
        return w / (nref * 2.99792458e8) * np.imag(self.chi1(hw))

    # ---- real time ---------------------------------------------------------------------------------
    def evolve(self, E0, hw, t_end_ps=None, steps_per_period=64, ramp_ps=1.0, harmonics=(1, 2), avg_periods=20):
        """Integrate the full nonlinear TDH equation with drive E_d(t) = 2 E0 cos(w t) (E0 = Boyd amplitude, V/m),
        smoothly ramped on over ramp_ps. t_end_ps defaults to ramp + 8 population lifetimes (population transients
        excited by the ramp decay on that scale and bias the harmonics if the run is cut short). Returns the complex polarisation amplitudes P_k (Boyd convention,
        P(t) = sum_k P_k e^{-i k w t} + c.c.) over the last avg_periods periods, and the final populations."""
        w = hw / HBAR_EVS
        T = 2 * np.pi / w
        if t_end_ps is None:
            if self.R is None:
                tl = self.tau
            else:                                    # slowest decay among excited states (not thermal up-rates)
                out_rates = self.R.sum(axis=1)[1:]
                tl = 1.0 / np.min(out_rates[out_rates > 0])
            t_end_ps = ramp_ps + 8 * tl * 1e12
        dt = T / steps_per_period                      # integer steps per period: exact harmonic windows
        nsteps = int(t_end_ps * 1e-12 / T) * steps_per_period
        H0 = np.diag(self.E)
        Zm = self.Z * 1e-10

        def rhs(t, r):
            env = 1.0 if t > ramp_ps * 1e-12 else np.sin(0.5 * np.pi * t / (ramp_ps * 1e-12)) ** 2
            Ef = 2 * E0 * env * np.cos(w * t)
            U = H0 + Ef * Zm + self.UH(r - self.rho0)
            return -(1j / HBAR_EVS) * (U @ r - r @ U) + self.relax(r - self.rho0)

        r = self.rho0.copy(); t = 0.0
        n_avg = avg_periods * steps_per_period
        acc = {k: 0j for k in harmonics}; cnt = 0
        for i in range(nsteps):
            k1 = rhs(t, r); k2 = rhs(t + dt / 2, r + dt / 2 * k1)
            k3 = rhs(t + dt / 2, r + dt / 2 * k2); k4 = rhs(t + dt, r + dt * k3)
            r = r + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
            if i >= nsteps - n_avg:
                P = -QE * self.N * np.real(np.trace(Zm @ r))
                for k in harmonics:
                    acc[k] += P * np.exp(1j * k * w * t)
                cnt += 1
        Pk = {k: acc[k] / cnt for k in harmonics}             # = P_k (since <P e^{ikwt}> picks the e^{-ikwt} term)
        return Pk, np.real(np.diag(r))
