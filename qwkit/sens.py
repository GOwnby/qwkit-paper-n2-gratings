"""
sens.py -- exact parameter sensitivities of the self-consistent Schroedinger-Poisson model (one period) by implicit
(adjoint-free, forward) differentiation, for gradient-based and growth-tolerance-aware design and for uncertainty
quantification.

Parameters (vector theta, names in Model.names):
  t0..t{n-1}   layer thicknesses (A); the right spacer absorbs the change so the period length is fixed
  F            applied mean field (kV/cm)
  ns           sheet density per period (cm^-2)
  dVb          shift of the barrier conduction-band offset (eV)
  mw, mb       band-edge masses of well and barrier (m0)
  Egw, Egb     nonparabolicity gaps of well and barrier (eV)
  epsw, epsb   static permittivities of well and barrier (Hartree field)
Material profiles on the cell-centred grid use interfaces smoothed by an error function of width sigma (default
0.5 A, one grid cell; a stand-in for the ~1 monolayer grading of real interfaces), so every profile is smooth in the
interface positions. For sigma -> 0 and interfaces on cell boundaries the model reduces to qwkit.sp.

Method. At self-consistency V_H = G(V_H, theta) with G = Hartree[n(V0 + Vext + V_H)]. Then
  dV_H/dtheta = (I - dG/dV_H)^-1 dG/dtheta|_{V_H},
where the density response dn/dV (an N x N matrix) and dn/dtheta follow from first-order perturbation theory of the
nonparabolic eigenproblems: for state k with mass m_k(z) = m_edge (1 + (E_k - V)/Eg),
  dE_k = <v|dH|v> / (1 - <v|dH/dE|v>),   dv = -R_k (dH + dH/dE dE_k - dE_k) v,
R_k the reduced resolvent of H(E_k), and Fermi-Dirac occupations at fixed total density. The post-processing
(Loewdin orthonormalisation, dipoles, depolarisation shifts, chi(2), absorption) is differentiated along each state
tangent by a Richardson-extrapolated central difference of the cheap closed-form observables (semi-analytic;
relative accuracy ~1e-9, limited by round-off).
Units: Angstrom, eV, kV/cm, cm^-2.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal, lapack
from scipy.sparse.linalg import LinearOperator, gmres
from . import core as q, sp

KB = 8.617333e-5
HBAR = 1.054571817e-34; M0 = 9.1093837e-31


class Model:
    """layers: list of (thickness_A, 'well'|'barrier'); spacer split equally on both sides (barrier)."""

    def __init__(self, layers, F=0.0, ns=5e11, T=300.0, spacer=200.0, dz=0.5, nstates=7, m_dos=0.043,
                 doping="spacer", sigma=0.5, dense=False, sigma_R=1e-10, refine_R=2, gmres_rtol=1e-13):
        """dense=True uses the original dense path (full eigendecompositions, explicit N x N density response);
        the default is matrix-free: tridiagonal solves for the reduced resolvents and GMRES for the Hartree
        feedback, O(n N) per Krylov step instead of O(n N^3)."""
        self.kinds = [k for _, k in layers]; self.sigma = sigma
        self.dense = dense; self.sigma_R = sigma_R; self.refine_R = refine_R; self.gmres_rtol = gmres_rtol
        self.dz = dz; self.T = T; self.nstates = nstates; self.m_dos = m_dos; self.doping = doping
        self.pad = spacer / 2.0
        Ltot = sum(t for t, _ in layers) + spacer
        n = int(round(Ltot / dz)); self.z = (np.arange(n) + 0.5) * dz; self.L = n * dz
        self.names = [f"t{i}" for i in range(len(layers))] + ["F", "ns", "dVb", "mw", "mb", "Egw", "Egb", "epsw", "epsb"]
        mw, mb = q.MAT["well"], q.MAT["barrier"]
        self.theta0 = np.array([t for t, _ in layers] + [F, ns, 0.0, mw["m"], mb["m"], mw["Eg"], mb["Eg"],
                                                        sp.EPS_R["well"], sp.EPS_R["barrier"]], float)
        self.nl = len(layers)
        self.Vb0 = mb["V"]

    # ---------------- geometry ----------------
    def _interfaces(self, th):
        x = self.pad + np.concatenate([[0.0], np.cumsum(th[:self.nl])])
        return x                                         # x[0] start of layer 0, x[nl] end of last layer

    def profiles(self, th):
        """Cell values of V0, m_edge, Eg, eps and the fraction of each cell that is well material."""
        z, dz = self.z, self.dz
        x = self._interfaces(th)
        from scipy.special import erf
        sg = self.sigma
        left = lambda xx: 0.5 * (1.0 + erf((xx - z) / (np.sqrt(2.0) * sg)))   # smoothed: material left of xx
        fw = np.zeros_like(z)
        for i, k in enumerate(self.kinds):
            if k == "well":
                fw += left(x[i + 1]) - left(x[i])
        fb = 1.0 - fw
        mwv, mbv, Egw, Egb, ew, eb = th[self.nl + 3:self.nl + 9]
        V0 = fb * (self.Vb0 + th[self.nl + 2])
        return dict(V0=V0, m=fw * mwv + fb * mbv, Eg=fw * Egw + fb * Egb, eps=fw * ew + fb * eb, fw=fw, x=x)

    def dprofiles(self, th):
        """Derivatives of (V0, m, Eg, eps) with respect to every parameter: arrays (N, p)."""
        z, dz = self.z, self.dz
        N, p = len(z), len(th)
        x = self._interfaces(th)
        P = self.profiles(th)
        sg = self.sigma
        dleft = lambda xx: np.exp(-0.5 * ((xx - z) / sg) ** 2) / (np.sqrt(2 * np.pi) * sg)
        dfw = np.zeros((N, p))
        # d fw / d x_j, then x_j depends on t_0..t_{j-1}
        dfw_dx = np.zeros((N, self.nl + 1))
        for i, k in enumerate(self.kinds):
            if k == "well":
                dfw_dx[:, i + 1] += dleft(x[i + 1]); dfw_dx[:, i] -= dleft(x[i])
        for j in range(self.nl + 1):
            for i in range(j):                      # x_j = pad + sum_{i<j} t_i
                dfw[:, i] += dfw_dx[:, j]
        mwv, mbv, Egw, Egb, ew, eb = th[self.nl + 3:self.nl + 9]
        Vb = self.Vb0 + th[self.nl + 2]
        out = {key: np.zeros((N, p)) for key in ("V0", "m", "Eg", "eps")}
        out["V0"] = -dfw * Vb; out["m"] = dfw * (mwv - mbv); out["Eg"] = dfw * (Egw - Egb); out["eps"] = dfw * (ew - eb)
        fw = P["fw"]; fb = 1 - fw; o = self.nl
        out["V0"][:, o + 2] += fb
        out["m"][:, o + 3] += fw; out["m"][:, o + 4] += fb
        out["Eg"][:, o + 5] += fw; out["Eg"][:, o + 6] += fb
        out["eps"][:, o + 7] += fw; out["eps"][:, o + 8] += fb
        return out

    def _zc(self, th):
        x = self._interfaces(th)
        return 0.5 * (x[0] + x[-1])

    def vext(self, th):
        return -th[self.nl] * 1e-5 * (self.z - self._zc(th))

    def dvext(self, th):
        """(N, p): dVext/dtheta (field and the shift of the tilt origin with the layer thicknesses)."""
        N, p = len(self.z), len(th)
        d = np.zeros((N, p))
        d[:, self.nl] = -1e-5 * (self.z - self._zc(th))
        for i in range(self.nl):
            d[:, i] = th[self.nl] * 1e-5 * 0.5           # d zc / d t_i = 1/2
        return d

    def nd(self, th):
        z = self.z
        nd = np.zeros_like(z); w = 20.0
        nd[(z < w / 2) | (z > self.L - w / 2)] = 1.0
        return nd / (nd.sum() * self.dz) * th[self.nl + 1]

    # ---------------- forward ----------------
    def forward(self, th, tol=1e-10, maxit=2000, mix=0.4, VH0=None):
        th = np.asarray(th, float)
        P = self.profiles(th)
        V0 = P["V0"] + self.vext(th)
        ns = th[self.nl + 1]
        nd = self.nd(th)
        VH = np.zeros_like(self.z) if VH0 is None else VH0.copy()
        Eprev = None
        for it in range(maxit):
            E, psi = q.solve_states(self.z, V0 + VH, P["m"], P["Eg"], nstates=self.nstates, E_guess=Eprev, tol=1e-12, iters=200)
            Eprev = E
            rho = q.populations(E, ns, self.T, m_eff=self.m_dos)
            n = (psi ** 2) @ (rho * ns)
            VHn = sp.hartree(self.z, nd, n, P["eps"])
            err = np.max(np.abs(VHn - VH))
            VH = (1 - mix) * VH + mix * VHn
            if err < tol:
                break
        E, psi = q.solve_states(self.z, V0 + VH, P["m"], P["Eg"], nstates=self.nstates, E_guess=Eprev, tol=1e-12, iters=200)
        rho = q.populations(E, ns, self.T, m_eff=self.m_dos)
        return dict(th=th, V=V0 + VH, VH=VH, E=E, psi=psi, rho=rho, P=P, nd=nd, iters=it + 1, err=err)

    # ---------------- linear response machinery ----------------
    def _state_ops(self, s):
        """Per state: unit vector v, masses, tridiagonal couplings t, reduced resolvent R, and b = <v|dH/dE|v>."""
        z, dz = self.z, self.dz
        P = s["P"]; V = s["V"]
        ops = []
        for k in range(self.nstates):
            Ek = s["E"][k]
            mz = np.clip(P["m"] * (1.0 + (Ek - V) / P["Eg"]), 0.2 * P["m"], None)
            mh = 0.5 * (mz[:-1] + mz[1:]); t = q.HB2M0 / mh / dz ** 2
            diag = V.copy(); diag[:-1] += t; diag[1:] += t
            diag[0] += q.HB2M0 / mz[0] / dz ** 2; diag[-1] += q.HB2M0 / mz[-1] / dz ** 2      # hard walls (core._solve)
            if self.dense:
                w, U = eigh_tridiagonal(diag, -t)
                kk = int(np.argmin(np.abs(w - Ek)))
                v = U[:, kk] * np.sign(np.sum(U[:, kk] * s["psi"][:, k]))
                den = w - Ek; den[kk] = np.inf
                R = (U / den) @ U.T
                lu = None
            else:
                # eigenvector of H(E_k) by one step of inverse iteration from the converged state, and an LU
                # factorization of H(E_k) - E_k + sigma (sigma small; the near-null direction v is projected out and
                # the O(sigma) bias removed by iterative refinement in _apply_R)
                sig = self.sigma_R
                dl = -t.copy(); du = -t.copy(); dd = diag - Ek + sig
                lu = lapack.dgttrf(dl, dd, du)
                v = s["psi"][:, k] * np.sqrt(dz)
                v = lapack.dgttrs(*lu[:5], v)[0]; v /= np.linalg.norm(v)
                v *= np.sign(np.sum(v * s["psi"][:, k]))
                R = None
            dmdE = P["m"] / P["Eg"] * (mz > 0.2 * P["m"])
            b = self._vTdTv(v, t, mh, dmdE, mz, dz)
            ops.append(dict(v=v, mz=mz, mh=mh, t=t, R=R, lu=lu, b=b, Ek=Ek))
        return ops

    def _apply_R(self, o, rhs):
        """Reduced resolvent R_k = (H(E_k) - E_k)^+ restricted to the complement of v, applied to rhs (N, q)."""
        if o["R"] is not None:
            return o["R"] @ rhs
        v = o["v"]; lu = o["lu"]; sig = self.sigma_R
        r = rhs - np.outer(v, v @ rhs)
        x = lapack.dgttrs(*lu[:5], r)[0]
        x -= np.outer(v, v @ x)
        for _ in range(self.refine_R):                     # (A + sig) x = r  =>  A x = r - sig x: refine
            dx = lapack.dgttrs(*lu[:5], sig * x)[0]
            dx -= np.outer(v, v @ dx)
            x += dx
        return x

    @staticmethod
    def _vTdTv(v, t, mh, dm, mz=None, dz=None):
        """<v| T'[dm] |v> for a mass perturbation dm (N,) or (N, q) (hard-wall end terms included if mz is given)."""
        g = -t / mh * (v[:-1] - v[1:]) ** 2                 # (N-1,)
        dmh = 0.5 * (dm[:-1] + dm[1:])
        out = np.tensordot(g, dmh, axes=(0, 0))
        if mz is not None:
            c = q.HB2M0 / dz ** 2
            out = out - c / mz[0] ** 2 * v[0] ** 2 * dm[0] - c / mz[-1] ** 2 * v[-1] ** 2 * dm[-1]
        return out

    @staticmethod
    def _TdTv(v, t, mh, dm, mz=None, dz=None):
        """T'[dm] v for dm (N, q): returns (N, q) (hard-wall end terms included if mz is given)."""
        dmh = 0.5 * (dm[:-1] + dm[1:])                     # (N-1, q)
        dt = -(t / mh)[:, None] * dmh
        out = np.zeros((len(v), dm.shape[1]))
        diff = (v[:-1] - v[1:])[:, None] * dt                # dt_i (v_i - v_{i+1})
        out[:-1] += diff; out[1:] -= diff
        if mz is not None:
            c = q.HB2M0 / dz ** 2
            out[0] -= c / mz[0] ** 2 * v[0] * dm[0]
            out[-1] -= c / mz[-1] ** 2 * v[-1] * dm[-1]
        return out

    def response(self, s, ops, dV, dm, dEg, dns):
        """First-order response of energies, states, occupations and density to perturbations
        dV, dm (band-edge mass), dEg (N, q) and dns (q,). Returns dE (n, q), dpsi (N, n, q), dnk (n, q), dn (N, q)."""
        dz = self.dz; P = s["P"]; V = s["V"]
        n, q_ = self.nstates, dV.shape[1]
        dE = np.zeros((n, q_)); dv = np.zeros((len(V), n, q_))
        for k, o in enumerate(ops):
            v = o["v"]; Ek = o["Ek"]
            act = (o["mz"] > 0.2 * P["m"])
            fac = (1.0 + (Ek - V) / P["Eg"])
            # mass change of state k at fixed E_k
            dmk = (dm * fac[:, None] - (P["m"] / P["Eg"])[:, None] * dV
                   - (P["m"] * (Ek - V) / P["Eg"] ** 2)[:, None] * dEg) * act[:, None] \
                + 0.2 * dm * (~act)[:, None]                 # clipped cells: m_z = 0.2 m_edge
            dHv = dV * v[:, None] + self._TdTv(v, o["t"], o["mh"], dmk, o["mz"], dz)
            a = v @ dHv
            dE[k] = a / (1.0 - o["b"])
            dHEv = self._TdTv(v, o["t"], o["mh"], (P["m"] / P["Eg"] * act)[:, None], o["mz"], dz)[:, 0]
            rhs = dHv + np.outer(dHEv, dE[k]) - np.outer(v, dE[k])
            dv[:, k, :] = -self._apply_R(o, rhs)
        # occupations at fixed total density (Fermi-Dirac, parabolic 2D DOS)
        E = s["E"]; ns = s["th"][self.nl + 1]
        D = self.m_dos * M0 / (np.pi * HBAR ** 2) * q.E_CH * 1e-4
        kT = KB * self.T
        mu = self._mu(E, ns)
        f = 1.0 / (1.0 + np.exp((E - mu) / kT))
        dmu = (dns / D + f @ dE) / f.sum()
        dnk = D * f[:, None] * (dmu[None, :] - dE)          # cm^-2
        nk = s["rho"] * ns
        psi = s["psi"]
        dpsi = dv / np.sqrt(dz)
        dn = (psi ** 2) @ dnk + 2 * np.einsum("ik,k,ikq->iq", psi, nk, dpsi)
        return dE, dpsi, dnk, dn

    def _mu(self, E, ns):
        from scipy.optimize import brentq
        D = self.m_dos * M0 / (np.pi * HBAR ** 2) * q.E_CH * 1e-4; kT = KB * self.T
        return brentq(lambda mu: np.sum(D * kT * np.logaddexp(0, (mu - E) / kT)) - ns, E[0] - 2.0, E[-1] + 2.0)

    def hartree_matrix(self, eps):
        """Linear map (N x N): VH = H_n (nd - n), for densities in cm^-2/A."""
        z, dz = self.z, self.dz
        N = len(z)
        Iden = np.eye(N)
        rho = q.E_CH * Iden * 1e4 / 1e-10
        D = (np.cumsum(rho, axis=0) - 0.5 * rho) * dz * 1e-10
        D -= (np.sum(D / eps[:, None], axis=0) / np.sum(1.0 / eps))[None, :]
        Ef = D / (eps[:, None] * q.EPS0)
        phi = -(np.cumsum(Ef, axis=0) - 0.5 * Ef) * dz * 1e-10
        VH = -phi
        return VH - VH.mean(axis=0, keepdims=True)

    def _hartree_lin(self, y, eps):
        """H_n y for densities y (N, q) in cm^-2/A, without forming the matrix (same discretization as sp.hartree)."""
        dz = self.dz
        rho = q.E_CH * y * 1e4 / 1e-10
        D = (np.cumsum(rho, axis=0) - 0.5 * rho) * dz * 1e-10
        D -= (np.sum(D / eps[:, None], axis=0) / np.sum(1.0 / eps))[None, :]
        Ef = D / (eps[:, None] * q.EPS0)
        VH = (np.cumsum(Ef, axis=0) - 0.5 * Ef) * dz * 1e-10
        return VH - VH.mean(axis=0, keepdims=True)

    def dhartree_deps(self, nd, n, eps, deps):
        """Directional derivative of sp.hartree with respect to eps along deps (N, q), by complex step."""
        h = 1e-30
        out = np.zeros(deps.shape)
        for j in range(deps.shape[1]):
            if not np.any(deps[:, j]):
                continue
            e = eps + 1j * h * deps[:, j]
            z, dz = self.z, self.dz
            rho = q.E_CH * (nd - n) * 1e4 / 1e-10
            D = (np.cumsum(rho) - 0.5 * rho) * dz * 1e-10
            D = D - np.sum(D / e) / np.sum(1.0 / e)
            Ef = D / (e * q.EPS0)
            phi = -(np.cumsum(Ef) - 0.5 * Ef) * dz * 1e-10
            VH = -phi; VH = VH - VH.mean()
            out[:, j] = VH.imag / h
        return out

    def tangents(self, s):
        """Exact derivatives of the converged solution with respect to all parameters:
        dV (N, p), dE (n, p), dpsi (N, n, p), drho (n, p)."""
        th = s["th"]; N, p = len(self.z), len(th)
        ops = self._state_ops(s)
        dP = self.dprofiles(th)
        eps = s["P"]["eps"]
        # explicit parameter dependence at fixed V_H
        dV0 = dP["V0"] + self.dvext(th)
        dns = np.zeros(p); dns[self.nl + 1] = 1.0
        _, _, _, Nth = self.response(s, ops, dV0, dP["m"], dP["Eg"], dns)
        nd = s["nd"]; n = (s["psi"] ** 2) @ (s["rho"] * th[self.nl + 1])
        dnd = np.zeros((N, p)); dnd[:, self.nl + 1] = nd / th[self.nl + 1]
        if self.dense:
            Hn = self.hartree_matrix(eps)
            # density response to potential: dn/dV (N x N)
            _, _, _, NV = self.response(s, ops, np.eye(N), np.zeros((N, N)), np.zeros((N, N)), np.zeros(N))
            b = Hn @ (dnd - Nth) + self.dhartree_deps(nd, n, eps, dP["eps"])
            K = -Hn @ NV
            dVH = np.linalg.solve(np.eye(N) - K, b)
        else:
            Hop = lambda y: self._hartree_lin(y, eps)       # V_H of a density (cm^-2/A), columns
            b = Hop(dnd - Nth) + self.dhartree_deps(nd, n, eps, dP["eps"])
            zN = np.zeros((N, 1)); z0 = np.zeros(1)

            def mv(x):
                _, _, _, dn = self.response(s, ops, x.reshape(N, 1), zN, zN, z0)
                return x + Hop(dn)[:, 0]                      # (I - K) x with K = -H_n dn/dV
            A = LinearOperator((N, N), matvec=mv, dtype=float)
            dVH = np.zeros((N, p)); self.gmres_iters = []
            for j in range(p):
                if not np.any(b[:, j]):
                    continue
                it = [0]
                x, info = gmres(A, b[:, j], rtol=self.gmres_rtol, atol=0.0, restart=60, maxiter=20,
                                callback=lambda r: it.__setitem__(0, it[0] + 1), callback_type="pr_norm")
                if info != 0:
                    import warnings
                    warnings.warn(f"sens.tangents: GMRES did not converge for parameter {self.names[j]} (info {info})")
                dVH[:, j] = x; self.gmres_iters.append(it[0])
        dV = dV0 + dVH
        dE, dpsi, dnk, _ = self.response(s, ops, dV, dP["m"], dP["Eg"], dns)
        ns = th[self.nl + 1]
        drho = (dnk - np.outer(s["rho"], dns)) / ns
        return dict(dV=dV, dVH=dVH, dE=dE, dpsi=dpsi, drho=drho)


# ---------------- observables ----------------
def observables(model, th, E, psi, rho, hw, gamma=0.010, nkeep=None, eps_depol=13.5, ladder=None, wg=(0.39, 0.51, 3.3, 0.16)):
    """Closed-form observables of a converged state: transition energies, dipoles, Ando-shifted energies, peak
    quantities at photon energy hw: chi2 (complex, pm/V, energy substitution), alpha at w and 2w (1/cm, material,
    n = 3.3), the NL-02 merit M = g(u) |chi2|^2 / (a_w a_2w) and its waveguide form
    M_wg = g(u') |chi2|^2 / (A1 A2) with modal losses A_i = Gamma_i a_i + background_i, u' = 2 A1/A2
    (wg = (Gamma_1, Gamma_2, background_1, background_2 in 1/cm); default: the N1 InP waveguide). Uses the same Loewdin
    orthonormalisation and conventions as qwkit.sp. `ladder` = (0, m, n) selects the reported transitions."""
    z = model.z; dz = model.dz
    x = model._interfaces(th); act = (x[0], x[-1]); zc = 0.5 * (act[0] + act[1])
    if nkeep is None:
        keep = q.localized(z, psi, act)
    else:
        # the nkeep states with the largest localized fraction, in energy order (a fixed count keeps the observables
        # differentiable; taking simply the lowest nkeep states would pick up a delocalized box state when a
        # localized level crosses above it)
        keep = np.sort(np.argsort(-q.localized_fraction(z, psi, act), kind="stable")[:nkeep])
    # Loewdin over all model states (the kept ones plus the box states above them), then keep: orthonormalizing the
    # kept set alone gives its top state spurious dipoles (core.lowdin_localized); model.nstates should exceed the
    # number of kept states by about ten
    S = psi.T @ psi * dz
    w, U = np.linalg.eigh(S)
    psi = psi @ (U @ np.diag(w ** -0.5) @ U.T)
    E, psi, rho = E[keep], psi[:, keep], rho[keep]
    Z = q.dipoles(z - zc, psi, orthonormal=False)
    ns = th[model.nl + 1]
    N = ns / (model.L * 1e-8)
    Et = sp.depol_energies(z, psi, E, rho, ns, eps_r=eps_depol)
    chi = sp.chi2_depol(Et, Z, rho, hw, gamma, N)
    NREF = 3.3; C0 = 2.99792458e8

    def alpha(h):
        s_ = 0.0
        for l in range(len(E)):
            for m in range(l + 1, len(E)):
                s_ += (rho[l] - rho[m]) * (Z[l, m] * 1e-10) ** 2 * gamma / ((Et[m, l] - h) ** 2 + gamma ** 2)
        return h * q.E_CH / 1.054571817e-34 / (NREF * C0 * q.EPS0) * N * 1e6 * q.E_CH * s_ / 100.0   # 1/cm
    a1, a2 = alpha(hw), alpha(2 * hw)
    u = 2 * a1 / a2
    g = u ** (-(u + 1) / (u - 1)) if abs(u - 1) > 1e-9 else np.exp(-2.0)
    G1, G2, b1, b2 = wg
    A1, A2 = G1 * a1 + b1, G2 * a2 + b2
    u2 = 2 * A1 / A2
    g2 = u2 ** (-(u2 + 1) / (u2 - 1)) if abs(u2 - 1) > 1e-9 else np.exp(-2.0)
    out = dict(chi_re=chi.real, chi_im=chi.imag, chi_abs=abs(chi), a1=a1, a2=a2, M=g * abs(chi) ** 2 / (a1 * a2) * 1e-8,
               M_wg=g2 * abs(chi) ** 2 / (A1 * A2) * 1e-8)
    if ladder is not None:
        _, m, n = ladder
        out.update(E1m=E[m] - E[0], E1n=E[n] - E[0], Et1m=Et[m, 0], Et1n=Et[n, 0], z1m=abs(Z[0, m]), zmn=abs(Z[m, n]),
                   z1n=abs(Z[0, n]), d12=Et[m, 0] - hw, d13=Et[n, 0] - 2 * hw)
    return out


def observable_jacobian(model, s, tg, hw, ladder=None, h=1e-4, richardson=True, **kw):
    """d(observables)/d(theta) along the exact state tangents. The closed-form post-processing is differentiated by a
    central difference along the (exact) tangent direction; with richardson=True the steps h and h/2 are combined
    (fourth order), which makes the truncation error negligible (~1e-10 relative) while keeping round-off small.
    The step is relative to the parameter's scale (|theta_j|, floored at the typical scale `kw['scale']` if given)."""
    th = s["th"]; p = len(th)
    ob = lambda t_, E_, p_, r_: observables(model, t_, E_, p_, r_, hw, ladder=ladder, nkeep=kw.get("nkeep"), wg=kw.get("wg", (0.39, 0.51, 3.3, 0.16)))
    base = ob(th, s["E"], s["psi"], s["rho"])
    J = {k: np.zeros(p) for k in base}
    scale = np.abs(th).copy()
    scale[scale == 0] = 1.0
    if "scale" in kw:
        scale = np.maximum(scale, kw["scale"])

    def cd(j, hh):
        e = np.zeros(p); e[j] = hh
        op = ob(th + e, s["E"] + hh * tg["dE"][:, j], s["psi"] + hh * tg["dpsi"][:, :, j], s["rho"] + hh * tg["drho"][:, j])
        om = ob(th - e, s["E"] - hh * tg["dE"][:, j], s["psi"] - hh * tg["dpsi"][:, :, j], s["rho"] - hh * tg["drho"][:, j])
        return {k: (op[k] - om[k]) / (2 * hh) for k in base}
    for j in range(p):
        sc_ = h * scale[j]
        d1 = cd(j, sc_)
        if richardson:
            d2 = cd(j, sc_ / 2)
            for k in base:
                J[k][j] = (4 * d2[k] - d1[k]) / 3
        else:
            for k in base:
                J[k][j] = d1[k]
    return base, J
