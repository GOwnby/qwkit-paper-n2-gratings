"""
propagation.py -- second-harmonic generation in an intersubband waveguide with saturation, pump depletion
and quasi-phase matching (generalises Eq. 15.29 of Vurgaftman, Lumb & Meyer).

Mode amplitudes a1 (pump, w) and a2 (SH, 2w) with |a|^2 = power (W):
    da1/dx = -(alpha1/2) a1 - i K s(x) chi* a2 a1* exp(-i dk x)
    da2/dx = -(alpha2/2) a2 - i K s(x) chi  a1^2  exp(+i dk x)
    K = (pi/lambda1) sqrt(2 O / (n1^2 n2 c eps0)),  O = |int_QW f1^2 f2* dz|^2 / w   (m^-2)
(lossless limit conserves power exactly; undepleted, lossless, phase-matched limit gives
 P2 = eta_n P1^2 L^2 with eta_n = 2 pi^2 chi^2 O / (n1^2 n2 c eps0 lambda1^2)).
Saturation: steady-state rate equations for all subband populations rho,
    0 = sum_l W_lm (rho_l - rho_m) - (rho_m - rho0_m)/tau          (single relaxation time), or
    0 = sum_l W_lm (rho_l - rho_m) + [L (rho - rho0)]_m,  L_ml = R_lm - delta_ml sum_k R_mk
                                                                     (level-resolved rates R, 1/s),
    W_lm = sigma_lm(w) I1/(hbar w) + sigma_lm(2w) I2/(2 hbar w),  I_j = P_j Gamma_j / (t_stack w)
chi(2) and the ISB absorption are linear in rho, so both are precomputed as coefficient vectors.
Modal loss: alpha_j = Gamma_j * alpha_ISB,j(rho) + alpha_bg,j.
QPM patterns s(x): 'none' (1, dk != 0), 'perfect' (1, dk = 0), 'sign' (+-eta_e), 'onoff' (1/0, ideal).
"""
import numpy as np
from . import sp

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12; C0 = 2.99792458e8; HBAR = 1.054571817e-34


class Medium:
    """Intersubband medium from a state set (E [eV], Z [A], Et [eV], rho0, N [cm^-3])."""

    def __init__(self, st, hw, gamma=0.010, tau_ps=1.0, nref=3.3, R=None):
        """R: optional relaxation matrix R[l, m] (1/s, rate from l to m, e.g. scattering.rate_matrix).
        If given, it replaces the single relaxation time: 0 = optical terms + L(rho - rho0)."""
        self.R = None if R is None else np.array(R, dtype=float)
        self.nref = nref
        self.E = np.array(st["E"]); self.Z = np.array(st["Z"]); self.Et = np.array(st["Et"])
        self.rho0 = np.array(st["rho"]); self.N = st["N"]; self.hw = hw; self.g = gamma
        self.tau = tau_ps * 1e-12; n = len(self.E)
        # chi(rho) = A . rho  (pm/V), alpha_ISB(rho) at w and 2w = B . rho (1/m)
        self.A = np.array([sp.chi2_depol(self.Et, self.Z, np.eye(n)[j], hw, gamma, self.N) for j in range(n)])
        self.sig1 = self._sigma(hw, nref); self.sig2 = self._sigma(2 * hw, nref)
        N = self.N * 1e6
        self.B1 = np.array([N * sum((1.0 if j == l else 0.0) * self.sig1[l, m] - (1.0 if j == m else 0.0) * self.sig1[l, m]
                                    for l in range(n) for m in range(l + 1, n)) for j in range(n)])
        self.B2 = np.array([N * sum((1.0 if j == l else 0.0) * self.sig2[l, m] - (1.0 if j == m else 0.0) * self.sig2[l, m]
                                    for l in range(n) for m in range(l + 1, n)) for j in range(n)])

    def _sigma(self, hw, nref=3.3):
        """Per-electron absorption cross-section sigma_lm (m^2), Lorentzian, depolarisation-shifted."""
        n = len(self.E); s = np.zeros((n, n))
        w = hw * QE / HBAR
        for l in range(n):
            for m in range(l + 1, n):
                Em = self.Et[m, l]
                L = (self.g / np.pi) / ((Em - hw) ** 2 + self.g ** 2) / QE
                s[l, m] = s[m, l] = np.pi * w * QE ** 2 * (self.Z[l, m] * 1e-10) ** 2 * L / (nref * C0 * EPS0)
        return s

    def populations(self, I1, I2):
        """Steady-state populations for local intensities I1 (w) and I2 (2w) in W/m^2."""
        W = self.sig1 * I1 / (self.hw * QE) + self.sig2 * I2 / (2 * self.hw * QE)
        if self.R is None:
            M = W - np.diag(W.sum(axis=1)) - np.eye(len(self.E)) / self.tau
            return np.linalg.solve(M, -self.rho0 / self.tau)
        # level-resolved relaxation: (dρ/dt)_m = sum_l R_lm ρ_l - ρ_m sum_l R_ml, applied to (ρ - ρ0)
        if not hasattr(self, "_L"):
            self._L = self.R.T - np.diag(self.R.sum(axis=1))
            self._b = self._L @ self.rho0; self._b[0] = self.rho0.sum()
        M = W - np.diag(W.sum(axis=1)) + self._L
        M[0, :] = 1.0                                              # replace one equation by sum(ρ) = const
        return np.linalg.solve(M, self._b)

    def chi1_coeffs(self, hw):
        """Real part of the ISB linear susceptibility at photon energy hw (eV) per unit population of each subband:
        Re chi1(rho) = C . rho (dimensionless), with the depolarization-shifted energies and linewidth gamma."""
        n = len(self.E); N = self.N * 1e6; C = np.zeros(n)
        for l in range(n):
            for m in range(l + 1, n):
                Em = self.Et[m, l]
                v = ((self.Z[l, m] * 1e-10) ** 2 * (1 / (Em - hw - 1j * self.g) + 1 / (Em + hw + 1j * self.g)) / QE).real
                C[l] += v; C[m] -= v
        return N * QE ** 2 / EPS0 * C

    def dk_shift(self, rho):
        """Change of the phase mismatch (1/m) from the equilibrium value when the populations are rho; zero unless
        the coefficient vector dkc (1/m per unit population) has been set, e.g. by set_index_saturation."""
        dkc = getattr(self, "dkc", None)
        return 0.0 if dkc is None else float(dkc @ (np.asarray(rho) - self.rho0))

    def set_index_saturation(self, wg):
        """Let the phase mismatch follow the populations through the ISB index at w and 2w:
        dk(rho) = dk(rho0) + (4 pi / lambda1) [G2 Re chi1(2w, rho - rho0) / (2 n2) - G1 Re chi1(w, rho - rho0) / (2 n1)]."""
        lam1 = wg["lam1_um"] * 1e-6
        self.dkc = 4 * np.pi / lam1 * (wg["G2"] * self.chi1_coeffs(2 * self.hw) / (2 * wg["n2"])
                                       - wg["G1"] * self.chi1_coeffs(self.hw) / (2 * wg["n1"]))
        return self

    def chi(self, rho):
        """chi(2) (m/V, complex) for populations rho."""
        return (self.A @ rho) * 1e-12                       # m/V

    def alpha(self, rho):
        """Material ISB absorption (1/m) at w and 2w for populations rho (multiply by the confinement
        factor for the modal value)."""
        return self.B1 @ rho, self.B2 @ rho                  # 1/m


def propagate(med, wg, P1, scheme, eta_e=1.0, L_max_um=5000.0, steps_per_lcoh=32, dk_override=None):
    """Integrate along x. wg: dict(n1, n2, G1, G2, O_um, abg1_cm, abg2_cm, t_stack_um, w_um, Lcoh_um, lam1_um).
    Returns dict with x (um), P1, P2 arrays and the maximum conversion efficiency P2/P1(0)."""
    lam1 = wg["lam1_um"] * 1e-6
    O = wg["O_um"] * 1e6 / (wg["w_um"] * 1e-6)              # m^-2
    K = (np.pi / lam1) * np.sqrt(2 * O / (wg["n1"] ** 2 * wg["n2"] * C0 * EPS0))
    Lc = wg["Lcoh_um"] * 1e-6
    dk = 0.0 if scheme == "perfect" else (np.pi / Lc if dk_override is None else dk_override)
    area = wg["t_stack_um"] * 1e-6 * wg["w_um"] * 1e-6
    abg1, abg2 = wg["abg1_cm"] * 100, wg["abg2_cm"] * 100
    dx = Lc / steps_per_lcoh
    nstep = int(L_max_um * 1e-6 / dx)

    def s_of(x):
        if scheme in ("none", "perfect"):
            return 1.0
        seg = int(np.floor(x / Lc)) % 2
        if scheme == "sign":
            return eta_e * (1.0 if seg == 0 else -1.0)
        if scheme == "onoff":
            return 1.0 if seg == 0 else 0.0
        raise ValueError(scheme)

    def rhs(x, a):
        a1, a2 = a
        I1 = abs(a1) ** 2 * wg["G1"] / area; I2 = abs(a2) ** 2 * wg["G2"] / area
        rho = med.populations(I1, I2)
        chi = med.chi(rho); al1, al2 = med.alpha(rho)
        A1 = wg["G1"] * al1 + abg1; A2 = wg["G2"] * al2 + abg2
        s = s_of(x)
        d1 = -0.5 * A1 * a1 - 1j * K * s * np.conj(chi) * a2 * np.conj(a1) * np.exp(-1j * dk * x)
        d2 = -0.5 * A2 * a2 - 1j * K * s * chi * a1 ** 2 * np.exp(1j * dk * x)
        return np.array([d1, d2])

    a = np.array([np.sqrt(P1) + 0j, 0j]); x = 0.0
    xs = [0.0]; p1 = [P1]; p2 = [0.0]
    for _ in range(nstep):
        # midpoint-of-segment RK4 (s is piecewise constant; segment edges fall on step boundaries)
        eps = 1e-6 * dx                  # evaluate the end points inside the current segment
        k1 = rhs(x + eps, a); k2 = rhs(x + dx / 2, a + dx / 2 * k1)
        k3 = rhs(x + dx / 2, a + dx / 2 * k2); k4 = rhs(x + dx - eps, a + dx * k3)
        a = a + dx / 6 * (k1 + 2 * k2 + 2 * k3 + k4); x += dx
        xs.append(x * 1e6); p1.append(abs(a[0]) ** 2); p2.append(abs(a[1]) ** 2)
    p2 = np.array(p2); i = int(np.argmax(p2))
    return dict(x_um=np.array(xs), P1=np.array(p1), P2=p2, eta=p2[i] / P1, L_opt_um=xs[i], K=K)


def propagate_segments(segs, wg, P1, L_max_um=5000.0, steps_per_seg=32):
    """Alternating media, e.g. realistic Stark on/off QPM. segs: list of (Medium, Lcoh_um, s) cycled along x;
    each segment has length Lcoh_j (one half-period of phase, pi = dk_j * Lcoh_j) and its own chi(2),
    absorption and phase mismatch dk_j = pi / Lcoh_j. The relative phase exp(i theta(x)) accumulates
    continuously across segments. Other waveguide quantities (n, Gamma, overlap, background loss) from wg."""
    lam1 = wg["lam1_um"] * 1e-6
    O = wg["O_um"] * 1e6 / (wg["w_um"] * 1e-6)
    K = (np.pi / lam1) * np.sqrt(2 * O / (wg["n1"] ** 2 * wg["n2"] * C0 * EPS0))
    area = wg["t_stack_um"] * 1e-6 * wg["w_um"] * 1e-6
    abg1, abg2 = wg["abg1_cm"] * 100, wg["abg2_cm"] * 100

    def rhs(med, s, a, theta):
        a1, a2 = a
        I1 = abs(a1) ** 2 * wg["G1"] / area; I2 = abs(a2) ** 2 * wg["G2"] / area
        rho = med.populations(I1, I2)
        chi = med.chi(rho); al1, al2 = med.alpha(rho)
        A1 = wg["G1"] * al1 + abg1; A2 = wg["G2"] * al2 + abg2
        d1 = -0.5 * A1 * a1 - 1j * K * s * np.conj(chi) * a2 * np.conj(a1) * np.exp(-1j * theta)
        d2 = -0.5 * A2 * a2 - 1j * K * s * chi * a1 ** 2 * np.exp(1j * theta)
        return np.array([d1, d2])

    a = np.array([np.sqrt(P1) + 0j, 0j]); x = 0.0; theta = 0.0
    xs = [0.0]; p1 = [P1]; p2 = [0.0]; j = 0
    while x < L_max_um * 1e-6 * (1 - 1e-9):
        med, Lc, s = segs[j % len(segs)]
        Lc = Lc * 1e-6; dk = np.pi / Lc; dx = Lc / steps_per_seg
        for _ in range(steps_per_seg):
            k1 = rhs(med, s, a, theta); k2 = rhs(med, s, a + dx / 2 * k1, theta + dk * dx / 2)
            k3 = rhs(med, s, a + dx / 2 * k2, theta + dk * dx / 2); k4 = rhs(med, s, a + dx * k3, theta + dk * dx)
            a = a + dx / 6 * (k1 + 2 * k2 + 2 * k3 + k4); x += dx; theta += dk * dx
            xs.append(x * 1e6); p1.append(abs(a[0]) ** 2); p2.append(abs(a[1]) ** 2)
        j += 1
    p2 = np.array(p2); i = int(np.argmax(p2))
    return dict(x_um=np.array(xs), P1=np.array(p1), P2=p2, eta=p2[i] / P1, L_opt_um=xs[i], K=K)


def propagate_path(media, dks, idx, dx_um, wg, P1, s=None, a0=None, theta0=0.0, record=True):
    """Propagate along an arbitrary sequence of media, one per step: step k (length dx_um) uses media[idx[k]]
    with phase mismatch dks[idx[k]] (1/m) and coupling scale s[k] (default 1). Used for electrically programmed
    gratings, where the local bias (hence chi(2), absorption, saturation and dk) can change at every step.
    The relative phase theta = int dk dx accumulates continuously, as in propagate_segments; media with a dk_shift
    method (Medium.set_index_saturation) add the population-dependent part of dk (ISB index). a0: starting
    amplitudes (a1, a2) (default sqrt(P1), 0); P1 is the input pump power used to normalize eta.
    Returns dict(x_um, P1, P2, amp (complex a1, a2), theta_x (arrays if record), eta = max P2 / P1, L_opt_um, a, theta)
    with a, theta at the end."""
    lam1 = wg["lam1_um"] * 1e-6
    O = wg["O_um"] * 1e6 / (wg["w_um"] * 1e-6)
    K = (np.pi / lam1) * np.sqrt(2 * O / (wg["n1"] ** 2 * wg["n2"] * C0 * EPS0))
    area = wg["t_stack_um"] * 1e-6 * wg["w_um"] * 1e-6
    abg1, abg2 = wg["abg1_cm"] * 100, wg["abg2_cm"] * 100
    G1, G2 = wg["G1"], wg["G2"]
    dx = dx_um * 1e-6

    def rhs(med, sc, dk, y):
        a1, a2, theta = y[0], y[1], y[2].real
        rho = med.populations(abs(a1) ** 2 * G1 / area, abs(a2) ** 2 * G2 / area)
        chi = med.chi(rho); al1, al2 = med.alpha(rho)
        A1 = G1 * al1 + abg1; A2 = G2 * al2 + abg2
        e = np.exp(1j * theta)
        sh = med.dk_shift(rho) if hasattr(med, "dk_shift") else 0.0
        return np.array([-0.5 * A1 * a1 - 1j * K * sc * np.conj(chi) * a2 * np.conj(a1) / e,
                         -0.5 * A2 * a2 - 1j * K * sc * chi * a1 ** 2 * e, dk + sh + 0j])

    a = np.array([np.sqrt(P1) + 0j, 0j]) if a0 is None else np.array(a0, dtype=complex)
    y = np.array([a[0], a[1], theta0 + 0j])
    p1 = [abs(y[0]) ** 2]; p2 = [abs(y[1]) ** 2]; best = (p2[0], 0); amp = [y[:2].copy()]; ths = [theta0]
    for k, i in enumerate(idx):
        med = media[i]; dk = dks[i]; sc = 1.0 if s is None else s[k]
        k1 = rhs(med, sc, dk, y); k2 = rhs(med, sc, dk, y + dx / 2 * k1)
        k3 = rhs(med, sc, dk, y + dx / 2 * k2); k4 = rhs(med, sc, dk, y + dx * k3)
        y = y + dx / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        P2 = abs(y[1]) ** 2
        if P2 > best[0]:
            best = (P2, k + 1)
        if record:
            p1.append(abs(y[0]) ** 2); p2.append(P2); amp.append(y[:2].copy()); ths.append(y[2].real)
    a = y[:2]; theta = y[2].real
    out = dict(eta=best[0] / P1, L_opt_um=best[1] * dx_um, a=a, theta=theta, K=K)
    if record:
        out.update(x_um=np.arange(len(p1)) * dx_um, P1=np.array(p1), P2=np.array(p2), amp=np.array(amp), theta_x=np.array(ths))
    return out
