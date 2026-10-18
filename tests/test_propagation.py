"""Validation of the coupled-wave propagation: analytic SHG limit, power conservation, QPM factors."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwkit import propagation as pr

class Stub:   # lossless medium with constant chi
    def __init__(self, chi): self.c = chi
    def populations(self, I1, I2): return None
    def chi(self, rho): return self.c
    def alpha(self, rho): return 0.0, 0.0

WG = dict(n1=3.2, n2=3.3, G1=0.4, G2=0.5, O_um=0.05, abg1_cm=0.0, abg2_cm=0.0, t_stack_um=1.5, w_um=15.0,
          Lcoh_um=20.0, lam1_um=10.0)

def test_undepleted_phase_matched_limit():
    chi = 50e-9; P1 = 1e-6; L = 200.0
    r = pr.propagate(Stub(chi), WG, P1, "perfect", L_max_um=L)
    O = WG["O_um"] * 1e6 / (WG["w_um"] * 1e-6)
    eta_n = 2 * np.pi ** 2 * chi ** 2 * O / (WG["n1"] ** 2 * WG["n2"] * pr.C0 * pr.EPS0 * (WG["lam1_um"] * 1e-6) ** 2)
    expect = eta_n * P1 ** 2 * (r["x_um"][-1] * 1e-6) ** 2
    assert abs(r["P2"][-1] - expect) / expect < 1e-3

def test_power_conserved_when_lossless():
    r = pr.propagate(Stub(5e-6), WG, 2.0, "perfect", L_max_um=300.0, steps_per_lcoh=256)
    tot = r["P1"] + r["P2"]
    assert np.max(np.abs(tot - 2.0)) / 2.0 < 1e-6 and r["P2"].max() > 0.1   # strongly depleted yet conserved

def test_qpm_factors():
    chi = 50e-9; P1 = 1e-6; L = 20.0 * 40
    perf = pr.propagate(Stub(chi), WG, P1, "perfect", L_max_um=L)["P2"][-1]
    sign = pr.propagate(Stub(chi), WG, P1, "sign", L_max_um=L)["P2"][-1]
    onoff = pr.propagate(Stub(chi), WG, P1, "onoff", L_max_um=L)["P2"][-1]
    assert abs(sign / perf - (2 / np.pi) ** 2) < 0.01 and abs(onoff / perf - (1 / np.pi) ** 2) < 0.01

def test_two_level_saturation():
    hw = 0.124
    st = dict(E=[0.0, hw], Z=[[0.0, 20.0], [20.0, 0.0]], Et=[[0.0, -hw], [hw, 0.0]], rho=[1.0, 0.0], N=1e17)
    med = pr.Medium(st, hw, gamma=0.01, tau_ps=1.0)
    sig = med.sig1[0, 1]
    Isat = hw * pr.QE / (2 * sig * med.tau)
    rho = med.populations(Isat, 0.0)
    assert abs((rho[0] - rho[1]) - 0.5) < 1e-6
    a0, _ = med.alpha(np.array([1.0, 0.0])); a1, _ = med.alpha(rho)
    assert abs(a1 / a0 - 0.5) < 1e-6

def test_path_reproduces_segments_with_saturation():
    """propagate_path with one medium per step reproduces propagate_segments for an alternating pattern of two
    saturable media with different chi(2), absorption and coherence lengths, and conserves power when lossless."""
    hw = 0.124
    def st(E2, z):
        return dict(E=[0.0, E2, 2 * hw], Z=[[0.0, z, 8.0], [z, 0.0, 22.0], [8.0, 22.0, 0.0]],
                    Et=[[0.0, -E2, -2 * hw], [E2, 0.0, -(2 * hw - E2)], [2 * hw, 2 * hw - E2, 0.0]], rho=[1.0, 0.0, 0.0], N=2e17)
    m1 = pr.Medium(st(hw - 0.012, 18.0), hw, gamma=0.01, tau_ps=1.0)
    m2 = pr.Medium(st(hw + 0.015, 20.0), hw, gamma=0.01, tau_ps=1.0)
    W = dict(WG, abg1_cm=3.0, abg2_cm=1.0)
    L1, L2 = 20.0, 25.0
    ref = pr.propagate_segments([(m1, L1, 1.0), (m2, L2, 1.0)], W, 2.0, L_max_um=6 * (L1 + L2), steps_per_seg=200)
    dx = 0.1
    idx = ([0] * int(round(L1 / dx)) + [1] * int(round(L2 / dx))) * 6
    out = pr.propagate_path([m1, m2], [np.pi / (L1 * 1e-6), np.pi / (L2 * 1e-6)], idx, dx, W, 2.0)
    assert abs(out["P2"][-1] / ref["P2"][-1] - 1) < 1e-5 and abs(out["P1"][-1] / ref["P1"][-1] - 1) < 1e-5
    lossless = pr.propagate_path([Stub(5e-6)], [0.0], [0] * 3000, 0.1, WG, 2.0)
    assert np.max(np.abs(lossless["P1"] + lossless["P2"] - 2.0)) < 1e-6

def test_index_saturation_coefficients():
    """chi1_coeffs . rho equals the direct sum over transitions of the ISB linear susceptibility, and the dk shift
    vanishes at equilibrium and has the sign of the bleached index."""
    hw = 0.124
    st = dict(E=[0.0, 0.11, 0.25], Z=[[0.0, 18.0, 6.0], [18.0, 0.0, 21.0], [6.0, 21.0, 0.0]],
              Et=[[0.0, -0.115, -0.26], [0.115, 0.0, -0.145], [0.26, 0.145, 0.0]], rho=[0.95, 0.05, 0.0], N=2e17)
    med = pr.Medium(st, hw, gamma=0.01, tau_ps=1.0)
    direct = 0.0
    E = np.array(st["Et"]); Z = np.array(st["Z"]); r = np.array(st["rho"])
    for l in range(3):
        for m in range(l + 1, 3):
            direct += ((r[l] - r[m]) * (Z[l, m] * 1e-10) ** 2 * (1 / (E[m, l] - hw - 1j * 0.01) + 1 / (E[m, l] + hw + 1j * 0.01)) / pr.QE).real
    direct *= st["N"] * 1e6 * pr.QE ** 2 / pr.EPS0
    assert abs(med.chi1_coeffs(hw) @ r / direct - 1) < 1e-12
    med.set_index_saturation(dict(WG, G1=0.4, G2=0.5))
    assert med.dk_shift(r) == 0.0
    bleached = np.array([0.6, 0.4, 0.0])
    lam = WG["lam1_um"] * 1e-6
    expect = 4 * np.pi / lam * (0.5 * (med.chi1_coeffs(2 * hw) @ (bleached - r)) / (2 * WG["n2"]) - 0.4 * (med.chi1_coeffs(hw) @ (bleached - r)) / (2 * WG["n1"]))
    assert abs(med.dk_shift(bleached) / expect - 1) < 1e-12
