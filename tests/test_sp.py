"""Validation of the self-consistent electrostatics and depolarisation."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwkit import sp, core as q

def test_sheet_charges_analytic():
    # +ns/2 sheets at both period edges, -ns sheet at the centre, uniform eps:
    # potential energy difference edge-to-centre = e ns L / (4 eps eps0)   (periodic, zero-mean field)
    L, dz, ns = 400.0, 0.5, 5e11
    z = (np.arange(int(L / dz)) + 0.5) * dz
    nd = np.zeros_like(z); n = np.zeros_like(z)
    nd[0] = nd[-1] = ns / 2 / dz
    n[len(z) // 2 - 1] = n[len(z) // 2] = ns / 2 / dz
    eps = np.full_like(z, 13.0)
    V = sp.hartree(z, nd, n, eps)
    dV_num = V[len(z) // 2] - V[0]
    dV_th = q.E_CH * ns * 1e4 * L * 1e-10 / (4 * 13.0 * q.EPS0)
    assert abs(abs(dV_num) - dV_th) / dV_th < 0.01, (dV_num, dV_th)
    assert dV_num > 0  # electron potential energy is higher at the (negatively charged) electron sheet

def test_symmetry_exact():
    TW = [(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")]
    hw = 1.23984 / 10
    a = sp.chi2_sc(TW, 40.0, hw, dz=1.0)["chi"]; b = sp.chi2_sc(TW, -40.0, hw, dz=1.0)["chi"]
    c = sp.chi2_sc(TW, 0.0, hw, dz=1.0)["chi"]
    assert abs(a + b) < 1e-6 * abs(a) and abs(c) < 1e-6 * abs(a)

def test_neutral_and_converged():
    TW = [(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")]
    r = sp.self_consistent(TW, 50.0, dz=1.0)
    dz = r["z"][1] - r["z"][0]
    n = (r["psi"] ** 2) @ (q.populations(r["E"], r["ns"], r["T"]) * r["ns"])
    assert abs(n.sum() * dz - r["nd"].sum() * dz) / r["ns"] < 1e-6
    assert r["err"] < 1e-5

def test_depolarisation_blue_shift():
    # A populated ground state must blue-shift its absorption (alpha > 0) and vanish as ns -> 0
    ad = [(64, "well"), (10, "barrier"), (30, "well")]
    hw = 1.23984 / 10
    hi = sp.chi2_sc(ad, 0.0, hw, ns_cm2=5e11, dz=1.0, sc=False)
    lo = sp.chi2_sc(ad, 0.0, hw, ns_cm2=1e9, dz=1.0, sc=False)
    d_hi = hi["Et"][1, 0] - (hi["E"][1] - hi["E"][0]); d_lo = lo["Et"][1, 0] - (lo["E"][1] - lo["E"][0])
    assert d_hi > 0 and d_lo < 0.01 * d_hi
