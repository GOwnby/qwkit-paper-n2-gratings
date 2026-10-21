"""N2 shared set-up: saturable media of the N1 structures for every bias, the N1 waveguide at each pump detuning,
the N1 electrode geometry, and programmed-grating propagation with ideal (abrupt) or realistic (fringing) fields.

Media: self-consistent state sets with LO-phonon relaxation (chimap.json for ns = 5e11 cm^-2, chimap_I2.json for
1e11), every 5 kV/cm, valid single-period fields only (N1 criterion). Waveguide: N1/NL-02 InP ridge recomputed at each
pump energy (studies/nl02/hw_scan_wg.json, via n1_paper/detuned_qpm.wg_at); each bias has its own coherence length
from its ISB index at w and 2w (detuned_qpm.wg_for), dk = pi / Lcoh.
Electrodes: top-plane electrodes 5.31 um above the ground plane, the QW stack 0.90-2.41 um above ground (N1
electrode_study.py), gaps with linear potential ramps (default 2 um, shorter than every electrode). A field F (kV/cm)
needs V = F d. Neighbouring electrodes may differ by at most E_BD * gap in voltage (lateral breakdown, 0.5 MV/cm).
Index saturation (default on, N2_INDEX_SAT=0 to switch off as in N1): dk follows the populations through the ISB
index at w and 2w (Medium.set_index_saturation).
Realistic fields: F(x) at three heights in the stack (bottom, centre, top; weights 1/4, 1/2, 1/4) from
electrodes.field_profile; each height takes the medium of the nearest valid grid field; chi(2), absorption and the
index (dk) are the weighted means over the heights (each height with its own populations)."""
import sys, os, json
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..")); sys.path.insert(0, os.path.join(here, "..", "n1_paper"))
from qwkit import propagation as pr, electrodes as el
import detuned_qpm as dq

HW0 = 1.23984 / 10
GAM = 0.010
D_UM = 5.312                                         # electrode plane to ground
Y_UM = (0.90, 0.90 + 1.512 / 2, 0.90 + 1.512)        # stack bottom, centre, top above ground
YW = (0.25, 0.5, 0.25)
GAP_UM = 2.0
E_BD = 500.0                                         # kV/cm, lateral breakdown field assumed in the electrode gaps (InP)
INDEX_SAT = os.environ.get("N2_INDEX_SAT", "1") == "1"   # dk follows the populations (ISB index saturation)


def dfmax(gap_um=GAP_UM):
    """Largest nominal-field difference (kV/cm) between neighbouring electrodes: |V_j - V_j+1| <= E_BD * gap."""
    return E_BD * gap_um / D_UM
_MAPS = {4: "chimap.json", 2: "chimap_I2.json"}
_CACHE = {}


def statesets(design, I=4):
    key = ("st", design, I)
    if key not in _CACHE:
        S = json.load(open(os.path.join(here, _MAPS.get(I, f"chimap_I{I}.json"))))
        _CACHE[key] = sorted([s for s in S if s["design"] == design and s["valid"]], key=lambda s: s["F"])
    return _CACHE[key]


class Bank:
    """All valid biases of one structure at one density and pump detuning: media, dk, chi at low intensity."""

    def __init__(self, design, I=4, dhw=0.0):
        self.design, self.I, self.dhw = design, I, dhw
        hw = HW0 + dhw
        self.wg = dq.wg_at(dhw)
        sts = statesets(design, I)
        self.F = np.array([s["F"] for s in sts], float)
        self.media = [pr.Medium(s["st"], hw, gamma=GAM, R=s["R"]) for s in sts]
        if INDEX_SAT:
            for m in self.media:
                m.set_index_saturation(self.wg)
        self.Lcoh = np.array([dq.wg_for(s["st"], hw, self.wg)["Lcoh_um"] for s in sts])
        self.dk = np.pi / (self.Lcoh * 1e-6)
        self.chi0 = np.array([m.chi(m.rho0) for m in self.media])
        self._mix = {}

    def nearest(self, F):
        return int(np.argmin(np.abs(self.F - F)))

    def index_of(self, F):
        i = self.nearest(F)
        assert abs(self.F[i] - F) < 1e-9, F
        return i

    def mix(self, idx3):
        """Medium averaged over the three stack heights (indices into the bank)."""
        idx3 = tuple(int(i) for i in idx3)
        if len(set(idx3)) == 1:
            return idx3[0]
        if idx3 not in self._mix:
            self.media.append(_Mix([self.media[i] for i in idx3], YW))
            self.dk = np.append(self.dk, sum(w * self.dk[i] for w, i in zip(YW, idx3)))
            self._mix[idx3] = len(self.media) - 1
        return self._mix[idx3]


class _Mix:
    def __init__(self, meds, w):
        self.meds, self.w = meds, w

    def populations(self, I1, I2):
        return [m.populations(I1, I2) for m in self.meds]

    def chi(self, rhos):
        return sum(w * m.chi(r) for w, m, r in zip(self.w, self.meds, rhos))

    def alpha(self, rhos):
        a = [m.alpha(r) for m, r in zip(self.meds, rhos)]
        return sum(w * x[0] for w, x in zip(self.w, a)), sum(w * x[1] for w, x in zip(self.w, a))

    def dk_shift(self, rhos):
        return sum(w * m.dk_shift(r) for w, m, r in zip(self.w, self.meds, rhos))


def ideal_steps(bank, fields, lengths_um, dx_um):
    """Step indices for abrupt fields: segment k at field fields[k] (must be a grid field) of length lengths_um[k]."""
    idx = []
    for F, L in zip(fields, np.broadcast_to(lengths_um, (len(fields),))):
        idx += [bank.index_of(F)] * int(round(L / dx_um))
    return idx


def top_potential_segments(x_um, F_seg, L_seg_um, gap_um=GAP_UM):
    """Top potential (V) of consecutive electrodes of lengths L_seg_um at nominal fields F_seg (kV/cm, V = F d), each
    electrode shortened by gap/2 at both ends, linear ramps in the gaps, 0 V beyond the ends."""
    assert gap_um < np.min(L_seg_um), "electrode gap must be shorter than every electrode"
    e = np.concatenate([[0.0], np.cumsum(L_seg_um)])
    V = np.asarray(F_seg, float) * 1e5 * D_UM * 1e-6
    xe = np.concatenate([[e[0] - gap_um / 2], np.ravel(np.column_stack([e[:-1] + gap_um / 2, e[1:] - gap_um / 2])),
                         [e[-1] + gap_um / 2]])
    ve = np.concatenate([[0.0], np.repeat(V, 2), [0.0]])
    return np.interp(x_um, xe, ve, left=0.0, right=0.0)


def realistic_fields(F_seg, L_seg_um, dx_um, L_um, gap_um=GAP_UM):
    """Field (kV/cm) at the three stack heights on x = (k + 1/2) dx, k < L/dx, under consecutive electrodes."""
    n = int(round(L_um / dx_um)); xs = (np.arange(n) + 0.5) * dx_um
    tot = float(np.sum(L_seg_um))
    X = np.arange(-40 * D_UM, tot + 40 * D_UM, dx_um / 2)
    phi = top_potential_segments(X, F_seg, L_seg_um, gap_um)
    return np.array([np.interp(xs, X, el.field_profile(X * 1e-6, phi, D_UM * 1e-6, y * 1e-6) / 1e5) for y in Y_UM])


def realistic_steps(bank, F_seg, L_seg_um, dx_um, L_um=None, gap_um=GAP_UM):
    """Step indices (with three-height mixtures) for consecutive electrodes; L_um defaults to the electrodes plus 3 d."""
    L_seg_um = np.broadcast_to(np.asarray(L_seg_um, float), (len(F_seg),))
    if L_um is None:
        L_um = float(np.sum(L_seg_um)) + 3 * D_UM
    Fy = realistic_fields(F_seg, L_seg_um, dx_um, L_um, gap_um)
    return [bank.mix([bank.nearest(f) for f in Fy[:, k]]) for k in range(Fy.shape[1])]


def propagate(bank, idx, dx_um, P, record=False):
    return pr.propagate_path(bank.media, bank.dk, idx, dx_um, bank.wg, P, record=record)
