"""N2 (1): bias maps of the N1 media for electrically programmable gratings. Self-consistent state sets (LO-phonon
rates, depolarization with eps 13.5, 0.6.2 buffered Loewdin) of the 64/10/29 A double well (ADQW) for
F = -140 ... +140 kV/cm and of the 58/14/34/14/58 A triple well (TW) for F = -100 ... +100 kV/cm, every 5 kV/cm,
at ns = 5e11 cm^-2 (N1 tables.json index 4; NS_I=2 gives 1e11 and chimap_I2.json), with the single-period validity flag of N1 (no partially localized
subband within 0.40 eV of the ground state). Writes chimap.json (state sets usable by propagation.Medium)."""
import sys, os, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "n1_paper")); sys.path.insert(0, os.path.join(here, "..", ".."))
from multiprocessing import Pool
import twofield_qpm as tq

FA = list(range(-140, 145, 5))
FT = list(range(-100, 105, 5))

I = int(os.environ.get("NS_I", "4"))
OUT = "chimap.json" if I == 4 else f"chimap_I{I}.json"

if __name__ == "__main__":
    jobs = [("ad", I, F) for F in FA] + [("tw", I, F) for F in FT]
    with Pool(int(os.environ.get("NPROC", "2"))) as pool:
        out = pool.map(tq.state, jobs, chunksize=1)
    json.dump(out, open(os.path.join(here, OUT), "w"))
    print(len(out), "state sets;", sum(o["valid"] for o in out), "valid")
