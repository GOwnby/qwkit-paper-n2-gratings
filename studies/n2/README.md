# N2: electrically programmable intersubband chi(2) gratings

- `common_n2.py`  media for every bias (chimap*.json), waveguide per pump energy, electrode fields (fringing), propagation
- `synth.py`      pattern synthesis: greedy with pump value mu, coordinate ascent (ideal and realistic fields)
- `sat_main.py`   design energy: uniform, N1 two-field / TW sign benchmarks, programmed patterns (N2_I=4 or 2)
- `ideal_bench.py` the same benchmarks under abrupt (ideal) fields, optimized with the same search (N2_I=4 or 2)
- `sat_tuning.py` tuning by reprogramming vs retuning a fixed device (1.26 W, 5e11)
- `sat_electrodes.py` electrode pitch, number of voltage levels, 1 um gaps, no breakdown limit
- `n1_check.py`   N1's schemes in N1's conventions with real electrode fields (LATER_STAGE 51)
- `conv_n2.py`, `anatomy.py`  numerical checks; in-phase chi(2), pump left, field range
- `tables.py`     paper table and numbers.log; `figs_n2.py` figures; `rerun_n2.sh` everything; `parts.py` resumable runs
  (parts/<tag>/<job>.json per finished job; sat_main.py also checkpoints stages as <job>.<stage>.json)
- Groundwork (small signal, superseded by the saturated studies, kept for the record): `grating.py`, `tuning.py`
- `pre_review/`  outputs of the first draft, before the first review round (kept for audit)
