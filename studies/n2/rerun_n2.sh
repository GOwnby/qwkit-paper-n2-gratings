#!/bin/bash
# Reproduce every N2 number and figure (about 10 h on 2 cores). Long steps keep finished jobs in parts/ and resume.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=$(cd ../.. && pwd) OMP_NUM_THREADS=1 NPROC=${NPROC:-2}
python3 -W ignore chimap.py > chimap.log                      # 5e11 bias maps
NS_I=2 python3 -W ignore chimap.py > chimap_I2.log            # 1e11 bias maps
N2_I=4 python3 -W ignore sat_main.py > sat_main_I4.log        # design energy, 5e11
N2_I=4 python3 -W ignore ideal_bench.py > ideal_bench_I4.log  # benchmarks under abrupt fields, same search
python3 -W ignore conv_n2.py > conv_n2.log
python3 -W ignore anatomy.py > anatomy.log
python3 -W ignore n1_check.py > n1_check.log                # N1 conventions with real fields
python3 -W ignore sat_tuning.py > sat_tuning.log
python3 -W ignore sat_electrodes.py > sat_electrodes.log
N2_I=2 python3 -W ignore sat_main.py > sat_main_I2.log        # design energy, 1e11
N2_I=2 python3 -W ignore ideal_bench.py > ideal_bench_I2.log
python3 tables.py > /dev/null
python3 -W ignore figs_n2.py
