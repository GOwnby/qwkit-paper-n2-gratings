import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT)

def run_script(desc, script_rel):
    print("=" * 70)
    print("[PROOF STEP] " + desc + " -> " + script_rel)
    print("=" * 70)
    script_path = ROOT / script_rel
    res = subprocess.run([sys.executable, str(script_path)], cwd=script_path.parent, env=ENV)
    if res.returncode != 0:
        print("[FAILED] " + desc, file=sys.stderr)
        sys.exit(res.returncode)
    print("[PASSED] " + desc)

if __name__ == "__main__":
    run_script("Running qwkit 69-test mathematical validation suite", "tests/run_tests.py")
    for s in ['studies/n2/tables.py', 'studies/n2/figs_n2.py']:
        run_script("Verifying N2 study and figures", s)
    print("[SUCCESS] All mathematical identities, bounds, and paper figures verified.")
