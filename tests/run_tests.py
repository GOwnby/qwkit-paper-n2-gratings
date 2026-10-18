"""Minimal test runner (no pytest needed): python3 tests/run_tests.py"""
import importlib.util, glob, os, sys, traceback, time
here = os.path.dirname(os.path.abspath(__file__))
fails = 0; n = 0
for path in sorted(glob.glob(os.path.join(here, "test_*.py"))):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            n += 1; t = time.time()
            try:
                getattr(mod, name)(); print(f"PASS {os.path.basename(path)}::{name} ({time.time()-t:.1f}s)")
            except Exception:
                fails += 1; print(f"FAIL {os.path.basename(path)}::{name}"); traceback.print_exc()
print(f"{n - fails}/{n} passed"); sys.exit(1 if fails else 0)
