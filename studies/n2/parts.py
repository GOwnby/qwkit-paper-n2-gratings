"""Resumable job runner: each job's result is stored in parts/<tag>/<key>.json as soon as it finishes, and finished
jobs are skipped when the script is run again (long runs can be split across several sessions)."""
import os, json
from multiprocessing import Pool
here = os.path.dirname(os.path.abspath(__file__))


def _key(args):
    return "_".join(str(a) for a in args).replace("/", "-")


def _run(item):
    fn, tag, args = item
    path = os.path.join(here, "parts", tag, _key(args) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    out = fn(args)
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w")); os.replace(tmp, path)
    return out


def run(fn, jobs, tag, nproc=2):
    os.makedirs(os.path.join(here, "parts", tag), exist_ok=True)
    with Pool(nproc) as pool:
        return pool.map(_run, [(fn, tag, a) for a in jobs], chunksize=1)
