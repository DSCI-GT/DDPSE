"""One place to find optimization solvers (merged 2026-10-30 from five per-part copies).

Search order for solvers that run as executables (ipopt, cbc, bonmin, couenne):
the PATH; then $IDAES_EXT; then ~/.idaes/bin (where `idaes get-extensions` installs them);
then $CONDA_PREFIX/idaes-ext. Solvers with a Python interface (names starting with 'appsi_',
e.g. 'appsi_highs') need no executable. Nothing here is specific to one machine: point
$IDAES_EXT at your solver folder if it lives somewhere else (the book's makefile does).
"""
import os
import shutil


def _dirs():
    out = [os.environ.get("IDAES_EXT", ""), os.path.expanduser("~/.idaes/bin")]
    if os.environ.get("CONDA_PREFIX"):
        out.append(os.path.join(os.environ["CONDA_PREFIX"], "idaes-ext"))
    return [d for d in out if d]


def find_executable(name):
    """Return the path of solver executable `name`, or None."""
    exe = shutil.which(name)
    if exe:
        return exe
    for d in _dirs():
        path = os.path.join(d, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def get_solver(name, message=False):
    """Return a ready Pyomo solver object for `name`, or None if it cannot run here."""
    import pyomo.environ as pyo
    if name.startswith("appsi_"):
        solver = pyo.SolverFactory(name)
    else:
        exe = find_executable(name)
        solver = pyo.SolverFactory(name, executable=exe) if exe else pyo.SolverFactory(name)
    try:
        ok = solver.available(exception_flag=False)
    except Exception:
        ok = False
    if not ok and message:
        print(f"[solver] {name} not found; using the fallback instead.")
    return solver if ok else None
