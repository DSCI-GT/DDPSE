"""Small helpers shared by Topics 1.3-1.5.

``get_solver`` is the solver finder introduced in Topic 1.3. It looks for a solver executable on the PATH,
then in the usual IDAES extension folders, and finally asks Pyomo whether the solver is available. It
returns ``None`` instead of raising when the solver is missing, so a page can print a message and fall
back to scipy.
"""
import os
import shutil

import pyomo.environ as pyo

from .solvers import find_executable, get_solver  # noqa: F401  (merged into _lib/solvers.py)



