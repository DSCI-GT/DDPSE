"""Restart the exercise counter at 1 on every problem-set page.

Why this exists
---------------
`sphinx-exercise` does not number exercises itself. Its `init_numfig` hook turns on
Sphinx's `numfig` machinery and registers `exercise` as an enumerable node type, so the
numbers actually come from `sphinx.environment.collectors.toctree.assign_figure_numbers`,
which keeps **one counter for the whole book**:

    def get_next_fignumber(figtype, secnum):
        counter = fignum_counter.setdefault(figtype, {})
        secnum = secnum[:env.config.numfig_secnum_depth]
        counter[secnum] = counter.get(secnum, 0) + 1
        return secnum + (counter[secnum],)

`secnum` is empty for every page in this book, because nothing in `_toc.yml` is
`:numbered:`, so all 161 exercises share one bucket. That is why the five exercises of a
problem set came out as "Exercise 142" through "Exercise 146" — a number that means
something to the book and nothing at all to a student holding one assignment, who has
Part A questions A1-A3 and Parts B and C in front of them and no idea what 142 counts.

The obvious-looking fix — numbering the toctree and setting `numfig_secnum_depth` — is
not usable here: it would put section numbers on all 27 chapters and restyle every
in-chapter exercise as "Exercise 3.2", a book-wide editorial change nobody asked for.

What it does
------------
Runs immediately after `assign_figure_numbers` and rewrites the `exercise` entries of
each page under `problem_sets/` to 1..N in document order.
`sphinx_exercise.utils.get_node_number` reads exactly this mapping —
`env.toc_fignumbers[docname]["exercise"][node_id]` — for both the HTML and LaTeX
writers, so rewriting it is enough; no directive, transform or template changes.

The hook point is the fiddly part
---------------------------------
`env-updated` is the intuitive choice and is **wrong**: `Builder.build()` calls
`self.read()` (which ends by emitting `env-updated`) and only *then* calls
`env.check_dependents()`, which is what emits `env-get-updated` and runs the collector.
A handler on `env-updated` therefore renumbers a mapping that has not been built yet and
is about to be overwritten — the symptom is a clean build in which nothing changed.

So this connects to `env-get-updated` at **priority 900**. `EventManager.emit` calls
listeners in ascending priority order and the collector registers at the default 500, so
900 is the first thing to run after the numbers exist. Handlers on this event must return
an iterable — `check_dependents` does `to_rewrite.extend(retval)`, which raises on `None`
— and the docnames returned are added to the rewrite list, which is also what guarantees
these pages are re-written on a build where nothing else about them changed.

Scope is deliberately `problem_sets/` only. Chapters keep their continuous 1..136 run:
the request was about problem sets, and chapter exercises are read in book order where a
running count is not confusing. Widening it is a one-line change to `RESTART_UNDER`.

Renumbering is safe because **nothing in the book cross-references an exercise**: there
is not one `{ref}`/`{numref}` to an `ex-` or `pr-` label, nor any prose naming an
exercise by number. Check that again before adding one.

Registered from `_config.yml`:

    sphinx:
      local_extensions:
        exercise_numbering: _ext/
"""

from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)

__version__ = "1.0"

#: Pages whose exercise numbering restarts at 1. Matched with `str.startswith`.
RESTART_UNDER = "problem_sets/"

#: The `numfig` figure type sphinx-exercise registers for the `{exercise}` directive.
FIGTYPE = "exercise"

#: Runs after the toctree collector, which connects to this event at the default 500.
PRIORITY = 900


def restart_exercise_numbers(app: Sphinx, env: BuildEnvironment) -> list[str]:
    touched = []
    for docname, figtypes in env.toc_fignumbers.items():
        if not docname.startswith(RESTART_UNDER):
            continue
        numbers = figtypes.get(FIGTYPE)
        if not numbers:
            continue
        # Sort on the number Sphinx assigned rather than trusting dict insertion order:
        # the assignment walk is document order, so the existing numbers already encode
        # it, and sorting says so explicitly.
        ordered = sorted(numbers, key=lambda node_id: numbers[node_id])
        for n, node_id in enumerate(ordered, start=1):
            numbers[node_id] = (n,)
        touched.append(docname)
        LOGGER.debug(
            "[exercise_numbering] %s: renumbered %d exercise(s) to 1..%d",
            docname, len(ordered), len(ordered),
        )
    return touched


def setup(app: Sphinx):
    app.connect("env-get-updated", restart_exercise_numbers, priority=PRIORITY)
    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
