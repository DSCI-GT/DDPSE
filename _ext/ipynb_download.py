"""Put a `.ipynb` entry in the theme's Download badge for every notebook page.

Why this exists
---------------
sphinx-book-theme already knows how to offer a `.ipynb` download — see
`sphinx_book_theme/header_buttons/__init__.py`, which appends a `.ipynb` button whenever
the page context carries `ipynb_source`:

    if context.get("ipynb_source"):
        download_buttons.append({... "text": ".ipynb" ...})

The catch is *where* `ipynb_source` gets set. The only place is
`header_buttons/launch.py::add_launch_buttons`, and that function returns early unless
`launch_buttons` is configured with at least one of `binderhub_url`, `jupyterhub_url`,
`thebe` or `colab_url`. So on a book with no Binder/Colab integration, pages written as
jupytext MyST markdown never advertise a notebook download — the badge offers only the
`.md` source.

Tying "can I download the notebook" to "do you use Binder" is incidental, not intended.
Enabling `launch_buttons` purely to unlock the download would also add a launch button
whose URL points at `<pagename>.md` in the repository (the extension is only swapped to
`.ipynb` when a sibling `.ipynb` exists on disk, which it does not here), i.e. a broken
link. So this extension sets `ipynb_source` directly and leaves launch buttons alone.

What it does
------------
For every HTML page that is a notebook but whose source is markdown, copy the executed
notebook that MyST-NB already wrote to `_build/jupyter_execute/<pagename>.ipynb` into
`_build/html/_sources/<pagename>.ipynb`, then set `context["ipynb_source"]`. This is the
same copy-and-set that `launch.py` performs; only the trigger differs.

Pages authored as `.ipynb` are skipped — Sphinx already serves their source verbatim.

Priority
--------
`add_header_buttons` is connected at priority 501, so this handler runs at the default
500 and the context variable is in place before the badge is assembled.

Registered from `_config.yml`:

    sphinx:
      local_extensions:
        ipynb_download: _ext/
"""

from pathlib import Path
from shutil import copy2

from sphinx.util import logging

LOGGER = logging.getLogger(__name__)

__version__ = "1.0"


def _is_notebook(app, pagename, context) -> bool:
    """Mirror sphinx-book-theme's own test."""
    metadata = app.env.metadata.get(pagename, {})
    if "kernelspec" in metadata:
        return True
    return "ipynb" in context.get("page_source_suffix", "")


def add_ipynb_download(app, pagename, templatename, context, doctree):
    if getattr(app.builder, "format", "") != "html":
        return
    if context.get("ipynb_source"):
        return  # launch.py already handled it; do not duplicate the copy
    if not _is_notebook(app, pagename, context):
        return

    sourcename = context.get("sourcename", "")
    if not (sourcename.endswith(".md") or sourcename.endswith(".md.txt")):
        # An .ipynb-sourced page already downloads as a notebook.
        return

    out_dir = Path(app.outdir)
    executed = out_dir.parent / "jupyter_execute" / f"{pagename}.ipynb"
    if not executed.exists():
        LOGGER.debug(
            "[ipynb_download] no executed notebook for %s at %s", pagename, executed
        )
        return

    destination = out_dir / "_sources" / f"{pagename}.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(executed, destination)
    context["ipynb_source"] = f"{pagename}.ipynb"


def setup(app):
    app.connect("html-page-context", add_ipynb_download, priority=500)
    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
