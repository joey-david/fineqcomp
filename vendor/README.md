# Vendored IFEval scorer

The Jean-Zay module stack cannot install packages from PyPI on the login path.
This directory carries the pinned IFEval scorer and its small pure-Python
dependency so batch jobs use the same scorer without a network install.
