# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

"""Compatibility alias for the renamed derivative-structure summary command."""

from clusterx.cli.summarize_derivative_structures import summarize_derivative_structures

commands = ["summarize_dsgenerator"]

# Preserve the public CLI command and Python import used by existing scripts.
summarize_dsgenerator = summarize_derivative_structures
