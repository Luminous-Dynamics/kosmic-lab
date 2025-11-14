API Reference
=============

This section contains the complete API reference for Kosmic Lab, automatically
generated from source code docstrings.

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   core
   fre
   scripts

Module Organization
-------------------

The API is organized into three main sections:

1. **Core**: Fundamental infrastructure (logging, K-Codex, bioelectric, utils)
2. **FRE**: Free Energy Rescue system (metrics, simulation, rescue, analysis)
3. **Scripts**: User-facing tools (dashboard, AI designer, analyzers)

Usage Examples
--------------

Quick imports for common tasks:

.. code-block:: python

   # Logging
   from core.logging_config import setup_logging, get_logger

   # K-Codex reproducibility
   from core.kcodex import KCodexWriter

   # Bioelectric simulation
   from core.bioelectric import BioelectricCircuit

   # Metrics
   from fre.metrics.k_index import k_index, bootstrap_k_ci
   from fre.metrics.k_lag import k_lag

   # Simulation
   from fre.simulate import UniverseSimulator, compute_metrics

   # Rescue mechanisms
   from fre.rescue import apply_bioelectric_rescue

See individual module documentation for detailed API information.

Conventions
-----------

**Type Hints**
  All functions use comprehensive type hints. See PEP 484.

**Docstring Format**
  Google-style docstrings with Args, Returns, Raises sections.

**Error Handling**
  Exceptions are documented in Raises section. Check before use.

**Logging**
  Most functions log at INFO level. Configure with setup_logging().

**Reproducibility**
  Random operations accept seed parameter. Use K-Codex for tracking.
