.. Kosmic Lab documentation master file

Welcome to Kosmic Lab's Documentation!
======================================

**Kosmic Lab** is an AI-accelerated research platform for consciousness modeling,
bioelectric simulation, and multi-universe exploration. This platform combines
cutting-edge AI with rigorous scientific methodology to advance consciousness research.

.. image:: https://img.shields.io/badge/Python-3.10%2B-blue
   :alt: Python Version

.. image:: https://img.shields.io/badge/Code%20Quality-A+-brightgreen
   :alt: Code Quality

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :alt: Code Style: Black

Quick Navigation
----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation
   examples

.. toctree::
   :maxdepth: 2
   :caption: Core Documentation

   architecture
   modules/index
   api/index

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/k_index
   guides/k_codex
   guides/bioelectric
   guides/simulation

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   development
   testing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Reference

   troubleshooting
   glossary
   references

What is Kosmic Lab?
-------------------

Kosmic Lab provides:

* **K-Index Metrics**: Novel correlation-based metrics for consciousness research
* **K-Codex System**: Comprehensive reproducibility tracking
* **Bioelectric Simulation**: Based on Michael Levin's morphogenetic field theory
* **Multi-Universe Framework**: Parallel exploration of parameter spaces
* **FRE Analysis**: Free Energy Rescue mechanisms for consciousness collapse
* **AI Integration**: LLM-powered experiment design and analysis

Key Features
------------

Reproducibility First
~~~~~~~~~~~~~~~~~~~~~

Every experiment generates a K-Codex record containing:

- Git SHA for code version
- Full parameter configuration
- Random seeds for determinism
- Execution environment details
- Metrics and outcomes

Scientific Rigor
~~~~~~~~~~~~~~~~

- Type-safe with mypy
- 100% test coverage goal
- CI/CD pipeline with quality gates
- Performance benchmarking
- Comprehensive logging

Research-Ready
~~~~~~~~~~~~~~

- Publication-quality visualizations
- Bootstrap confidence intervals
- Statistical power analysis
- Multi-universe parameter sweeps
- Real-time dashboard monitoring

Quick Start Example
-------------------

.. code-block:: python

   from core.logging_config import setup_logging, get_logger
   from core.kcodex import KCodexWriter
   from fre.metrics.k_index import k_index
   import numpy as np

   # Setup
   setup_logging(level="INFO")
   logger = get_logger(__name__)

   # Generate data
   observed = np.abs(np.random.randn(100))
   actual = np.abs(np.random.randn(100))

   # Compute K-Index
   k = k_index(observed, actual)
   logger.info(f"K-Index: {k:.4f}")

   # Track with K-Codex
   kcodex = KCodexWriter("experiment.json")
   kcodex.log_experiment(
       experiment_name="quick_start",
       params={"n_samples": 100},
       metrics={"k_index": k},
       seed=42
   )

See :doc:`examples` for more comprehensive tutorials.

Main Components
---------------

Core Module
~~~~~~~~~~~

- **logging_config**: Centralized logging system
- **kcodex**: Reproducibility tracking (K-Codex)
- **bioelectric**: Bioelectric circuit simulation
- **kpass**: Multi-universe passage tracking
- **utils**: Shared utilities

FRE Module (Free Energy Rescue)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **metrics.k_index**: K-Index computation and bootstrap CIs
- **metrics.k_lag**: Temporal lag analysis
- **simulate**: Universe simulation engine
- **rescue**: Bioelectric rescue mechanisms
- **analyze**: Batch experiment analysis

Scripts
~~~~~~~

- **kosmic_dashboard.py**: Real-time monitoring dashboard
- **ai_experiment_designer.py**: AI-powered experiment design
- **fre_analyzer.py**: Batch FRE analysis

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Getting Help
============

- **Troubleshooting**: See :doc:`troubleshooting`
- **Examples**: See :doc:`examples`
- **API Reference**: See :doc:`api/index`
- **Contributing**: See :doc:`contributing`

License
=======

Kosmic Lab is open source. See the LICENSE file for details.

.. note::
   This documentation is auto-generated from source code docstrings and
   updated with each release. For the latest development version, see
   the GitHub repository.
