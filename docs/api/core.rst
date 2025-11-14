Core Module
===========

The core module provides fundamental infrastructure for Kosmic Lab.

.. contents:: Table of Contents
   :local:
   :depth: 2

Logging Configuration
---------------------

.. automodule:: core.logging_config
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example usage:

.. code-block:: python

   from core.logging_config import setup_logging, get_logger

   # Setup logging once at application start
   setup_logging(
       level="INFO",
       log_file="logs/my_experiment.log",
       colored=True
   )

   # Get logger in any module
   logger = get_logger(__name__)
   logger.info("Experiment started")

K-Codex System
--------------

.. automodule:: core.kcodex
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The K-Codex system provides comprehensive reproducibility tracking.

Example usage:

.. code-block:: python

   from core.kcodex import KCodexWriter

   kcodex = KCodexWriter("logs/experiment.json")
   kcodex.log_experiment(
       experiment_name="my_experiment",
       params={"learning_rate": 0.01, "epochs": 100},
       metrics={"accuracy": 0.95, "loss": 0.05},
       seed=42,
       extra_metadata={"notes": "Baseline run"}
   )

Bioelectric Circuits
--------------------

.. automodule:: core.bioelectric
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Bioelectric simulation inspired by Michael Levin's work.

Example usage:

.. code-block:: python

   from core.bioelectric import BioelectricCircuit

   circuit = BioelectricCircuit(n_cells=100, resting_voltage=-70.0)
   circuit.apply_stimulus(0.5)
   circuit.step()
   voltages = circuit.get_voltages()

K-Pass (Multi-Universe)
------------------------

.. automodule:: core.kpass
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Multi-universe passage tracking.

Shared Utilities
----------------

.. automodule:: core.utils
   :members:
   :undoc-members:
   :show-inheritance:

Common utilities used throughout Kosmic Lab.

Example usage:

.. code-block:: python

   from core.utils import (
       infer_git_sha,
       hash_config,
       bootstrap_confidence_interval
   )

   # Get current git commit
   sha = infer_git_sha()

   # Hash configuration for reproducibility
   config_hash = hash_config({"param1": 1, "param2": 2})

   # Bootstrap CI for any statistic
   import numpy as np
   data = np.random.randn(100)
   ci_lower, ci_upper = bootstrap_confidence_interval(
       data,
       np.mean,
       n_bootstrap=1000,
       confidence_level=0.95
   )
