FRE Module (Free Energy Rescue)
================================

The FRE module implements the Free Energy Rescue system for consciousness
modeling and bioelectric intervention.

.. contents:: Table of Contents
   :local:
   :depth: 2

Metrics: K-Index
----------------

.. automodule:: fre.metrics.k_index
   :members:
   :undoc-members:
   :show-inheritance:

The K-Index is a novel correlation-based metric for consciousness research.

Example usage:

.. code-block:: python

   from fre.metrics.k_index import k_index, bootstrap_k_ci
   import numpy as np

   # Generate data
   observed = np.abs(np.random.randn(100))
   actual = np.abs(np.random.randn(100))

   # Compute K-Index
   k = k_index(observed, actual)

   # Get bootstrap confidence interval
   ci_lower, ci_upper = bootstrap_k_ci(
       observed,
       actual,
       n_bootstrap=1000,
       confidence_level=0.95
   )

   print(f"K-Index: {k:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

Metrics: K-Lag
--------------

.. automodule:: fre.metrics.k_lag
   :members:
   :undoc-members:
   :show-inheritance:

Temporal lag analysis for time series data.

Example usage:

.. code-block:: python

   from fre.metrics.k_lag import k_lag
   import numpy as np

   # Time series data
   observed = np.abs(np.random.randn(200))
   actual = np.roll(observed, 5)  # 5-step lag

   # Analyze lags
   results = k_lag(observed, actual, max_lag=20)

   print(f"Best lag: {results['best_lag']}")
   print(f"K-Index at best lag: {results['k_at_best_lag']:.4f}")

Simulation
----------

.. automodule:: fre.simulate
   :members:
   :undoc-members:
   :show-inheritance:

Universe simulation engine.

Example usage:

.. code-block:: python

   from fre.simulate import UniverseSimulator, compute_metrics

   # Create simulator
   sim = UniverseSimulator(n_agents=50, seed=42)

   # Run simulation
   params = {"consciousness": 0.7, "coherence": 0.8, "fep": 0.5}
   for t in range(100):
       sim.step(params)

   # Compute final metrics
   metrics = compute_metrics(params, seed=42)
   print(f"Harmony: {metrics['harmony']:.4f}")

Rescue Mechanisms
-----------------

.. automodule:: fre.rescue
   :members:
   :undoc-members:
   :show-inheritance:

Bioelectric rescue for consciousness collapse.

Example usage:

.. code-block:: python

   from fre.rescue import (
       detect_consciousness_collapse,
       apply_bioelectric_rescue
   )

   # Detect collapse
   fep_error = 0.7
   coherence = 0.3
   collapse = detect_consciousness_collapse(fep_error, coherence)

   if collapse:
       # Apply rescue
       correction = apply_bioelectric_rescue(
           current_voltage=-50.0,
           target_voltage=-70.0,
           fep_error=fep_error,
           momentum=0.9
       )
       print(f"Correction: {correction:.2f} mV")

Analysis Tools
--------------

.. automodule:: fre.analyze
   :members:
   :undoc-members:
   :show-inheritance:

Batch experiment analysis.

Example usage:

.. code-block:: python

   from fre.analyze import load_logs, aggregate_metrics
   from pathlib import Path

   # Load all experiment logs
   records = load_logs(Path("logs"))

   # Aggregate metrics
   aggregated = aggregate_metrics(records)
   print(f"Mean K-Index: {aggregated['k_index']['mean']:.4f}")
