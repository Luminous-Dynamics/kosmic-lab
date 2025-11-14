Scripts
=======

User-facing scripts and tools.

.. contents:: Table of Contents
   :local:
   :depth: 2

Kosmic Dashboard
----------------

.. automodule:: scripts.kosmic_dashboard
   :members:
   :undoc-members:
   :show-inheritance:

Real-time monitoring dashboard built with Dash/Plotly.

Usage:

.. code-block:: bash

   # Launch dashboard
   python scripts/kosmic_dashboard.py --logdir logs --port 8050

   # Then open browser to http://localhost:8050

Or with Make:

.. code-block:: bash

   make dashboard

AI Experiment Designer
----------------------

.. automodule:: scripts.ai_experiment_designer
   :members:
   :undoc-members:
   :show-inheritance:

AI-powered experiment design and hypothesis generation.

Usage:

.. code-block:: bash

   python scripts/ai_experiment_designer.py \
       --hypothesis "Test bioelectric rescue effectiveness" \
       --output experiments/ai_designed.py

FRE Analyzer
------------

.. automodule:: scripts.fre_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

Batch analysis of FRE experiments.

Usage:

.. code-block:: bash

   python scripts/fre_analyzer.py --logdir logs --output results.json
