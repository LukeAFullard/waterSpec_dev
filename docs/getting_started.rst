***************
Getting Started
***************

This guide will walk you through installing `waterSpec` and running your first analysis.

Installation
============

This package is not yet on PyPI. To install it, clone this repository and install it in editable mode using pip.

For standard use:

.. code-block:: bash

   git clone https://github.com/example/waterSpec.git
   cd waterSpec
   pip install -e .

For development (to run the tests):

.. code-block:: bash

   # After cloning and changing directory:
   pip install -e '.[test]'

Quick Start
===========

The recommended workflow is centered around the ``waterSpec.Analysis`` object. You can run a complete analysis, generating a plot and a detailed text summary, with just a few lines of code.

.. code-block:: python

   from waterSpec import Analysis

   # 1. Define the path to your data file
   file_path = 'examples/sample_data.csv'

   # 2. Create the analyzer object
   # This loads and preprocesses the data immediately.
   analyzer = Analysis(
       file_path=file_path,
       time_col='timestamp',
       data_col='concentration',
       param_name='Nitrate Concentration at Site A' # A descriptive name for plots
   )

   # 3. Run the full analysis
   # This command runs the analysis, saves the outputs, and returns the results.
   results = analyzer.run_full_analysis(output_dir='example_output')

   # The summary text is available in the returned dictionary
   print(results['summary_text'])

This will produce a plot (``example_output/Nitrate_Concentration_at_Site_A_spectrum_plot.png``) and a text summary (``example_output/Nitrate_Concentration_at_Site_A_summary.txt``).