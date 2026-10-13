# Tutorials

Install `.[analysis,notebooks]`, launch `jupyter lab` from the repository root
and select the Python environment containing this package.

1. `01_run_a_design.ipynb`: edit YAML, simulate, inspect physical checks and plots.
2. `02_compare_designs.ipynb`: run 25 designs, read the CSV/maps, refine finalists
   and optionally compare a selected design in MATLAB.
3. `03_nerd_optimization.ipynb`: understand recorded neural predictions, a speed
   cost, failed proposals and independent solver verification. Training is off
   by default and requires the optional learning setup.

The first two tutorials require neither AI packages nor a GPU. Generated files
go under ignored `outputs/`. Notebook files are committed without bulky outputs.
Commands, units and validation scope are explained under `docs/`.
