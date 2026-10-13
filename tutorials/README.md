# Tutorials

Activate the Python environment you will use as the notebook kernel, then run
`python -m pip install -e ".[analysis,notebooks]"` from the repository root.
This installs the package and its interactive controls in that environment.
Launch `python -m jupyterlab`, or open the notebooks in VS Code and select that
environment with **Select Kernel → Python Environments**. Restart the kernel
after installing dependencies. See [notebook setup](../docs/installation.md#notebook-kernel).
Run the cells in order; each step explains its inputs, action and results.

1. `01_run_a_design.ipynb`: choose design parameters with sliders, preview the
   initial geometry, save YAML and simulate. Inspect the motion with a time
   slider, read physical checks and play the HTML animation.
2. `02_compare_designs.ipynb`: select damping/thickness combinations, preview the
   grid and run the sweep. Compare speed and penetration maps, explore individual
   cases and animate the verified finalist. A final step runs a MATLAB comparison.
3. `03_nerd_optimization.ipynb`: vary the target speed to explore the cost, compare
   recorded neural predictions with solver verification, and inspect the
   physical-search candidate. The last step provides the training workflow.

Design controls update previews. Execute the simulation or sweep cell to compute
new trajectories. Each execution saves a fresh experiment under ignored
`outputs/`, together with its input YAML. The time slider displays sampled
recorded states; physical reports retain checks at every integration step.

`_widgets.py` provides the shared robot drawing and trajectory views. Notebook
files are committed with empty outputs. Commands, units and validation scope
are explained under `docs/`.
