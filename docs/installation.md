# Installation

Use Python 3.11 and run commands from the repository root.

```bash
# In the existing environment: conda activate jaxgpu
# Or create one: python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[analysis,notebooks,test]"
```

`analysis` supplies Matplotlib and SciPy for plots and MATLAB exchange;
`notebooks` supplies JupyterLab, the Python kernel and ipywidgets controls;
`test` supplies pytest. JAX, NumPy and PyYAML are the core dependencies.

Check the installation in this environment:

```bash
python -c "import sys, jax, ipywidgets, walking_robot; print(sys.executable); print('JAX:', jax.__version__); print('ipywidgets:', ipywidgets.__version__)"
```

Alternatively, run `conda env create -f environment.yml`, then
`conda activate walking-robot`. The provided environment includes CPU tools and
notebooks. Keep using the existing `jaxgpu` environment if it already has the
required CUDA installation.

The environment name does not select the device. `JAX_PLATFORMS=cpu` selects
CPU; `JAX_PLATFORMS=cuda` requests a working CUDA-enabled JAX installation.
The sweep runner instead accepts `--backend cpu` or `--backend cuda`.

For GPU installation, follow the
[official JAX instructions](https://docs.jax.dev/en/latest/installation.html)
for your CUDA driver. Check the devices with:

```bash
python -c "import jax; print(jax.devices())"
```

Scientific examples enable float64. Changing to float32 needs separate
numerical validation. See [performance](performance.md).

## Notebook kernel

The notebook kernel runs Python in its selected environment. Install the package
with the `notebooks` extra in that same environment, even when Jupyter or VS Code
is already installed elsewhere.

For JupyterLab, run `python -m jupyterlab` from the repository root in the
environment used for installation and open the numbered notebooks in `tutorials/`.

For VS Code, install the Python and Jupyter extensions, open a notebook, and use
**Select Kernel → Python Environments** to select the environment used above
(for example, `jaxgpu`). Restart the kernel after installing dependencies, then
run the cells in order. See the [VS Code notebook guide](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

The setup cell prints its Python executable. If you see
`ModuleNotFoundError: No module named 'ipywidgets'`, check that executable against
the terminal's `python -c "import sys; print(sys.executable)"`. Activate the kernel's
environment and rerun `python -m pip install -e ".[analysis,notebooks]"` from the
repository root, then restart the kernel. Installing only `.[analysis,test]`
provides simulation and test tools but omits the notebook controls. See the
[ipywidgets installation guide](https://ipywidgets.readthedocs.io/en/stable/user_install.html)
for separate server and kernel environments.

To execute the first tutorial with a kernel using the current Python interpreter:

```bash
python tools/check_notebooks.py
```

The check runs the full tutorial and retains generated artifacts under `outputs/`;
it leaves the source notebook's outputs empty. CI runs this check after installing
the notebook dependencies in its own environment.

HTML animations need no encoder;
MP4 requires FFmpeg with H.264 support. Neural training additionally requires
`.[learning]` and the official NeRD source; see [optimization](optimization.md).
The recorded neural experiments used PyTorch 2.9.1 and JAX 0.8.2.
