# Installation

Use Python 3.11 and run commands from the repository root.

```bash
# In the existing environment: conda activate jaxgpu
# Or create one: python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[analysis,test]"
```

`analysis` supplies Matplotlib and SciPy for plots and MATLAB exchange; `test`
supplies pytest. JAX, NumPy and PyYAML are the core dependencies.

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

```bash
python -m pip install -e ".[analysis,notebooks]"
jupyter lab
```

Open the numbered notebooks under `tutorials/`. HTML animations need no encoder;
MP4 requires FFmpeg with H.264 support. Neural training additionally requires
`.[learning]` and the official NeRD source; see [optimization](optimization.md).
The recorded neural experiments used PyTorch 2.9.1 and JAX 0.8.2.
