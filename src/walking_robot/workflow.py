"""One design -> compiled dynamics -> recorded trajectory, figures and movie."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import jax
import numpy as np
import yaml
from .initial_pose import solve_initial_pose
from .integrators import rollout_scan
from .validation import inspect_trajectory
from .visualization import render_trajectory

_initial_pose = jax.jit(solve_initial_pose)
_rollout = jax.jit(rollout_scan, static_argnames=("dt", "steps", "method"))


def run_design(configuration, output, *, render_only=False, show=False):
    """Run once and write all artifacts. Caller enables JAX float64 first.

    Failed physical checks are reported without suppressing figures or movies.
    render_only verifies design and simulation settings before reusing a file.
    The same compiled rollout can be reused for subsequent design values.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before running a design.")
    output = Path(output)
    settings = configuration.simulation
    p = configuration.design.parameters()
    initial = _initial_pose(p).at[1].add(settings.initial_clearance_um*1e-6)
    trajectory_path = output/"trajectory.npz"
    elapsed = None
    if render_only:
        with np.load(trajectory_path, allow_pickle=False) as saved:
            t, x, recorded_p = saved["t"], saved["x"], saved["P"]
        expected_times = np.arange(settings.steps+1)*settings.timestep_s
        if (recorded_p.shape != (42,) or not np.allclose(recorded_p, np.asarray(p), rtol=1e-12, atol=1e-14)
                or t.shape != expected_times.shape or not np.allclose(t, expected_times, rtol=0, atol=1e-12)
                or not np.allclose(x[0], np.asarray(initial), rtol=1e-10, atol=1e-12)):
            raise ValueError("Saved trajectory belongs to a different design or simulation; rerun without --render-only.")
    else:
        print(f"Simulating {settings.duration_s:g} s ({settings.steps} RK4 steps) on {jax.devices()}...", flush=True)
        start = time.perf_counter()
        t, x = _rollout(initial, p, dt=settings.timestep_s, steps=settings.steps, method="rk4")
        x.block_until_ready()
        elapsed = time.perf_counter()-start
        t, x = np.asarray(t), np.asarray(x)
    output.mkdir(parents=True, exist_ok=True)
    report = inspect_trajectory(t, x, p, configuration.physical_limits)
    if not render_only:
        np.savez_compressed(trajectory_path, t=t, x=x, P=np.asarray(p),
                            configuration_json=json.dumps(asdict(configuration), sort_keys=True))
    (output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (output/"design.yaml").write_text(yaml.safe_dump(asdict(configuration), sort_keys=False))
    # Save the physical result before rendering, including when an encoder fails.
    metadata = dict(configuration=asdict(configuration), jax_version=jax.__version__,
                    devices=[str(device) for device in jax.devices()], precision="float64",
                    compile_and_simulate_s=elapsed, reused_trajectory=render_only,
                    physical_checks_passed=report["passed"], artifacts=["trajectory.npz", "validation.json", "design.yaml"],
                    trajectory_sha256=hashlib.sha256(trajectory_path.read_bytes()).hexdigest(), rendering_complete=False)
    manifest = output/"run.json"
    manifest.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    visualization = configuration.visualization
    print(f"Physical screening: {'PASS' if report['passed'] else 'FAIL '+', '.join(report['violations'])}. Rendering...", flush=True)
    render_trajectory(t, x, p, output, configuration.physical_limits,
                      max_frames=visualization.max_frames, formats=visualization.formats,
                      fps=visualization.fps, playback_speed=visualization.playback_speed,
                      stop_at_violation=visualization.stop_at_violation, show=show)
    metadata["artifacts"] += [name for name in ["diagnostics.png", "motion.png", "robot.png", "animation.json"] if (output/name).exists()]
    metadata["artifacts"] += [f"robot.{extension}" for extension in dict.fromkeys(visualization.formats)
                              if (output/f"robot.{extension}").exists()]
    metadata["rendering_complete"] = True
    manifest.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    return report
