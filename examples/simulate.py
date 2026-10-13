"""Run from an installed checkout: python examples/simulate.py --output outputs/run.npz."""
import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np

from walking_robot import calculate_P_from_meta_jax, solve_initial_pose, rollout_scan
from walking_robot.validation import PhysicalLimits, inspect_trajectory


def main():
    parser = argparse.ArgumentParser(description="Simulate one robot design.")
    parser.add_argument("--zeta", type=float, default=0.01)
    parser.add_argument("--had", type=float, default=1e-6, help="Adhesive thickness in meters.")
    parser.add_argument("--volt", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("outputs/trajectory.npz"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--limits", type=Path, help="JSON object overriding provisional physical limits.")
    args = parser.parse_args()
    if args.dt <= 0 or args.steps < 1 or args.zeta < 0 or args.had < 0:
        parser.error("dt and steps must be positive; zeta and had must be nonnegative.")

    jax.config.update("jax_enable_x64", True)
    parameters = calculate_P_from_meta_jax(args.zeta, args.had, args.volt)
    x0 = jax.jit(solve_initial_pose)(parameters)
    run = jax.jit(lambda x, p: rollout_scan(x, p, dt=args.dt, steps=args.steps))
    start = time.perf_counter()
    ts, xs = run(x0, parameters)
    xs.block_until_ready()
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    ts, xs = run(x0, parameters)
    xs.block_until_ready()
    run_seconds = time.perf_counter() - start
    limits = PhysicalLimits(**json.loads(args.limits.read_text())) if args.limits else PhysicalLimits()
    report = inspect_trajectory(np.asarray(ts), np.asarray(xs), np.asarray(parameters), limits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, t=np.asarray(ts), x=np.asarray(xs),
                        P=np.asarray(parameters), meta=[args.zeta, args.had, args.volt])
    print(f"Devices: {jax.devices()}")
    print(f"Compile + first rollout: {first_seconds:.3f}s; subsequent rollout: {run_seconds:.3f}s")
    print(f"Simulated {args.steps * args.dt:g}s; saved {args.output}")
    args.output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if not report["passed"]:
        raise RuntimeError(f"Physical screening failed: {report['violations']}. Saved trajectory and report for diagnosis.")
    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, sharex=True)
        for ax, index, label in zip(axes, [0, 1], ["x (m)", "y (m)"]):
            ax.plot(ts, xs[:, index])
            ax.set_ylabel(label)
            ax.grid(True)
        axes[-1].set_xlabel("Time (s)")
        plt.show()


if __name__ == "__main__":
    main()
