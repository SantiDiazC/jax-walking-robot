"""Timestep sensitivity and physical screening, without changing model equations."""
import argparse
import json
from pathlib import Path
import time
import jax
import numpy as np
from walking_robot import calculate_P_from_meta_jax, solve_initial_pose, rollout_scan
from walking_robot.validation import PhysicalLimits, inspect_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=float, default=.03)
    parser.add_argument("--timesteps", type=float, nargs="+", default=[1e-5, 5e-6, 2.5e-6])
    parser.add_argument("--output", type=Path, default=Path("outputs/physics_validation"))
    parser.add_argument("--limits", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.horizon) or args.horizon <= 0 or len(set(args.timesteps)) < 2 or any(not np.isfinite(dt) or dt <= 0 for dt in args.timesteps):
        parser.error("Positive horizon and at least two positive timesteps required.")
    limits = PhysicalLimits(**json.loads(args.limits.read_text())) if args.limits else PhysicalLimits()
    jax.config.update("jax_enable_x64", True)
    p = calculate_P_from_meta_jax(.01, 1e-6, 200.)
    x0 = jax.jit(solve_initial_pose)(p)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for dt in sorted(set(args.timesteps), reverse=True):
        steps = round(args.horizon/dt)
        if not np.isclose(steps*dt, args.horizon, atol=1e-12, rtol=0):
            parser.error("Horizon must be an integer multiple of each timestep.")
        run = jax.jit(lambda x, p: rollout_scan(x, p, dt=dt, steps=steps))
        start = time.perf_counter()
        t, x = run(x0, p)
        x.block_until_ready()
        report = inspect_trajectory(t, x, p, limits)
        report.update(dt_s=dt, horizon_s=steps*dt, compile_and_run_s=time.perf_counter()-start)
        results.append(report)
        np.savez_compressed(args.output/f"dt_{dt:g}.npz", t=t, x=x, P=p)
        print(json.dumps(report), flush=True)
    finest = results[-1]["metrics"]
    comparisons = []
    def difference(a, b, name):
        values = [a.get(name), b.get(name)]
        if any(v is None or not np.isfinite(v) for v in values):
            return None
        return abs(values[0]-values[1])
    for result in results[:-1]:
        m = result["metrics"]
        speed_error = difference(m, finest, "mean_com_speed_m_s")
        penetration_error = difference(m, finest, "max_foot_penetration_m")
        comparisons.append({"dt_s": result["dt_s"], "mean_speed_difference_m_s": speed_error,
                            "max_penetration_difference_m": penetration_error,
                            "passed_provisional_timestep_check": bool(speed_error is not None and penetration_error is not None and speed_error <= .01 and penetration_error <= 2e-5)})
    summary = {"cases": results, "compared_to_dt_s": results[-1]["dt_s"], "comparisons": comparisons,
               "all_physical_checks_passed": all(r["passed"] for r in results),
               "scope": "One design over the stated horizon; aggregate agreement is not proof of gradient or full trajectory convergence."}
    (args.output/"report.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
