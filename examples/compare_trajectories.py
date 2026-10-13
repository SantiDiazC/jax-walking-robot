"""Compare a physical trajectory and a neural rollout, including failed states."""
import argparse
import json
from pathlib import Path
import numpy as np
from walking_robot.geometry import points, point_velocities
from walking_robot.validation import PhysicalLimits, inspect_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison.png"))
    parser.add_argument("--limits", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    limits = PhysicalLimits(**json.loads(args.limits.read_text())) if args.limits else PhysicalLimits()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for path, label, style in [(args.reference, "JAX", "-"), (args.prediction, "NeRD", "--")]:
        with np.load(path, allow_pickle=False) as data:
            t, x, p = data["t"], data["x"], data["P"]
        report = inspect_trajectory(t, x, p, limits)
        finite = np.isfinite(x).all(axis=1)
        g, v = points(x[finite], p), point_velocities(x[finite], p)
        for i, foot in enumerate(["front", "hind"]):
            axes[0].plot(t[finite], g["feet"][:, i, 1]*1000, style, label=f"{label} {foot}")
        com_x = (g["centers"][..., 0]*(p[:4]/p[:4].sum())).sum(axis=1)
        com_vx = (v["centers"][..., 0]*(p[:4]/p[:4].sum())).sum(axis=1)
        axes[1].plot(t[finite], (com_x-com_x[0])*1000, style, label=label)
        axes[2].plot(t[finite], com_vx, style, label=label)
        if report["first_violation_time_s"] is not None:
            for ax in axes:
                ax.axvline(report["first_violation_time_s"], ls=":", alpha=.5)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].axhline(-limits.max_foot_penetration_m*1000, color="red", ls=":", label="Penetration limit")
    for ax, ylabel in zip(axes, ["Foot height (mm)", "COM displacement (mm)", "COM velocity (m/s)"]):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Full rollouts: failed states retained for diagnosis")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
