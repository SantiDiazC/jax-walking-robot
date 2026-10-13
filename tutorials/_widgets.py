"""Plotting helpers for the step-by-step notebooks.

These views use the solver's geometry and recorded states. Display sampling
only reduces plotting work; physical checks are performed by the run commands.
"""
import base64
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import IFrame, display

from walking_robot.geometry import points, point_velocities


def animation_view(filename):
    """Embed a self-contained HTML animation regardless of the notebook URL."""
    encoded = base64.b64encode(Path(filename).read_bytes()).decode("ascii")
    return IFrame(src="data:text/html;base64," + encoded, width="100%", height=550)


def draw_robot(ax, state, parameters, *, follow=True):
    """Draw one recorded state in millimetres with the ground at y=0."""
    geometry = points(np.asarray(state), np.asarray(parameters))
    body, joints, feet = [geometry[key] * 1000 for key in ("body", "joints", "feet")]
    if not all(np.isfinite(value).all() for value in (body, joints, feet)):
        ax.text(.5, .5, "Nonfinite state: inspect the validation report", ha="center",
                transform=ax.transAxes)
        return
    ax.plot(body[:, 0], body[:, 1], "o-", color="tab:blue", label="Body")
    for index in range(2):
        leg = np.stack([joints[index], feet[index]])
        ax.plot(leg[:, 0], leg[:, 1], "o-", color="tab:orange",
                label="Legs" if index == 0 else None)
    ax.axhline(0, color="black", label="Ground")
    ax.axhspan(-.5, 0, color="gray", alpha=.15)
    if follow:
        center = float(body[:, 0].mean())
        span = max(float(np.ptp(np.concatenate([body[:, 0], feet[:, 0]]))) * .6, 15)
        ax.set_xlim(center - span, center + span)
    ax.set(xlabel="Horizontal position (mm)", ylabel="Height (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.2)
    ax.legend(loc="upper right")


def preview_design(design):
    """Show the supported initial pose using the same initialization as a run."""
    import jax
    from walking_robot.initial_pose import solve_initial_pose

    parameters = design.parameters()
    state = jax.jit(solve_initial_pose)(parameters)
    fig, ax = plt.subplots(figsize=(9, 3))
    draw_robot(ax, state, parameters)
    ax.set_title(f"Supported initial pose · applied frequency {float(parameters[22]):g} Hz")
    fig.tight_layout()
    display(fig)
    plt.close(fig)


def show_checks(report):
    """Summarize the full-resolution validation report, retaining failures."""
    print("Physical screening:", "PASS" if report["passed"] else "FAIL")
    for violation in report["violations"]:
        print("  Exceeded:", violation)
    metrics = report["metrics"]
    for key, label, scale, unit in [
        ("late_mean_com_speed_m_s", "Mean COM speed, final half", 1, "m/s"),
        ("max_foot_penetration_m", "Maximum foot penetration", 1000, "mm"),
        ("max_center_speed_m_s", "Maximum segment-center speed", 1, "m/s"),
    ]:
        value = metrics.get(key)
        print(f"{label}: {value * scale:.6g} {unit}" if value is not None else f"{label}: unavailable")
    print("Provisional foot-penetration limit:", report["limits"]["max_foot_penetration_m"] * 1000, "mm")


def trajectory_view(filename, *, foot_limit_m=.00025):
    """Return a time scrubber with robot pose, COM speed and foot heights.

    Accept both single-design and sweep NPZ layouts. At most 2,000 recorded
    samples are drawn; the displayed time always comes from the selected state.
    """
    with np.load(Path(filename), allow_pickle=False) as saved:
        single = "t" in saved.files
        times = saved["t" if single else "times"].copy()
        states = saved["x" if single else "states"].copy()
        parameters = saved["P" if single else "parameters"].copy()
    indices = np.unique(np.linspace(0, len(times) - 1, min(len(times), 2000), dtype=int))
    times, states = times[indices], states[indices]
    geometry = points(states, parameters)
    velocity = point_velocities(states, parameters)
    com_speed = (velocity["centers"][..., 0] * (parameters[:4] / parameters[:4].sum())).sum(axis=1)
    slider = widgets.IntSlider(value=0, min=0, max=len(times)-1, description="Frame",
                               continuous_update=False, layout=widgets.Layout(width="75%"))
    follow = widgets.Checkbox(value=True, description="Follow robot")

    def render(frame, follow_robot):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1, 1]})
        draw_robot(axes[0], states[frame], parameters, follow=follow_robot)
        if not follow_robot and np.isfinite(geometry["body"]).all():
            x = geometry["body"][..., 0] * 1000
            axes[0].set_xlim(float(x.min()) - 5, float(x.max()) + 5)
        axes[0].set_title(f"Recorded motion · t = {times[frame]:.6f} s")
        axes[1].plot(times, com_speed)
        axes[1].set(ylabel="COM speed (m/s)")
        axes[2].plot(times, geometry["feet"][:, 0, 1] * 1000, label="Front foot")
        axes[2].plot(times, geometry["feet"][:, 1, 1] * 1000, label="Hind foot")
        axes[2].axhline(0, color="black", linewidth=1)
        axes[2].axhline(-foot_limit_m * 1000, color="red", linestyle="--", label="Penetration limit")
        axes[2].set(xlabel="Time (s)", ylabel="Foot height (mm)")
        axes[2].legend(loc="upper right")
        for ax in axes[1:]:
            ax.axvline(times[frame], color="tab:orange")
            ax.grid(alpha=.2)
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    plot = widgets.interactive_output(render, {"frame": slider, "follow_robot": follow})
    return widgets.VBox([widgets.HBox([slider, follow]), plot])
