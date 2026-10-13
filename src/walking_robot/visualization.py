"""MATLAB-style plots and faithful playback of recorded planar trajectories."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from .geometry import points, point_velocities
from .validation import PhysicalLimits, inspect_trajectory


def _video_encoder(configured_path):
    """Prefer an H.264-capable ffmpeg, including other installed PATH entries."""
    candidates = [shutil.which(configured_path)]
    candidates += [shutil.which("ffmpeg", path=directory) for directory in os.get_exec_path()]
    fallback = None
    for candidate in dict.fromkeys(path for path in candidates if path):
        try:
            result = subprocess.run([candidate, "-hide_banner", "-encoders"], capture_output=True,
                                    text=True, check=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        if re.search(r"^\s*V\S*\s+libx264\s", result.stdout, re.M):
            return candidate, "libx264"
        if fallback is None and re.search(r"^\s*V\S*\s+mpeg4\s", result.stdout, re.M):
            fallback = candidate, "mpeg4"
    if fallback is not None:
        return fallback
    raise RuntimeError("No usable MP4 encoder found. Choose --formats gif html.")


def animation_frames(times, finite_samples, first_violation, *, max_frames, fps,
                     playback_speed=None, stop_at_violation=True):
    """Sample recorded states in physical time; never interpolate contact motion."""
    t = np.asarray(times)
    count = min(len(t), finite_samples)
    if stop_at_violation and first_violation is not None:
        count = min(count, first_violation+1)
    if count == 0:
        return np.array([], dtype=int)
    span = t[count-1]-t[0]
    requested = max_frames if playback_speed is None else max(2, int(np.ceil(span*fps/playback_speed))+1)
    desired_times = np.linspace(t[0], t[count-1], min(max_frames, requested))
    right = np.minimum(np.searchsorted(t[:count], desired_times), count-1)
    left = np.maximum(right-1, 0)
    indices = np.where(abs(t[left]-desired_times) < abs(t[right]-desired_times), left, right)
    return np.unique(indices)


def render_trajectory(times, states, parameters, output, limits=None, max_frames=160,
                      *, formats=("html",), fps=30, playback_speed=None,
                      stop_at_violation=True, show=False):
    """Write plots and HTML/GIF/MP4, optionally open a Matplotlib animation.

    Full playback labels violations and stops before nonfinite states. The
    default preserves the earlier stop-at-first-violation behavior. No state is
    clamped, repaired or resimulated by this renderer. Space pauses a live view.
    """
    if isinstance(formats, str) or not formats or not set(formats) <= {"html", "gif", "mp4"}:
        raise ValueError("Choose animation formats from html, gif and mp4.")
    if not isinstance(fps, int) or fps < 1 or not isinstance(max_frames, int) or max_frames < 2:
        raise ValueError("fps must be positive and max_frames an integer >= 2.")
    if playback_speed is not None and (not np.isfinite(playback_speed) or playback_speed <= 0):
        raise ValueError("Playback speed must be finite and positive.")
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter, writers
    if "mp4" in formats and not writers.is_available("ffmpeg"):
        raise RuntimeError("MP4 requires ffmpeg. Install it or choose --formats gif html.")
    video_encoder, video_codec = None, None
    if "mp4" in formats:
        video_encoder, video_codec = _video_encoder(matplotlib.rcParams["animation.ffmpeg_path"])
    limits = limits or PhysicalLimits()
    output = Path(output)
    t, x, p = np.asarray(times), np.asarray(states), np.asarray(parameters)
    report = inspect_trajectory(t, x, p, limits)
    output.mkdir(parents=True, exist_ok=True)
    for name in ["robot.html", "robot.gif", "robot.mp4", "robot.png", "diagnostics.png", "motion.png", "animation.json"]:
        (output/name).unlink(missing_ok=True)
    # Physical failures remain in plots. A nonfinite state ends displayed data;
    # the report retains the original number of samples and the failure.
    count = report["finite_samples"]
    xx, tt = x[:count], t[:count]
    g, v = points(xx, p), point_velocities(xx, p)
    mass_fraction = p[:4]/p[:4].sum()
    com = (g["centers"]*mass_fraction[None,:,None]).sum(axis=1)
    com_velocity = (v["centers"]*mass_fraction[None,:,None]).sum(axis=1)
    status = "PASS (provisional limits)" if report["passed"] else "FAIL: " + ", ".join(report["violations"])

    def finish_plot(fig, axes, filename):
        for ax in np.asarray(axes).flat:
            ax.grid(alpha=.25)
            ax.legend(loc="best", fontsize=8)
            if report["first_violation_time_s"] is not None:
                ax.axvline(report["first_violation_time_s"], color="red", ls=":", alpha=.6)
        fig.suptitle(status)
        fig.savefig(output/filename, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for i, label in enumerate(["Front foot", "Hind foot"]):
        axes[0].plot(tt, g["feet"][:,i,1]*1000, label=label)
        axes[1].plot(tt, v["feet"][:,i,0], label=label)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].axhline(-limits.max_foot_penetration_m*1000, color="red", ls="--", label="Penetration limit")
    axes[0].set_ylabel("Foot height (mm)")
    axes[1].set_ylabel("Foot vx (m/s)")
    for i in range(4):
        axes[2].plot(tt, xx[:,8+i], label=f"Body {i+1}")
    axes[2].set_ylabel("Angular rate (rad/s)")
    axes[2].set_xlabel("Time (s)")
    finish_plot(fig, axes, "diagnostics.png")

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for i, label in enumerate(["x", "y"]):
        axes[0].plot(tt, (com[:,i]-com[0,i])*1000 if count else [], label=label)
        axes[1].plot(tt, com_velocity[:,i], label=label)
    for i in range(3):
        axes[2].plot(tt, np.rad2deg(xx[:,3+i]-xx[:,2+i]), label=f"Joint {i+1}")
    axes[3].plot(tt, np.linalg.norm(v["centers"], axis=-1).max(axis=1), label="Maximum body-center speed")
    axes[3].plot(tt, np.linalg.norm(v["feet"], axis=-1).max(axis=1), label="Maximum foot speed")
    axes[3].axhline(limits.max_center_speed_m_s, color="C0", ls="--", label="Body-center limit")
    axes[3].axhline(limits.max_foot_speed_m_s, color="C1", ls="--", label="Foot limit")
    for ax, label in zip(axes, ["COM displacement (mm)", "COM velocity (m/s)", "Relative angle (deg)", "Speed (m/s)"]):
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Time (s)")
    finish_plot(fig, axes, "motion.png")

    stop = report["first_violation_index"]
    frames = animation_frames(t, count, stop, max_frames=max_frames, fps=fps,
                              playback_speed=playback_speed, stop_at_violation=stop_at_violation)
    span = float(t[frames[-1]]-t[frames[0]]) if len(frames) else 0.
    metadata = dict(frame_count=len(frames), fps=fps, formats=list(dict.fromkeys(formats)),
                    first_sample_index=int(frames[0]) if len(frames) else None,
                    last_sample_index=int(frames[-1]) if len(frames) else None,
                    simulated_span_s=span, movie_duration_s=len(frames)/fps,
                    requested_playback_speed=playback_speed,
                    effective_playback_speed=span*fps/(len(frames)-1) if len(frames)>1 else None,
                    stop_at_violation=stop_at_violation, physical_checks_passed=report["passed"],
                    first_violation_time_s=report["first_violation_time_s"],
                    nonfinite_tail_excluded=count < len(x), frame_sampling="Nearest recorded state; no interpolation.",
                    video_codec=video_codec, video_encoder=video_encoder)
    (output/"animation.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    if not len(frames):
        return report
    selected = {key:value[frames] for key,value in g.items()}
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_ylim(-max(.5, limits.max_foot_penetration_m*1500), limits.max_body_height_m*1000+1)
    ax.axhspan(-2, 0, color="#e3dfd5")
    ax.axhline(0, color="black", lw=1)
    ax.axhline(-limits.max_foot_penetration_m*1000, color="red", ls="--", lw=1)
    ax.grid(axis="x", alpha=.2)
    body, = ax.plot([], [], "o-", color="#2463a0", lw=3, ms=4)
    legs = [ax.plot([], [], color="#252525", lw=2)[0] for _ in range(2)]
    markers = ax.scatter([], [], s=65, zorder=5)
    centers = ax.scatter([], [], s=18, color="#2463a0")
    com_marker = ax.scatter([], [], marker="D", s=28, color="#e09f23", zorder=5)
    caption = ax.text(.02, .95, "", transform=ax.transAxes, va="top", fontsize=9)

    def update(i):
        index = frames[i]
        body.set_data(selected["body"][i,:,0]*1000, selected["body"][i,:,1]*1000)
        for k, leg in enumerate(legs):
            line = np.stack([selected["joints"][i,k], selected["feet"][i,k]])*1000
            leg.set_data(line[:,0], line[:,1])
        markers.set_offsets(selected["feet"][i]*1000)
        markers.set_color(["#c73d39" if height < 0 else "#2f9258" for height in selected["feet"][i,:,1]])
        centers.set_offsets(selected["centers"][i]*1000)
        com_marker.set_offsets(com[index:index+1]*1000)
        mid = com[index,0]*1000
        ax.set_xlim(mid-18, mid+18)
        crossed = stop is not None and index >= stop
        frame_status = f"Limit crossed at {t[stop]*1000:.2f} ms" if crossed else "Within provisional limits"
        ax.set_title(f"t={t[index]*1000:.2f} ms | {frame_status}", color="#a52626" if crossed else "black")
        penetration = max(0., -selected["feet"][i,:,1].min())*1000
        caption.set_text(f"Foot penetration: {penetration:.3f} mm / {limits.max_foot_penetration_m*1000:.3f} mm\n"
                         f"COM vx: {com_velocity[index,0]:.3f} m/s   |   Displacement: {(com[index,0]-com[0,0])*1000:.2f} mm")
        return body, *legs, markers, centers, com_marker, caption

    update(0)
    fig.savefig(output/"robot.png", dpi=150)
    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False, repeat=True)
    try:
        for extension in dict.fromkeys(formats):
            print(f"Writing {output/f'robot.{extension}'} ({len(frames)} frames)...", flush=True)
            if extension == "html":
                with matplotlib.rc_context({"animation.embed_limit":max(20.,len(frames)*.25)}):
                    (output/"robot.html").write_text(animation.to_jshtml(fps=fps, default_mode="loop"))
            elif extension == "gif":
                animation.save(output/"robot.gif", writer=PillowWriter(fps=fps), dpi=100)
            else:
                quality = ["-crf", "20"] if video_codec == "libx264" else ["-q:v", "2"]
                with matplotlib.rc_context({"animation.ffmpeg_path":video_encoder}):
                    animation.save(output/"robot.mp4", writer=FFMpegWriter(fps=fps, codec=video_codec,
                                   extra_args=["-pix_fmt", "yuv420p"]+quality), dpi=100)
        if show:
            running = True
            def key_press(event):
                nonlocal running
                if event.key == " ":
                    animation.pause() if running else animation.resume()
                    running = not running
            fig.canvas.mpl_connect("key_press_event", key_press)
            update(0)
            plt.show()
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Video encoding failed; trajectory and physical report were retained. Try --formats gif html.") from error
    finally:
        plt.close(fig)
    return report
