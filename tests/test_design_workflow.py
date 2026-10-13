"""Check configuration units, reuse safeguards and unhidden playback failures."""
from dataclasses import replace
import json
import subprocess
import numpy as np
import pytest
import jax
from walking_robot import calculate_P_from_meta_jax, solve_initial_pose
from walking_robot.design import DesignRun, RobotDesign, SimulationSettings, VisualizationSettings
from walking_robot.visualization import animation_frames, render_trajectory
from walking_robot.workflow import run_design


def test_yaml_units_and_comments(tmp_path):
    path = tmp_path/"robot.yaml"
    path.write_text("design:\n  adhesive_thickness_um: 2.0 # micrometers\n  front_leg_offset_mm: 2.8\n")
    design = DesignRun.from_file(path).design
    expected = calculate_P_from_meta_jax(.01, 2e-6, 200.)
    expected = expected.at[17].set(.0028)
    np.testing.assert_allclose(design.parameters(), expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("text", ["design:\n  voltgae_v: 200\n", "design: [1, 2]\n", "design: {voltage_v: .nan}",
                                      "simulation: {duration_s: 0.1, timestep_s: 0.03}",
                                      "visualization: {formats: [avi]}", "design: ["])
def test_bad_yaml_fails_before_simulation(tmp_path, text):
    path = tmp_path/"bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        DesignRun.from_file(path)


def test_animation_time_sampling_and_failure_policy():
    times = np.arange(1001)*1e-4
    full = animation_frames(times, 1001, 100, max_frames=600, fps=30,
                            playback_speed=.01, stop_at_violation=False)
    assert len(full) == 301
    assert full[0] == 0 and full[-1] == 1000
    stopped = animation_frames(times, 1001, 100, max_frames=600, fps=30,
                               playback_speed=.01, stop_at_violation=True)
    assert stopped[-1] == 100
    finite = animation_frames(times, 500, 500, max_frames=20, fps=30, stop_at_violation=False)
    assert finite[-1] == 499


@pytest.fixture(scope="module")
def initial():
    p = RobotDesign().parameters()
    return np.asarray(p), np.asarray(jax.jit(solve_initial_pose)(p))


def test_full_movie_keeps_failed_samples_and_report(tmp_path, initial):
    pytest.importorskip("matplotlib")
    pil = pytest.importorskip("PIL.Image")
    p, initial_state = initial
    states = np.tile(initial_state, (5, 1))
    states[2:,1] -= .001
    report = render_trajectory(np.arange(5)*1e-4, states, p, tmp_path, max_frames=5,
                               formats=("gif", "html"), fps=10, stop_at_violation=False)
    assert not report["passed"] and report["first_violation_index"] == 2
    animation = json.loads((tmp_path/"animation.json").read_text())
    assert animation["last_sample_index"] == 4 and not animation["physical_checks_passed"]
    with pil.open(tmp_path/"robot.gif") as movie:
        assert movie.n_frames == 5
        movie.seek(4)  # Last frame is readable, even though the physical screen failed.
    for name in ["robot.png", "motion.png", "diagnostics.png", "robot.html"]:
        assert (tmp_path/name).stat().st_size > 100


def test_no_nonfinite_movie_is_invented(tmp_path, initial):
    pytest.importorskip("matplotlib")
    p, _ = initial
    report = render_trajectory([0., .01], np.full((2,14), np.nan), p, tmp_path, formats=("gif",))
    assert not report["passed"]
    assert json.loads((tmp_path/"animation.json").read_text())["frame_count"] == 0
    assert not (tmp_path/"robot.gif").exists()


def test_mp4_roundtrip_preserves_frame_colors(tmp_path, initial):
    pytest.importorskip("matplotlib")
    from matplotlib.animation import writers
    if not writers.is_available("ffmpeg"):
        pytest.skip("Optional ffmpeg not installed.")
    p, state = initial
    render_trajectory(np.arange(3)*1e-4, np.tile(state,(3,1)), p, tmp_path, formats=("mp4",), max_frames=3)
    metadata = json.loads((tmp_path/"animation.json").read_text())
    frame = subprocess.run([metadata["video_encoder"], "-v", "error", "-i", str(tmp_path/"robot.mp4"),
                            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                           check=True, capture_output=True).stdout
    assert len(frame) == 1000*400*3
    # This catches a real encoder/color-conversion failure seen during testing.
    assert min(frame[:3]) >= 240


@pytest.mark.parametrize("change", ["design", "duration"])
def test_render_only_rejects_a_different_design_or_simulation(tmp_path, initial, change):
    p, state = initial
    config = DesignRun(simulation=SimulationSettings(duration_s=.0002))
    np.savez_compressed(tmp_path/"trajectory.npz", t=np.arange(3)*1e-4, x=np.tile(state,(3,1)), P=p)
    config = replace(config, simulation=SimulationSettings(duration_s=.0002, timestep_s=.0001))
    if change == "design":
        config = replace(config, design=RobotDesign(adhesive_thickness_um=2.))
    else:
        config = replace(config, simulation=SimulationSettings(duration_s=.0003, timestep_s=.0001))
    with pytest.raises(ValueError, match="different design or simulation"):
        run_design(config, tmp_path, render_only=True)
    assert not (tmp_path/"run.json").exists()
