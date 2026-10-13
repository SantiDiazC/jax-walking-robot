"""Render a saved simulation with physical screening and MATLAB geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
from walking_robot.validation import PhysicalLimits
from walking_robot.visualization import render_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/visualization"))
    parser.add_argument("--limits", type=Path, help="JSON object of PhysicalLimits overrides.")
    parser.add_argument("--formats", nargs="+", choices=["html", "gif", "mp4"], default=["html"])
    parser.add_argument("--full", action="store_true", help="Show the complete finite trajectory and label physical violations.")
    parser.add_argument("--show", action="store_true", help="Open the Matplotlib animation; space pauses/resumes.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--playback-speed", type=float)
    args = parser.parse_args()
    limits = PhysicalLimits(**json.loads(args.limits.read_text())) if args.limits else PhysicalLimits()
    with np.load(args.trajectory, allow_pickle=False) as data:
        report = render_trajectory(data["t"], data["x"], data["P"], args.output, limits,
                                   max_frames=args.max_frames, formats=args.formats, fps=args.fps,
                                   playback_speed=args.playback_speed, stop_at_violation=not args.full, show=args.show)
    (args.output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
