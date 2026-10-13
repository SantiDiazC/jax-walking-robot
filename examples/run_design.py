"""Generate every artifact from one editable robot design."""
import argparse
from dataclasses import replace
from pathlib import Path
import jax
from walking_robot.design import DesignRun
from walking_robot.workflow import run_design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=Path(__file__).parent/"designs/baseline.yaml")
    parser.add_argument("--output", type=Path, default=Path("outputs/design_baseline"))
    parser.add_argument("--formats", nargs="+", choices=["html", "gif", "mp4"])
    parser.add_argument("--duration", type=float, help="Override simulated duration in seconds.")
    parser.add_argument("--render-only", action="store_true", help="Reuse the matching trajectory in the output directory.")
    parser.add_argument("--show", action="store_true", help="Also open the Matplotlib animation; space pauses/resumes.")
    args = parser.parse_args()
    try:
        configuration = DesignRun.from_file(args.design)
        if args.formats:
            configuration = replace(configuration, visualization=replace(configuration.visualization, formats=tuple(args.formats)))
        if args.duration is not None:
            configuration = replace(configuration, simulation=replace(configuration.simulation, duration_s=args.duration))
        jax.config.update("jax_enable_x64", True)
        report = run_design(configuration, args.output, render_only=args.render_only, show=args.show)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved trajectory, figures, animation and validation in {args.output}", flush=True)
    if not report["passed"]:
        parser.exit(2, "Physical limits exceeded; artifacts retained for inspection.\n")


if __name__ == "__main__":
    main()
