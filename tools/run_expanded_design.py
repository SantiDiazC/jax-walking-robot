"""Simulate/plot an expanded design YAML, including explicit fixed drive frequency."""
import argparse
from pathlib import Path
import jax
from walking_robot.expanded_design import ExpandedDesignRun
from walking_robot.workflow import run_design

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('design',type=Path)
parser.add_argument('--output',type=Path,default=Path('outputs/expanded_design_run'))
parser.add_argument('--render-only',action='store_true')
parser.add_argument('--show',action='store_true')
args = parser.parse_args()
jax.config.update('jax_enable_x64',True)
run_design(ExpandedDesignRun.from_file(args.design),args.output,render_only=args.render_only,show=args.show)
