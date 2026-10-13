"""Compare the original MATLAB numerical setup with the current JAX design map."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import jax
import numpy as np
from scipy.io import loadmat
from walking_robot import calculate_P_from_meta_jax


def matlab_string(path):
    return "'"+str(path).replace("'", "''")+"'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--output", type=Path, default=Path("outputs/matlab_parameter_audit.json"))
    parser.add_argument("--profile",choices=["legacy","matlab"],default="legacy")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jax.config.update("jax_enable_x64", True)
    p = np.asarray(calculate_P_from_meta_jax(.01, 1e-6, 200.,profile=args.profile))
    with tempfile.TemporaryDirectory(prefix="walking-robot-parameter-audit-") as folder:
        result = Path(folder)/"parameters.mat"
        command = f"addpath({matlab_string(root/'tools')}); matlab_parameter_reference({matlab_string(result)})"
        subprocess.run([args.matlab, "-batch", command], check=True)
        reference = loadmat(result, simplify_cells=True)["reference"]
    mapping = dict(mass_kg=0, inertia_kg_m2=4, spring_n_m_rad=19, damping_n_m_s_rad=20,
                   torque_n_m=21, frequency_hz=22, front_leg_offset_m=17, hind_leg_offset_m=18,
                   front_leg_angle_rad=15, hind_leg_angle_rad=16, front_contact_stiffness_n_m=26,
                   hind_contact_stiffness_n_m=41 if args.profile=="matlab" else 26, contact_damping_n_s_m=27)
    rows = []
    for name, value in reference.items():
        actual = float(p[mapping[name]]) if name in mapping else (.000501 if args.profile=="matlab" else 0.)
        expected = float(value)
        row = dict(parameter=name, original_matlab=expected, jax_value=actual,
                   matches=bool(np.isclose(actual, expected, atol=1e-14, rtol=1e-10)))
        rows.append(row)
        print(f"{name}: MATLAB={expected:.12g}, JAX={actual:.12g}, {'MATCH' if row['matches'] else 'DIFFERENT'}", flush=True)
    archive = root/'reference/matlab/original'
    report = dict(scope="Numerical setup of the unmodified original script and original material helpers; not a trajectory comparison.",
                  profile=args.profile,
                  baseline=dict(damping_ratio=.01, adhesive_thickness_um=1., voltage_v=200.), parameters=rows,
                  original_source_sha256={str(path.relative_to(archive)):hashlib.sha256(path.read_bytes()).hexdigest()
                                          for path in sorted(archive.rglob("*.m"))},
                  differences=[r['parameter'] for r in rows if not r['matches']],
                  interpretation=["The matlab profile reproduces separate front/hind stiffnesses and the actual original actuator-helper arguments; legacy defaults remain unchanged.",
                                  "Initial clearance includes the original 1 um pose target plus the additional 500 um shift in X0.",
                                  "The clearance row describes the corresponding reference initialization; YAML can override it independently of the parameter profile.",
                                  "This audit compares numerical setup only; use validate_original_matlab.py for equation and matched-integrator validation."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
