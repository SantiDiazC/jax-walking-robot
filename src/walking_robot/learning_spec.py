"""Shared design domain and state scales, without a neural-framework import."""

DESIGN_LOWER = (.5,90.)  # Adhesive thickness (micrometers), front-leg angle (deg).
DESIGN_UPPER = (2.,100.)
STATE_SCALES = (.025,.0025,.2,.2,.2,.2,1.,1.,500.,500.,500.,500.,1e-7,1e-7)
