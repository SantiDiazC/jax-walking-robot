# Physical model

The planar piezoelectric robot has four body segments, two legs and compliant
contact with a horizontal floor. Original MATLAB code is under
`reference/matlab/`; the port is under `src/walking_robot/`.

The 14-element state contains the first segment's horizontal/vertical position,
four segment angles, their six velocities and two friction-memory states.
Geometry reconstructs body vertices, leg attachments, feet and COM.
The 42-element parameter vector contains mass/inertia, geometry, structural
properties, drive and contact settings.

Each time step assembles and solves a 12 × 12 analytical system for accelerations
and reaction forces, retaining the inherited trace-scaled regularizer. Fixed-step
RK4 advances the state. Independent trajectories can be vectorized; consecutive
time steps within a trajectory remain sequential.

Directional Karnopp/Stribeck friction includes sticking/sliding transitions and
memory. Contact is piecewise, so smooth gradients are not guaranteed across
transitions. Smooth-contact teachers are used for experimental neural training;
accepted physical proposals are checked with original hard contact.

`profile: matlab` reconciles the original piezo torque helper call, attachment
offsets and separate front/hind contact stiffnesses. Always match profile,
initial state, frequency and objective interval when comparing results.
See [validation](validation.md) for the recorded checks and their scope.
